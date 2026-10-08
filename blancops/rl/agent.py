import random

import numpy as np
import torch

from blancops.configs.constants import NO_FILTER_SIGNAL, WAIT_SIGNAL
from blancops.ephemerides import ephemerides
from blancops.math.interpolate import interpolate_on_sphere

import logging
from blancops.configs.enums import has_filter, is_field_level
logger = logging.getLogger(__name__)


def filter_first_decode(scores, action_mask, visible_bin_mask, num_filters):
    """Filter-first action decode over a batch.

    Choose filter by best Q over observable bins (entire sky), then the best available bin for that filter.
    Single source of truth shared by the live/offline Agent and the single-step evaluator.

    Args:
        scores: Raw network scores, [batch, n_bins * num_filters].
        action_mask: Valid-action mask, [batch, n_bins * num_filters].
        visible_bin_mask: Observable-bin mask, [batch, n_bins].
        num_filters: Number of filters (grizY for DES).

    Returns:
        Tuple of (bin_idx, filter_idx) index tensors, each [batch].
    """
    batch = scores.shape[0]
    n_bins = scores.shape[1] // num_filters
    q_map = scores.view(batch, n_bins, num_filters)
    avail = action_mask.view(batch, n_bins, num_filters).bool()
    neg_inf = torch.finfo(q_map.dtype).min

    # Filter score: best Q over observable bins, restricted to filters with at
    # least one currently-available action.
    vis_scores = q_map.masked_fill(~visible_bin_mask.unsqueeze(-1), neg_inf)
    filter_scores = vis_scores.max(dim=1).values
    filter_scores = filter_scores.masked_fill(~avail.any(dim=1), neg_inf)
    filter_idx = filter_scores.argmax(dim=1)

    # Best available bin for the chosen filter.
    gather_idx = filter_idx.view(batch, 1, 1).expand(batch, n_bins, 1)
    q_for_filter = q_map.gather(2, gather_idx).squeeze(-1)
    avail_for_filter = avail.gather(2, gather_idx).squeeze(-1)
    bin_scores = q_for_filter.masked_fill(~avail_for_filter, neg_inf)
    bin_idx = bin_scores.argmax(dim=1)
    return bin_idx, filter_idx


class Agent:
    def __init__(self, policy, cfg, lookups, field_choice_method='interp', action_decode='joint', device=None):
        self.policy = policy
        self.lookups = lookups
        self.cfg = cfg
        self.device = device if device is not None else next(policy.parameters()).device
        self.field_choice_method = field_choice_method
        self.action_decode = action_decode

    def _choose_bin_and_filter(self, x_glob, x_cand, action_mask, info, epsilon=None):
        do_filt = has_filter(self.cfg.data.action_space)
        visible_bin_mask = info.get('visible_bin_mask') if info is not None else None

        if self.action_decode == 'filter_first' and do_filt and action_mask is not None and visible_bin_mask is not None:
            return self._filter_first_decode(x_glob, x_cand, action_mask, visible_bin_mask)

        # Joint argmax over the flat (bin, filter) table (default).
        action_tensor = self.policy.select_action(x_glob=x_glob, x_cand=x_cand, action_mask=action_mask)
        action = int(action_tensor.item()) if hasattr(action_tensor, 'item') else int(action_tensor)

        if do_filt:
            bin_idx = action // self.policy.num_filters
            filter_idx = action % self.policy.num_filters
        else:
            bin_idx = action
            filter_idx = NO_FILTER_SIGNAL
        return bin_idx, filter_idx

    def _filter_first_decode(self, x_glob, x_cand, action_mask, visible_bin_mask):
        """Choose the filter over all visible bins, then the best available bin for it."""
        with torch.no_grad():
            scores = self.policy.core_net(x_glob, x_cand)
        visible = torch.as_tensor(visible_bin_mask, device=scores.device, dtype=torch.bool)
        bin_idx, filter_idx = filter_first_decode(
            scores, action_mask.view(1, -1), visible.view(1, -1), self.policy.num_filters
        )
        return int(bin_idx.item()), int(filter_idx.item())

    def _determine_valid_fields(self, bin_idx, filter_idx, info):
        """Valid given airmass / horizon / completion conditions"""
        # Unpack info and get valid fields in bin
        valid_fields_per_bin = info.get('valid_fields_per_bin', {})
        valid_fields_in_bin = np.array(valid_fields_per_bin.get(int(bin_idx), []))
        assert len(valid_fields_in_bin) != 0, f"No valid fields are in bin {bin_idx}. Check environment's output mask."

        survey_tracker = info.get('survey_progress_tracker')
        survey_counts = survey_tracker.raw_counts
        target_survey_counts = survey_tracker.target_counts
        incomplete_mask = survey_tracker.get_incomplete_mask()

        # Filter out completed fields in (bin, filter)
        field_ids_in_bin = [fid for fid in valid_fields_in_bin if survey_counts[fid, filter_idx] < target_survey_counts[fid, filter_idx]]

        assert len(field_ids_in_bin) != 0, "No valid fields are in bin...check environment's output mask."
        logger.debug(f'Chosen bin contains {len(field_ids_in_bin)} incomplete fields out of {len(valid_fields_in_bin)} fields total')
        return field_ids_in_bin

    def command_to_env_action(self, bin_idx: int, filter_idx: int, field_id: int) -> dict:
        """Environment action for a decoded choice: a physical command for field_filter, else the bin action.

        Parameters
        ----------
        bin_idx : int
            Grid bin (bookkeeping on field_filter), or WAIT_SIGNAL.
        filter_idx : int
            Filter index.
        field_id : int
            Field id.

        Returns
        -------
        dict
            {'pointing', 'filter', 'wait'} for field_filter; {'bin', 'field_id', 'filter_idx'} otherwise.
        """
        if not is_field_level(self.cfg.data.action_space):
            return {'bin': np.int32(bin_idx), 'field_id': np.int32(field_id), 'filter_idx': np.int32(filter_idx)}
        if bin_idx == WAIT_SIGNAL:
            return {'pointing': np.zeros(2), 'filter': 0, 'wait': 1}
        fields = self.lookups.fields
        return {'pointing': np.array([fields['ra'].to_numpy()[field_id], fields['dec'].to_numpy()[field_id]]),
                'filter': int(filter_idx), 'wait': 0}

    def _choose_field_filter(self, glob_tensor, cand_tensor, action_mask, info, hpGrid):
        """Field-level decode: masked argmax over (field, filter); wait when nothing is allowed.

        Returns
        -------
        tuple
            (grid bin of the field for bookkeeping or WAIT_SIGNAL, filter_idx, field_id).
        """
        if action_mask is not None and not bool(action_mask.any()):
            return WAIT_SIGNAL, 0, -1
        with torch.no_grad():
            action = int(self.policy.select_action(x_glob=glob_tensor, x_cand=cand_tensor, action_mask=action_mask))
        field_id, filter_idx = divmod(action, self.policy.num_filters)
        ra, dec = self.lookups.fields['ra'].to_numpy()[field_id], self.lookups.fields['dec'].to_numpy()[field_id]
        if hpGrid.is_azel:
            ra, dec = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=info.get('timestamp'))
        grid_bin = hpGrid.ang2idx(lon=ra, lat=dec)
        return (grid_bin if grid_bin is not None else 0), filter_idx, field_id

    def choose_bin_filter_field(self, obs, info, hpGrid, epsilon=None):
        """
        Choose field in bin based on interpolated Q-values; field_filter chooses the field directly.
        """
        # Unpack obs
        glob_tensor = torch.as_tensor(obs['global_state'], device=self.device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        cand_tensor = torch.as_tensor(obs['candidate_state'], device=self.device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action_tensor_mask = torch.as_tensor(info.get('action_mask', None), device=self.device, dtype=torch.bool) if info.get('action_mask', None) is not None else None

        if is_field_level(self.cfg.data.action_space):
            return self._choose_field_filter(glob_tensor, cand_tensor, action_tensor_mask, info, hpGrid)

        # Choose action in action space
        bin_idx, filter_idx = self._choose_bin_and_filter(glob_tensor, cand_tensor, action_tensor_mask, info, epsilon)

        # Get valid fields in bin
        valid_field_ids = self._determine_valid_fields(bin_idx, filter_idx, info)

        if self.field_choice_method == 'interp':
            with torch.no_grad():
                # glob_tensor = torch.as_tensor(glob_tensor, device=self.device, dtype=torch.float32).unsqueeze(0)
                # cand_tensor = torch.as_tensor(cand_tensor, device=self.device, dtype=torch.float32).unsqueeze(0)

                raw_scores = self.policy.core_net(glob_tensor, cand_tensor)

                n_bins = cand_tensor.shape[1]
                n_filters = raw_scores.shape[-1] // n_bins

                # Reshape to (n_bins, n_filters) and slice the specific filter
                q_map = raw_scores.view(n_bins, n_filters)[:, filter_idx].cpu().numpy()

            lon_data = hpGrid.lon
            lat_data = hpGrid.lat

            # CHECK
            target_coords = self.lookups.fields.loc[valid_field_ids, ['ra', 'dec']].to_numpy()

            if hpGrid.is_azel:
                # Project RA/Dec to local Az/El frame using the current timestamp
                timestamp = info.get('timestamp')
                target_lons, target_lats = ephemerides.equatorial_to_topographic(
                    ra=target_coords[:, 0],
                    dec=target_coords[:, 1],
                    time=timestamp
                )
            else:
                target_lons = target_coords[:, 0]
                target_lats = target_coords[:, 1]

            q_interpolated = interpolate_on_sphere(
                az=target_lons,
                el=target_lats,  # Target coordinates
                az_data=lon_data,
                el_data=lat_data,        # Bin centers (grid)
                values=q_map                      # Filter-specific Q-values
            )

            best_idx = np.argmax(q_interpolated)

            field_id = valid_field_ids[best_idx]

        elif self.field_choice_method == 'random':
            field_id = random.choice(valid_field_ids)

        return bin_idx, filter_idx, field_id
