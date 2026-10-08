"""Pure-function helpers for angular/airmass/ephemerides calculations.

These functions are independent of the Evaluator/DataContainer plumbing so they
can be reused (and tested) on their own.
"""
from __future__ import annotations

import numpy as np

from blancops.math.geometry import angular_separation
from blancops.ephemerides.ephemerides import get_source_ra_dec
from blancops.data.features.glob_features import (
    calc_moon_phase as _calc_moon_phase,
    calc_sun_and_moon_positions as _calc_sun_and_moon_pos,
)

import logging
logger = logging.getLogger(__name__)


def calc_airmass(el):
    """Plane-parallel airmass approximation. `el` in radians."""
    return 1 / np.cos(np.pi / 2 - el)


def calc_slew_distance(prev_radecs, radecs):
    """Per-row angular separation between two arrays of (ra, dec) pairs."""
    n = len(prev_radecs)
    out = np.zeros(n)
    for i in range(n):
        out[i] = angular_separation(prev_radecs[i], radecs[i])
    return out


def calc_moon_dist(radecs, timestamps):
    """Angular distance from each (ra, dec) to the moon at the matching timestamp."""
    out = np.zeros(len(timestamps))
    for i, t in enumerate(timestamps):
        moon_radec = get_source_ra_dec('moon', time=t)
        out[i] = angular_separation(moon_radec, radecs[i])
    return out


def calc_moon_phase(timestamps):
    out = np.empty(len(timestamps))
    for i, t in enumerate(timestamps):
        out[i] = _calc_moon_phase(t)
    return out


def calc_sun_and_moon_pos(timestamps):
    """Returns (sun_az, sun_el, moon_az, moon_el) arrays."""
    sun_azel = np.empty((len(timestamps), 2))
    moon_azel = np.empty((len(timestamps), 2))
    for i, t in enumerate(timestamps):
        _, sun_azel[i], _, moon_azel[i] = _calc_sun_and_moon_pos(t)
    return sun_azel[:, 0], sun_azel[:, 1], moon_azel[:, 0], moon_azel[:, 1]


def dump_filter_q_breakdown(policy, obs, info, idx2filter=None):
    """Report per-filter best-Q over visible vs available bins for one step.

    Diagnoses the joint-vs-filter-first coupling: for a single (obs, info),
    prints for each filter the best Q over *visible* bins (sky as if fully
    populated) against the best Q over *available* bins (sparse island layout),
    then the joint-argmax choice against the filter-first choice. The bug is
    confirmed when a filter wins on best-over-visible but its far/under-valued
    island bin loses the joint argmax to another filter's nearby bin.

    Args
    ----
    policy : trained flat-score policy exposing ``core_net`` and ``num_filters``.
    obs : dict with ``global_state`` and ``candidate_state`` arrays.
    info : dict with ``action_mask`` and ``visible_bin_mask``.
    idx2filter : optional mapping from filter index to a printable label.

    Returns
    -------
    dict keyed by filter index with ``max_q_visible``, ``max_q_available``,
    and ``n_avail_bins``.
    """
    import torch

    device = next(policy.parameters()).device
    x_glob = torch.as_tensor(obs['global_state'], device=device, dtype=torch.float32).unsqueeze(0)
    x_cand = torch.as_tensor(obs['candidate_state'], device=device, dtype=torch.float32).unsqueeze(0)

    num_filters = policy.num_filters
    with torch.no_grad():
        raw_scores = policy.core_net(x_glob, x_cand)
    n_bins = x_cand.shape[1]
    q_map = raw_scores.view(n_bins, num_filters).cpu()

    avail = torch.as_tensor(info['action_mask'], dtype=torch.bool).view(n_bins, num_filters)
    visible = torch.as_tensor(info['visible_bin_mask'], dtype=torch.bool)
    neg_inf = torch.finfo(q_map.dtype).min

    q_vis = q_map.masked_fill(~visible.unsqueeze(1), neg_inf)
    q_av = q_map.masked_fill(~avail, neg_inf)

    breakdown = {}
    logger.info("filter | max_q_visible | max_q_available | n_avail_bins")
    for f in range(num_filters):
        label = idx2filter[f] if idx2filter is not None else f
        max_vis = q_vis[:, f].max().item()
        n_av = int(avail[:, f].sum().item())
        max_av = q_av[:, f].max().item() if n_av else float('nan')
        breakdown[f] = {'max_q_visible': max_vis, 'max_q_available': max_av, 'n_avail_bins': n_av}
        logger.info(f"{label} | {max_vis:.4f} | {max_av:.4f} | {n_av}")

    joint_bin, joint_filter = divmod(int(q_av.flatten().argmax().item()), num_filters)

    filter_scores = q_vis.max(dim=0).values.masked_fill(~avail.any(dim=0), neg_inf)
    ff_filter = int(filter_scores.argmax().item())
    ff_bin = int(q_av[:, ff_filter].argmax().item())

    jf = idx2filter[joint_filter] if idx2filter is not None else joint_filter
    ff = idx2filter[ff_filter] if idx2filter is not None else ff_filter
    logger.info(f"joint choice:        bin={joint_bin} filter={jf}")
    logger.info(f"filter-first choice: bin={ff_bin} filter={ff}")
    return breakdown
