"""Per-field features for the field_filter action space, shared by offline precompute and the environments.

One helper, `compute_field_features`, computes every field feature for one timestep; the offline driver
`FieldFeatureEngineer` and the environments both call it. Positional features reuse the candidate ephemeris
math evaluated at field centers through `FieldGrid`.

Progress convention: counts start each night from the lookups (valid exposures only, like the targets) and
advance for every exposure of the night, as on the bin path. The features of a state include that state's own
exposure, matching the environment after a step. Out-of-plan field-filters are NaN internally and become the sentinel in
normalization.
"""
import numpy as np
from tqdm import tqdm

from skyway.configs.constants import ZENITH_FIELD_ID, _FIELD_FEATURES
from skyway.data.features.candidate_features import compute_candidate_ephemeris_features, get_relative_feature
from skyway.data.features.glob_features import compute_global_mean_tiling_features, get_night_boundaries
from skyway.data.features.normalizations import StateNormalizer
from skyway.ephemerides import ephemerides
from skyway.ephemerides.ephemerides import HealpixGrid
from skyway.survey.profiles import DES, SurveyProfile

_PER_FILTER_FEATURES = ('completion', 'rel_completion', 't_since_last_visit')
_AIRMASS_CAP = 3.0
_LOG_FEATURES = ('t_since_last_visit',)
_NEVER_VISITED_AGE = 5 * 365.25 * 86400.0

# Physical range (lo, hi) per base feature; normalization maps it to about [-1, 1].
# t_since_last_visit is in observing-time seconds and ranged after a log transform.
FIELD_FEATURE_RANGES = {
    'el': (-np.pi / 2, np.pi / 2),
    'airmass': (1.0, _AIRMASS_CAP),
    'ha': (-np.pi, np.pi),
    'moon_distance': (0.0, np.pi),
    'sun_distance': (0.0, np.pi),
    'pointing_distance': (0.0, np.pi),
    'delta_az': (-np.pi, np.pi),
    'delta_el': (-np.pi, np.pi),
    't_until_set': (0.0, 2.5),
    'rel_ha': (-np.pi, np.pi),
    'rel_moon_distance': (-np.pi, np.pi),
    'completion': (0.0, 1.0),
    'rel_completion': (-1.0, 1.0),
    't_since_last_visit': (np.log(60.0), np.log(_NEVER_VISITED_AGE)),
}


class FieldGrid(HealpixGrid):
    """HealpixGrid-compatible container over survey field centers (RA/Dec).

    Parameters
    ----------
    ra, dec : np.ndarray
        Field centers in radians, indexed by field_id.
    """

    def __init__(self, ra: np.ndarray, dec: np.ndarray):
        self.nside = None
        self.is_azel = False
        self.lon = np.asarray(ra, dtype=float)
        self.lat = np.asarray(dec, dtype=float)
        self.heal_idx = np.arange(len(self.lon))
        self.idx_lookup = {i: i for i in range(len(self.lon))}
        self.npix = len(self.lon)


def expand_field_feature_names(base_features: list[str], survey: SurveyProfile = DES) -> list[str]:
    """Expand per-filter base names into `name_{filter}` columns, in the survey's filter order.

    Parameters
    ----------
    base_features : list of str
        Base names from `_FIELD_FEATURES`.
    survey : SurveyProfile
        Survey whose filters suffix the per-filter names.

    Returns
    -------
    list of str
        Stacking order of the field feature columns.
    """
    names = []
    for f in base_features:
        if f in _PER_FILTER_FEATURES:
            names.extend(f"{f}_{filt}" for filt in survey.filters)
        else:
            names.append(f)
    return names


def field_norm_stats(feature_names: list[str]) -> dict:
    """Constant normalization stats {name: {'mean', 'std'}} mapping each physical range to about [-1, 1].

    Parameters
    ----------
    feature_names : list of str
        Expanded field feature names.

    Returns
    -------
    dict
        Stats in the form StateNormalizer.transform expects for z-scored features.
    """
    stats = {}
    for name in feature_names:
        base = next(b for b in FIELD_FEATURE_RANGES if name == b or name.startswith(f"{b}_"))
        lo, hi = FIELD_FEATURE_RANGES[base]
        stats[name] = {'mean': (lo + hi) / 2, 'std': (hi - lo) / 2}
    return stats


def build_field_normalizer(feature_names: list[str]) -> StateNormalizer:
    """StateNormalizer for field features: log for staleness, then constant-stat scaling, NaN to -1.

    Parameters
    ----------
    feature_names : list of str
        Expanded field feature names.

    Returns
    -------
    StateNormalizer
        Normalizer to use with `transform(state, field_norm_stats(names), {})`.
    """
    log_feats = [n for n in feature_names if any(n.startswith(b) for b in _LOG_FEATURES)]
    return StateNormalizer(
        state_feature_names=feature_names, sin_norm_feature_names=[], log_norm_feature_names=log_feats,
        fractional_norm_feature_names=[], z_score_feature_names=list(feature_names),
        local_mean_z_score_feature_names=[], do_local_mean_z_score=False,
    )


def compute_field_features(timestamp: float, pointing_radec, field_grid: FieldGrid, night_duration_sec: float,
                           counts: np.ndarray, targets: np.ndarray, last_visit_ot: np.ndarray,
                           ot_now: float, survey: SurveyProfile = DES) -> dict:
    """All field features for one timestep.

    Completion = count / target (capped at 1); airmass is capped at 3; time since last visit = OT now - last visit, only for
    in-plan incomplete field-filters, and the top of its range when never visited; relative features subtract the mean over fields above the horizon.

    Parameters
    ----------
    timestamp : float
        Unix timestamp (UTC).
    pointing_radec : tuple
        Current telescope (RA, Dec) in radians.
    field_grid : FieldGrid
        Field centers.
    night_duration_sec : float
        Night duration used to scale t_until_set.
    counts, targets : np.ndarray
        (n_fields, n_filters) visit counts and survey targets.
    last_visit_ot : np.ndarray
        (n_fields, n_filters) seconds (in observing time OT) of the last visit, NaN if never.
    ot_now : float
        Current OT seconds.
    survey : SurveyProfile
        Survey whose filter order indexes axis 1 of the counts.

    Returns
    -------
    dict
        Feature name -> (n_fields,) array; per-filter features as `name_{filter}`.
    """
    features = compute_candidate_ephemeris_features(
        timestamp=timestamp, pointing_radec=pointing_radec, grid=field_grid,
        night_duration_in_sec=night_duration_sec,
    )
    features['airmass'] = np.minimum(np.nan_to_num(features['airmass'], nan=_AIRMASS_CAP, posinf=_AIRMASS_CAP),
                                     _AIRMASS_CAP)
    el_mask = features['el'] > 0
    features['rel_ha'] = get_relative_feature(features['ha'], el_mask)
    features['rel_moon_distance'] = get_relative_feature(features['moon_distance'], el_mask)

    in_plan = targets > 0
    incomplete = counts < targets
    completion = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(counts, targets, out=completion, where=in_plan)
    completion = np.minimum(completion, 1.0)
    never_visited = np.isnan(last_visit_ot)
    age = np.where(never_visited, _NEVER_VISITED_AGE, np.maximum(ot_now - np.nan_to_num(last_visit_ot), 0.0))
    age = np.where(in_plan & incomplete, age, np.nan)
    for f, filt in survey.idx2filter.items():
        features[f"completion_{filt}"] = completion[:, f]
        features[f"rel_completion_{filt}"] = get_relative_feature(completion[:, f], el_mask)
        features[f"t_since_last_visit_{filt}"] = age[:, f]
    return features


def stack_field_features(features: dict, feature_names: list[str]) -> np.ndarray:
    """Stack a feature dict into (n_fields, n_features) in `feature_names` order.

    Parameters
    ----------
    features : dict
        Output of `compute_field_features`.
    feature_names : list of str
        Expanded field feature names.

    Returns
    -------
    np.ndarray
        float32 array (n_fields, n_features).
    """
    return np.stack([np.asarray(features[n], dtype=np.float32) for n in feature_names], axis=-1)


def label_mask_report(global_df, current_state_idxs, next_state_idxs, field_features: np.ndarray,
                      feature_names: list[str], field_dec: np.ndarray, telescope, airmass_limit: float,
                      survey: SurveyProfile = DES) -> dict:
    """Count expert labels that the field-level mask of their own state would forbid, by reason.

    Reasons: not visible (airmass at or above the limit, or outside the mount envelope), already complete
    (valid counts at target), and out of plan (no target). A label can have several reasons.

    Parameters
    ----------
    global_df : pd.DataFrame
        Cached global frame.
    current_state_idxs, next_state_idxs : np.ndarray
        Transition row indices.
    field_features : np.ndarray
        (n_rows, n_fields, n_features) raw field features.
    feature_names : list of str
        Expanded field feature names; must include el, ha, and completion.
    field_dec : np.ndarray
        Field declinations in radians, indexed by field_id.
    telescope : TelescopeProfile
        Supplies the visibility rule and mount envelope.
    airmass_limit : float
        Effective airmass limit.
    survey : SurveyProfile
        Survey (carries filters and their ordering).

    Returns
    -------
    dict
        Reason -> number of labels, plus 'any' (at least one reason) and 'total'.
    """
    nxt = global_df.iloc[next_state_idxs]
    fid = nxt['field_id'].to_numpy(dtype=np.int64)
    filt = nxt['filter'].map(survey.filter2idx).to_numpy(dtype=np.int64)
    rows = np.asarray(current_state_idxs)
    col = {n: i for i, n in enumerate(feature_names)}
    el = field_features[rows, fid, col['el']]
    ha = field_features[rows, fid, col['ha']]
    comp_cols = np.array([col[f"completion_{f}"] for f in survey.filters])[filt]
    completion = field_features[rows, fid, comp_cols]
    not_visible = ~telescope.visible(el, ha, field_dec[fid], airmass_limit)
    out_of_plan = np.isnan(completion)
    complete = ~out_of_plan & (completion >= 1.0)
    return {
        'not_visible': int(not_visible.sum()), 'already_complete': int(complete.sum()),
        'out_of_plan': int(out_of_plan.sum()), 'any': int((not_visible | complete | out_of_plan).sum()),
        'total': len(rows),
    }


class FieldFeatureEngineer:
    """Engineer for field-level features.

    Seeds from the night-start lookups.

    Parameters
    ----------
    lookups : LookupTables
        Survey lookups (fields, targets, night-start counts, last-visit OT, OT clock).
    base_features : list of str
        Requested base names from `_FIELD_FEATURES`.
    """

    def __init__(self, lookups, base_features: list[str]):
        unknown = [f for f in base_features if f not in _FIELD_FEATURES]
        if unknown:
            raise ValueError(f"Unknown field features: {unknown}")
        self.lookups = lookups
        self.survey = lookups.survey
        self.base_features = list(base_features)
        self.feature_names = expand_field_feature_names(self.base_features, self.survey)
        self.grid = FieldGrid(lookups.fields['ra'].to_numpy(), lookups.fields['dec'].to_numpy())

    def _pointing(self, field_id: int, timestamp: float) -> np.ndarray:
        """Pointing as the environment defines it: the field center, or the zenith at ``timestamp``."""
        if field_id == ZENITH_FIELD_ID:
            return np.array(ephemerides.blanco_observer(time=timestamp).radec_of('0', '90'))
        return np.array([self.grid.lon[field_id], self.grid.lat[field_id]])

    def _require(self, name: str):
        if not hasattr(self.lookups, name):
            raise AttributeError(f"LookupTables is missing '{name}'; rebuild lookups via build_train_lookups.")
        return getattr(self.lookups, name)

    def transform(self, pt_df, out: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Field features for every row, plus global mean tiling.

        Parameters
        ----------
        pt_df : pd.DataFrame
            Global frame with night, timestamp, ra, dec, field_id, filter, teff; timestamps strictly
            increasing.
        out : np.ndarray or None
            Preallocated (n_rows, n_fields, n_features) float32 array to fill, e.g. a disk memmap.

        Returns
        -------
        tuple
            (features (n_rows, n_fields, n_features) float32,
             {'global_mean_tiling': (n_rows,), 'global_mean_tiling_{filt}': (n_rows,)} after each row).
        """
        timestamps = pt_df['timestamp'].to_numpy()
        assert np.all(np.diff(timestamps) > 0), "Timestamps must be strictly increasing."
        targets = self.lookups.target_fidfilt_counts
        visit_hist = self._require('night2fidfilt_visit_hist')
        last_visit_hist = self._require('night2fidfilt_last_visit_ot')
        ot_clock = self._require('night2ot_clock_seconds')
        n_rows, n_fields = len(pt_df), len(self.grid.lon)
        if out is None:
            out = np.empty((n_rows, n_fields, len(self.feature_names)), dtype=np.float32)
        tiling_keys = ['global_mean_tiling'] + [f'global_mean_tiling_{f}' for f in self.survey.filters]
        tiling = {k: np.empty(n_rows, dtype=np.float32) for k in tiling_keys}
        filt_idx = pt_df['filter'].map(self.survey.filter2idx).fillna(-1).to_numpy(dtype=np.int64)

        i = 0
        pbar = tqdm(total=n_rows, desc='Computing field features')
        for night, group in pt_df.groupby('night', sort=False):
            sunset_ts, sunrise_ts = get_night_boundaries(group['timestamp'], sun_el_limit=DES.sun_el_limit)
            counts = visit_hist[night].copy().astype(np.int64)
            last_visit = last_visit_hist[night].copy().astype(np.float64)
            ot_sunset = ot_clock[night]
            ts = group['timestamp'].to_numpy()
            fids = group['field_id'].to_numpy(dtype=np.int64)
            for j in range(len(group)):
                obs_t = ot_sunset + (ts[j] - sunset_ts)
                f = filt_idx[i]
                if fids[j] != ZENITH_FIELD_ID and f >= 0:
                    counts[fids[j], f] += 1
                    last_visit[fids[j], f] = obs_t
                feats = compute_field_features(
                    timestamp=ts[j], pointing_radec=self._pointing(fids[j], ts[j]), field_grid=self.grid,
                    night_duration_sec=sunrise_ts - sunset_ts, counts=counts, targets=targets,
                    last_visit_ot=last_visit, ot_now=obs_t, survey=self.survey,
                )
                out[i] = stack_field_features(feats, self.feature_names)
                mt = compute_global_mean_tiling_features(running_counts=counts, target_counts=targets,
                                                         survey=self.survey)
                for k in tiling_keys:
                    tiling[k][i] = mt[k]
                i += 1
                pbar.update(1)
        pbar.close()
        return out, tiling
