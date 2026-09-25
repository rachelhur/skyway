from typing import Callable

import numpy as np

from blancops.configs.enums import RewardTerm
from blancops.configs.experiment_schema import RewardConfig


def expert_action_reward(n_transitions: int) -> np.ndarray:
    """Constant reward of 1 for taking expert actions, 0 otherwise.

    Parameters
    ----------
    n_transitions : int
        Number of transitions in the episode.

    Returns
    -------
    np.ndarray
        Array of ones, shape (n_transitions,).
    """
    return np.ones(n_transitions, dtype=np.float32)


def teff_reward(teff: np.ndarray) -> np.ndarray:
    """Reward based on the effective exposure time (t_eff).

    Parameters
    ----------
    teff : np.ndarray
        Effective exposure time, shape (n_transitions,).

    Returns
    -------
    np.ndarray
        Reward, shape (n_transitions,). NaN values are replaced with 0.0.
    """
    return np.nan_to_num(teff, nan=0.0)


def slew_reward(excess_times: np.ndarray, decay_time: float = 10.0) -> np.ndarray:
    """Reward of 1 when the slew adds no dead time, decaying exponentially with the time it adds.

    R = exp(-t_excess / tau), where t_excess = t_dead - t_o is the dead time between
    exposures beyond the per-visit overhead, and tau the decay time.

    Parameters
    ----------
    excess_times : np.ndarray
        Dead time added by the slew beyond the per-visit overhead in seconds,
        shape (n_transitions,).
    decay_time : float
        Excess dead time tau in seconds that reduces the reward by 1/e.

    Returns
    -------
    np.ndarray
        Reward in (0, 1], shape (n_transitions,).
    """
    return np.exp(-excess_times / decay_time)


def uniformity_reward(
    completion: np.ndarray,
    filter_mean: np.ndarray,
    total_target: np.ndarray,
    pass_size: np.ndarray,
    ref_gap: float = 0.2,
) -> np.ndarray:
    """Change in target-weighted survey uniformity from one exposure of a field-filter.

    U = -(1/c) * sum_b sum_f T_fb * (x_fb - m_b)^2 with c = 2 * ref_gap and m_b = sum_f n_fb / W_b,
    so one exposure in filter b gives R = [2 * (m_b - x) - (delta - 1/W_b)] / c.

    Parameters
    ----------
    completion : np.ndarray
        Completion x = count / target of the exposed field-filter before the exposure, shape (n_transitions,).
    filter_mean : np.ndarray
        Completion m_b = sum_f n_fb / W_b of the exposure's filter over in-plan fields, before the exposure.
    total_target : np.ndarray
        Total target W_b = sum_f T_fb over in-plan fields in the exposure's filter.
    pass_size : np.ndarray
        One exposure in completion units, delta = 1 / target; 0 gives reward 0 (out-of-plan or failed).
    ref_gap : float
        Reference completion gap; an exposure ref_gap behind its filter scores about 1.

    Returns
    -------
    np.ndarray
        Reward, shape (n_transitions,).
    """
    in_plan = pass_size > 0
    safe_total = np.where(in_plan, total_target, 1.0)
    delta_u = (2.0 * (filter_mean - completion) - (pass_size - 1.0 / safe_total)) / (2.0 * ref_gap)
    return np.where(in_plan, delta_u, 0.0)


def uniformity_inputs(counts: np.ndarray, targets: np.ndarray, field_id: int, filter_idx: int) -> dict:
    """Inputs of uniformity_reward for one exposure, from survey counts before the exposure.

    Parameters
    ----------
    counts : np.ndarray
        Visit counts per field-filter, shape (n_fields, n_filters).
    targets : np.ndarray
        Target counts per field-filter, shape (n_fields, n_filters); in-plan where > 0.
    field_id : int
        Exposed field.
    filter_idx : int
        Exposed filter.

    Returns
    -------
    dict
        completion, filter_mean, total_target, pass_size as arrays of shape (1,); pass_size = 0 (reward 0)
        for an out-of-plan field-filter.
    """
    if counts.ndim != 2:
        raise ValueError("The uniformity reward needs field-filter (2D) counts, i.e. a filter action space.")
    target = targets[field_id, filter_idx]
    if target <= 0:
        return dict(completion=np.zeros(1), filter_mean=np.zeros(1), total_target=np.ones(1), pass_size=np.zeros(1))
    in_plan = targets[:, filter_idx] > 0
    total_target = targets[in_plan, filter_idx].sum()
    return dict(
        completion=np.array([counts[field_id, filter_idx] / target]),
        filter_mean=np.array([counts[in_plan, filter_idx].sum() / total_target]),
        total_target=np.array([total_target], dtype=float),
        pass_size=np.array([1.0 / target]),
    )


REWARD_TERMS: dict[RewardTerm, Callable[..., np.ndarray]] = {
    RewardTerm.EXPERT: expert_action_reward,
    RewardTerm.TEFF: teff_reward,
    RewardTerm.SLEW: slew_reward,
    RewardTerm.UNIFORMITY: uniformity_reward,
}


def combine_rewards(cfg: RewardConfig, term_inputs: dict[RewardTerm, Callable[[], dict]]) -> np.ndarray:
    """Weighted sum of the configured terms, R = sum_k w_k * R_k (unnormalized).

    Parameters
    ----------
    cfg : RewardConfig
        Reward terms and weights.
    term_inputs : dict[RewardTerm, Callable[[], dict]]
        Per term, a callable returning that term's keyword arguments.

    Returns
    -------
    np.ndarray
        Unnormalized reward per transition, float32, shape (n_transitions,).
    """
    R_tot = sum(w * REWARD_TERMS[term](**term_inputs[term]()) for term, w in cfg.terms.items())
    return np.asarray(R_tot, dtype=np.float32)


# -------------------------------------------------------------- #
# -------------------- REWARD NORMS ---------------------------- #
# -------------------------------------------------------------- #

def _minmax_fit(R_tot: np.ndarray) -> dict:
    return {'min': float(R_tot.min()), 'max': float(R_tot.max())}


def _minmax_apply(R_tot: np.ndarray, stats: dict) -> np.ndarray:
    span = stats['max'] - stats['min']
    return (R_tot - stats['min']) / span if span > 0 else R_tot


REWARD_NORMS: dict[str, tuple[Callable, Callable]] = {
    'minmax': (_minmax_fit, _minmax_apply),
}


def reward_norm_stats(cfg: RewardConfig, R_tot: np.ndarray) -> dict | None:
    """Fit reward normalization stats.

    Parameters
    ----------
    cfg : RewardConfig
        Reward configuration; cfg.norm selects the normalization.
    R_tot : np.ndarray
        Unnormalized training rewards, shape (n_transitions,).

    Returns
    -------
    dict or None
        Fitted stats (for minmax: {'min', 'max'}); None when cfg.norm is None.
    """
    return None if cfg.norm is None else REWARD_NORMS[cfg.norm][0](R_tot)


def normalize_rewards(cfg: RewardConfig, R_tot: np.ndarray, stats: dict | None) -> np.ndarray:
    """Apply reward normalization.

    For minmax: R' = (R - min) / (max - min), identity when max == min.

    Parameters
    ----------
    cfg : RewardConfig
        Reward configuration; cfg.norm selects the normalization.
    R_tot : np.ndarray
        Unnormalized rewards.
    stats : dict or None
        Stats from reward_norm_stats on the training set; required unless cfg.norm is None.

    Returns
    -------
    np.ndarray
        Normalized rewards, float32, same shape as R_tot.
    """
    if cfg.norm is None:
        return R_tot
    if stats is None:
        raise ValueError(f"reward.norm == '{cfg.norm}' requires stats fitted on the training set.")
    return np.asarray(REWARD_NORMS[cfg.norm][1](R_tot, stats), dtype=np.float32)
