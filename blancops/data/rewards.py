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


REWARD_TERMS: dict[RewardTerm, Callable[..., np.ndarray]] = {
    RewardTerm.EXPERT: expert_action_reward,
    RewardTerm.TEFF: teff_reward,
    RewardTerm.SLEW: slew_reward,
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
