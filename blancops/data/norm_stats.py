"""Normalization parameters fitted on the training set and applied everywhere else."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormStats:
    """Feature and reward normalization parameters fitted on the training set.

    Parameters
    ----------
    z_score : dict
        {'global_features': dict | None, 'bin_features': dict | None} z-score stats.
    rel_norm : dict
        {'global_features': dict | None, 'bin_features': dict | None} relative-norm stats.
    reward : dict or None
        {'min': float, 'max': float} of the unnormalized reward; None without a reward.
    """

    z_score: dict
    rel_norm: dict
    reward: dict | None = None

    def normalizer_kwargs(self, group: str) -> dict:
        """Stats for one feature group, as keyword arguments for StateNormalizer.transform.

        Parameters
        ----------
        group : str
            'global_features' or 'bin_features'.

        Returns
        -------
        dict
            {'z_stats_dict': dict, 'rel_stats_dict': dict}.
        """
        return {
            'z_stats_dict': self.z_score.get(group) or {},
            'rel_stats_dict': self.rel_norm.get(group) or {},
        }

    def to_dict(self) -> dict:
        """Plain-dict form stored in checkpoints.

        Returns
        -------
        dict
            {'z_score': dict, 'rel_norm': dict, 'reward': dict | None}.
        """
        return {'z_score': self.z_score, 'rel_norm': self.rel_norm, 'reward': self.reward}

    @classmethod
    def from_dict(cls, d: dict) -> NormStats:
        """Rebuild from a checkpoint dict; checkpoints without reward stats load with reward=None.

        Parameters
        ----------
        d : dict
            Output of to_dict().

        Returns
        -------
        NormStats
            The normalization parameters.
        """
        return cls(z_score=d['z_score'], rel_norm=d['rel_norm'], reward=d.get('reward'))
