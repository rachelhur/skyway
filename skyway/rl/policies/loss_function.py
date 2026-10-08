"""Custom loss functions for action prediction.

These are `nn.Module`s so weights/buffers can move with `.to(device)`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from skyway.configs.constants import FILTER_ALPHA_WEIGHTS
from skyway.math import units


class FocalLoss(nn.Module):
    """Focal loss for class imbalance. https://arxiv.org/abs/1708.02002

    `alpha`: per-class weighting. Defaults to `FILTER_ALPHA_WEIGHTS`. Pass
    `alpha=None` explicitly to disable class weighting.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean", alpha=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            alpha_tensor = None
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.float()
        else:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)

        # Register as buffer so `.to(device)` moves it with the module.
        if alpha_tensor is not None:
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce

        if self.alpha is not None:
            focal = self.alpha[targets] * focal

        return _reduce(focal, self.reduction)


class FilterFocalLoss(nn.Module):
    """Focal loss specialized for filter-only prediction (always uses
    `FILTER_ALPHA_WEIGHTS`). Used by the hybrid-marginal strategy."""
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "alpha", torch.tensor(FILTER_ALPHA_WEIGHTS, dtype=torch.float32)
        )

    def forward(self, filter_logits: torch.Tensor, filter_targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(filter_logits, filter_targets, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        focal = self.alpha[filter_targets] * focal
        return _reduce(focal, self.reduction)


class SlewDistanceFocalLoss(nn.Module):
    """Focal loss scaled by the physical slew distance the expert took.

    Larger slews → higher loss weight. The `+ 1.0` ensures zero-degree
    slews still contribute.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.scale = 1.0 / units.deg

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        expert_slew_distances: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce

        weights = expert_slew_distances * self.scale + 1.0
        focal = weights * focal
        return _reduce(focal, self.reduction)


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss