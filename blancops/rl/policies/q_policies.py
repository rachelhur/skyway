"""Q-value adapters. Translate a network's output into a flat
`(batch, num_actions)` Q-tensor that DDQN/CQL can index into."""
from __future__ import annotations

import torch
from torch import nn

from blancops.rl.policies.base import QPolicyBase
from blancops.rl.policies.loss_function import SlewDistanceFocalLoss


class QFlatPolicy(QPolicyBase):
    """Trivial pass-through for networks that already emit flat Q-values."""

    def __init__(self, core_net: nn.Module, num_filters: int):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters

    def get_q_values(self, x_glob, x_cand) -> torch.Tensor:
        return self.core_net(x_glob=x_glob, x_cand=x_cand)
    

class QAutoregressivePolicy(QPolicyBase):
    """Builds the full flat Q-table from an autoregressive network.

    Q(s, a₁, a₂) = Q₁(s, a₁) + Q₂(s, a₂ | a₁) is enumerated over all (a₁, a₂)
    pairs so the algorithm can do standard argmax/gather operations.
    """

    def __init__(self, core_ar_net: nn.Module, num_filters: int):
        super().__init__()
        self.core_net = core_ar_net
        self.num_filters = num_filters
        self._filt_idx = core_ar_net._filt_idx
        self._bin_idx = core_ar_net._bin_idx

    def get_q_values(self, x_glob, x_cand) -> torch.Tensor:
        x_latent = self.core_net.state_encoder(x_glob, x_cand)
        batch_size = x_latent.size(0)

        # First head: Q over the first action dim.
        q_1 = self.core_net.action_heads[0](x_latent)
        dim1 = q_1.size(1)

        # Second head: enumerate over all first-head choices.
        all_choices_1 = torch.arange(dim1, device=x_latent.device)
        emb_1 = self.core_net.action_embeddings[0](all_choices_1)

        x_latent_exp = x_latent.unsqueeze(1).expand(-1, dim1, -1)
        emb_1_exp = emb_1.unsqueeze(0).expand(batch_size, -1, -1)
        x_current_2 = torch.cat([x_latent_exp, emb_1_exp], dim=-1)

        q_2 = self.core_net.action_heads[1](x_current_2)  # (batch, dim1, dim2)

        # Q(s, a1, a2) = Q(s, a1) + Q(s, a2 | a1)
        q_joint = q_1.unsqueeze(2) + q_2  # (batch, dim1, dim2)

        # Flatten to match dataloader's flat_action = bin * num_filters + filter.
        if self._filt_idx == 0:
            q_joint = q_joint.transpose(1, 2)
        return q_joint.flatten(start_dim=1)