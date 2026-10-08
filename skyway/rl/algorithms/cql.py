import torch
import torch.nn.functional as F

from skyway.configs.enums import Algorithm
from skyway.rl.algorithms.ddqn import DDQN

import logging
logger = logging.getLogger(__name__)


class CQL(DDQN):
    """Conservative Q-Learning = DDQN + a logsumexp penalty over OOD actions."""
    name = Algorithm.CQL

    def __init__(
        self,
        cql_alpha: float = 1.0,
        cql_margin: float = 0.0,
        dist_matrix=None,
        dist_scaling_factor: float = 1.0,
        **ddqn_kwargs,
    ):
        super().__init__(**ddqn_kwargs)
        self.cql_alpha = cql_alpha
        self.cql_margin = cql_margin
        self.dist_matrix = dist_matrix
        self.dist_scaling_factor = dist_scaling_factor

    def _compute_loss(self, batch_dict, candidate_grid=None, compute_metrics=False):
        q_vals_all, q_val, q_expected = self._forward_q(batch_dict)

        td_loss = self._td_loss(q_val, q_expected)
        cql_penalty = self._calculate_cql_penalty(
            q_vals_all,
            q_val,
            batch_dict["actions"],
            batch_dict["action_masks"],
            margin=self.cql_margin,
        )
        cql_loss = self.cql_alpha * cql_penalty
        loss = td_loss + cql_loss

        metrics = {}
        if compute_metrics:
            metrics = self._build_metrics(q_vals_all, q_val, q_expected, batch_dict, candidate_grid) # inherited from ddqn
            metrics["td_loss"] = td_loss.item()
            metrics["cql_penalty"] = cql_penalty.item()
            metrics["cql_loss"] = cql_loss.item()
        return loss, metrics

    def _calculate_cql_penalty(
        self, q_vals_all, q_val_expert, actions, action_masks, margin: float = 0.0) -> torch.Tensor:
        """Unscaled conservative penalty L_CQL = mean_s [logsumexp_{valid a} Q(s,a) - Q(s,a_data)].

        Parameters
        ----------
        q_vals_all : torch.Tensor
            Q-values for every action, shape (batch, n_actions).
        q_val_expert : torch.Tensor
            Q of the dataset action, shape (batch,).
        actions : torch.Tensor
            Dataset actions, shape (batch, 1).
        action_masks : torch.Tensor
            Valid-action mask, bool, shape (batch, n_actions).
        margin : float
            Added to every non-dataset action inside the logsumexp.

        Returns
        -------
        torch.Tensor
            Scalar penalty; the loss term is cql_alpha times this.
        """
        num_actions = q_vals_all.size(1)
        mask_val = torch.finfo(q_vals_all.dtype).min

        q_cql = q_vals_all.clone()
        q_cql[~action_masks] = mask_val

        if margin != 0.0:
            expert_oh = F.one_hot(actions.squeeze(1), num_classes=num_actions).bool()
            q_cql = q_cql + margin * (~expert_oh).to(q_cql.dtype)

        cql_logsumexp = torch.logsumexp(q_cql, dim=1)
        return (cql_logsumexp - q_val_expert).mean()