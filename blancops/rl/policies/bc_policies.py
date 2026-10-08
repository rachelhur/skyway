"""BC loss strategies. Each defines how expert actions map to a loss given
the network's output shape."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from blancops.rl.policies.base import (
    BCPolicyBase,
)
from blancops.rl.policies.loss_function import SlewDistanceFocalLoss


class BCPureJointPolicy(BCPolicyBase):
    """Flat cross-entropy over the full joint (candidate x filter) action space."""

    def __init__(self, core_net: nn.Module, loss_function: nn.Module, num_filters: int):
        super().__init__()
        self.core_net = core_net
        self.loss_function = loss_function
        self.num_filters = num_filters

    def compute_loss_and_metrics(self, batch, candidate_grid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch["state"], x_cand=batch["candidate_states"])
        action_masks = batch["action_masks"]
        expert_flat = batch["expert_actions"]
        exp_slew_dists = batch.get("slew_dists", None)

        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        # Dispatch on loss-function signature.
        if exp_slew_dists is not None and isinstance(self.loss_function, SlewDistanceFocalLoss):
            loss = self.loss_function(action_logits, expert_flat, exp_slew_dists)
        else:
            loss = self.loss_function(action_logits, expert_flat)

        metrics: dict = {}
        if compute_metrics:
            metrics = self.compute_standard_metrics(
                action_logits, expert_flat, action_masks, self.num_filters, candidate_grid
            )
            metrics.update(self._marginal_loss_diagnostics(action_logits, expert_flat))

        return loss, metrics

    def _marginal_loss_diagnostics(self, action_logits, expert_flat) -> dict:
        """Auxiliary: report what candidate- and filter-marginal losses would be
        if you trained them separately. Useful for understanding what the
        joint loss is implicitly weighting."""
        with torch.no_grad():
            batch_size = action_logits.size(0)
            n_cands = action_logits.size(1) // self.num_filters
            logits_2d = action_logits.view(batch_size, n_cands, self.num_filters)

            log_norm = torch.logsumexp(action_logits, dim=-1, keepdim=True)   # (batch, 1)

            cand_log_probs   = torch.logsumexp(logits_2d, dim=2) - log_norm   # (batch, n_cands)
            filter_log_probs = torch.logsumexp(logits_2d, dim=1) - log_norm   # (batch, n_filters)

            expert_cand = expert_flat // self.num_filters
            expert_filter = expert_flat % self.num_filters

            cand_loss = F.nll_loss(cand_log_probs, expert_cand)
            filter_loss = F.nll_loss(filter_log_probs, expert_filter)
        return {"candidate_loss": cand_loss.item(), "filter_loss": filter_loss.item()}

    def select_action(self, x_glob, x_cand, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_cand=x_cand)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)


class BCPseudoAutoregressivePolicy(BCPolicyBase):
    """Factored loss on simultaneous logits: marginalized candidate loss + filter
    loss conditioned on the expert candidate. Encourages the network to get the
    candidate right first, then the filter given that candidate."""

    def __init__(self, core_net: nn.Module, num_filters: int, filter_penalty: float = 5.0):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self.filter_penalty = filter_penalty

    def compute_loss_and_metrics(self, batch, candidate_grid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch["state"], x_cand=batch["candidate_states"])
        action_masks = batch["action_masks"]
        expert_flat = batch["expert_actions"]

        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        batch_size = action_logits.size(0)
        n_cands = action_logits.size(1) // self.num_filters
        logits_2d = action_logits.view(batch_size, n_cands, self.num_filters)

        expert_cands = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters

        cand_logits = torch.logsumexp(logits_2d, dim=2)
        cand_loss = F.cross_entropy(cand_logits, expert_cands)

        batch_idx = torch.arange(batch_size, device=action_logits.device)
        filter_logits_at_expert_cand = logits_2d[batch_idx, expert_cands, :]
        filter_loss = F.cross_entropy(filter_logits_at_expert_cand, expert_filters)

        loss = cand_loss + self.filter_penalty * filter_loss

        metrics: dict = {}
        if compute_metrics:
            metrics = self.compute_standard_metrics(
                action_logits, expert_flat, action_masks, self.num_filters, candidate_grid
            )
            metrics["candidate_loss"] = cand_loss.item()
            metrics["filter_loss"] = filter_loss.item()

        return loss, metrics

    def select_action(self, x_glob, x_cand, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_cand=x_cand)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)



class BCHybridMarginalPolicy(BCPolicyBase):
    """Weighted sum of candidate-marginal + filter-marginal + joint losses.

    α·candidate_loss + β·filter_loss + ζ·joint_loss. If using focal loss on the
    filter head, bump β to balance against the focal-loss scaling.
    """

    def __init__(
        self,
        core_net: nn.Module,
        num_filters: int,
        candidate_loss_function: nn.Module,
        filter_loss_function: nn.Module,
        joint_loss_function: nn.Module,
        alpha_candidate: float = 1.0,
        beta_filter: float = 5.0,
        zeta_joint: float = 0.1,
    ):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self.alpha = alpha_candidate
        self.beta = beta_filter
        self.zeta = zeta_joint
        self.candidate_loss_function = candidate_loss_function
        self.filter_loss_function = filter_loss_function
        self.joint_loss_function = joint_loss_function

    def compute_loss_and_metrics(self, batch, candidate_grid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch["state"], x_cand=batch["candidate_states"])
        action_masks = batch["action_masks"]
        expert_flat = batch["expert_actions"]

        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        batch_size = action_logits.size(0)
        n_cands = action_logits.size(1) // self.num_filters
        logits_2d = action_logits.view(batch_size, n_cands, self.num_filters)

        expert_cands = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters

        cand_logits_marginal = torch.logsumexp(logits_2d, dim=1) #XXX check dims before running
        cand_loss = self.candidate_loss_function(cand_logits_marginal, expert_cands)

        filter_logits_marginal = torch.logsumexp(logits_2d, dim=2)
        filter_loss = self.filter_loss_function(filter_logits_marginal, expert_filters)

        joint_loss = self.joint_loss_function(action_logits, expert_flat)

        total_loss = self.alpha * cand_loss + self.beta * filter_loss + self.zeta * joint_loss

        metrics: dict = {}
        if compute_metrics:
            metrics = self.compute_standard_metrics(
                action_logits, expert_flat, action_masks, self.num_filters, candidate_grid
            )
            metrics.update({
                "candidate_loss": cand_loss.item(),
                "filter_loss": filter_loss.item(),
                "joint_loss": joint_loss.item(),
            })

        return total_loss, metrics

    def select_action(self, x_glob, x_cand, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_cand=x_cand)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)


class BCAutoregressivePolicy(BCPolicyBase):
    """Pairs with `AutoregressiveNet`. The network samples actions
    sequentially with embedding conditioning; the loss is the negative
    joint log-probability of the expert action sequence."""

    def __init__(self, core_net: nn.Module, num_filters: int):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self._filt_idx = core_net._filt_idx
        self._bin_idx = core_net._bin_idx

    def compute_loss_and_metrics(self, batch, candidate_grid=None, compute_metrics=False):
        x_glob = batch["state"]
        x_cand = batch["candidate_states"]
        expert_flat = batch["expert_actions"]
        action_masks = batch["action_masks"]

        expert_bins = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters
        if self._filt_idx == 0:
            expert_multidim = torch.stack([expert_filters, expert_bins], dim=1)
        else:
            expert_multidim = torch.stack([expert_bins, expert_filters], dim=1)

        _, joint_logp, joint_entropy = self.core_net(
            x_glob=x_glob,
            x_cand=x_cand,
            action=expert_multidim,
            action_mask=action_masks,
        )

        loss = -joint_logp.mean()

        metrics: dict = {}
        if compute_metrics:
            with torch.no_grad():
                metrics["entropy"] = joint_entropy.mean().item()
                metrics["logp_expert_action"] = joint_logp.mean().item()

                pred_multidim, _, _ = self.core_net(
                    x_glob=x_glob, x_cand=x_cand, action_mask=action_masks, action=None
                )
                pred_filters = pred_multidim[:, self._filt_idx]
                pred_bins = pred_multidim[:, self._bin_idx]
                pred_flat = pred_bins * self.num_filters + pred_filters

                if candidate_grid is not None:
                    heavy = self.compute_heavy_metrics(pred_flat, expert_flat, candidate_grid, self.num_filters)
                    metrics.update(heavy)

        return loss, metrics

    def select_action(self, x_glob, x_cand, action_mask=None):
        sampled, _, _ = self.core_net(
            x_glob=x_glob, x_cand=x_cand, action_mask=action_mask, action=None
        )
        pred_filters = sampled[:, self._filt_idx]
        pred_bins = sampled[:, self._bin_idx]
        return pred_bins * self.num_filters + pred_filters
    