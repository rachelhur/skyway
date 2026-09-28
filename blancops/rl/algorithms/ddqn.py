import numpy as np
import torch

from blancops.configs.enums import Algorithm
from blancops.ephemerides.ephemerides import HealpixGrid
from blancops.rl.algorithms.base import AlgorithmBase

import logging
logger = logging.getLogger(__name__)


class DDQN(AlgorithmBase):
    """(Double) DQN.

    Set `use_double=False` to fall back to vanilla DQN. The CQL subclass
    extends this by adding a conservative penalty in `_compute_loss`.
    """
    name = Algorithm.DDQN

    def __init__(
        self,
        policy,
        target,
        optimizer,
        loss_function,
        gamma: float = 0.99,
        tau: float = 0.005,
        use_double: bool = True,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        lr_scheduler_epoch_start: int = 1,
        lr_scheduler_num_epochs: int = 50,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            policy=policy,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_epoch_start=lr_scheduler_epoch_start,
            lr_scheduler_num_epochs=lr_scheduler_num_epochs,
            device=device,
        )
        
        assert loss_function is not None, "loss_function must be provided"

        self.loss_function = loss_function
        self.gamma = gamma
        self.tau = tau
        self.use_double = use_double

        self.target_net = target.to(device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

    # ----------------------------------------------------------------------- #
    # Hook implementations
    # ----------------------------------------------------------------------- #

    def _unpack_batch(self, batch) -> dict:
        
        (state, actions_flat, rewards, next_state, 
         dones, action_masks, next_action_masks, candidate_states, next_candidate_states, slew_dists) = batch

        return {
            "state":                 self._to_dev(state, torch.float32),
            "next_state":            self._to_dev(next_state, torch.float32),
            "candidate_states":      self._to_dev(candidate_states, torch.float32),
            "next_candidate_states": self._to_dev(next_candidate_states, torch.float32),
            "actions":               self._to_dev(actions_flat, torch.long).unsqueeze(1),
            "rewards":               self._to_dev(rewards, torch.float32),
            "dones":                 self._to_dev(dones, torch.float32),
            "action_masks":          self._to_dev(action_masks, torch.bool),
            "next_action_masks":     self._to_dev(next_action_masks, torch.bool),
        }

    def _compute_loss(self, batch_dict, candidate_grid=None, compute_metrics=False):
        q_vals_all, q_val, q_expected = self._forward_q(batch_dict)
        loss = self._td_loss(q_val, q_expected)

        metrics = {}
        if compute_metrics:
            metrics = self._build_metrics(q_vals_all, q_val, q_expected, batch_dict, candidate_grid)
        return loss, metrics

    def _post_step(self) -> None:
        self._soft_update()

    # ----------------------------------------------------------------------- #
    # Q-learning math — broken out so CQL can reuse it
    # ----------------------------------------------------------------------- #

    def _forward_q(self, batch_dict):
        """Forward pass: current Q values, taken-action Q, and TD target."""
        state                 = batch_dict["state"]
        candidate_states      = batch_dict["candidate_states"]
        next_state            = batch_dict["next_state"]
        next_candidate_states = batch_dict["next_candidate_states"]
        actions               = batch_dict["actions"]
        rewards               = batch_dict["rewards"]
        dones                 = batch_dict["dones"]
        next_masks            = batch_dict["next_action_masks"]

        q_vals_all = self.policy.get_q_values(state, candidate_states)
        q_val = q_vals_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                q_vals_next = self.policy.get_q_values(next_state, next_candidate_states)
                mask_val = torch.finfo(q_vals_next.dtype).min
                q_vals_next = q_vals_next.masked_fill(~next_masks, mask_val)
                a_best = q_vals_next.argmax(1)

                target_q_next = self.target_net.get_q_values(next_state, next_candidate_states)
                target_q_state = target_q_next.gather(1, a_best.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net.get_q_values(next_state, next_candidate_states)
                mask_val = torch.finfo(next_q.dtype).min
                next_q = next_q.masked_fill(~next_masks, mask_val)
                target_q_state = next_q.max(dim=1)[0]

            q_expected = rewards + self.gamma * target_q_state * (1 - dones)

        return q_vals_all, q_val, q_expected

    def _td_loss(self, q_val, q_expected) -> torch.Tensor:
        return self.loss_function(q_val, q_expected)

    def _build_metrics(self, q_vals_all, q_val, q_expected, batch_dict, candidate_grid):
        actions = batch_dict["actions"]
        action_masks = batch_dict["action_masks"]

        # Warn if any expert action is masked invalid (mainly a sanity check).
        expert_squeezed = actions.squeeze(1)
        invalid = ~action_masks[torch.arange(action_masks.size(0)), expert_squeezed]
        if invalid.any():
            logger.debug(
                f"{invalid.sum().item()} expert actions in this batch are masked invalid"
            )

        q_eval = q_vals_all.clone()
        mask_val = torch.finfo(q_eval.dtype).min
        q_eval = q_eval.masked_fill(~action_masks, mask_val)
        predicted_actions = q_eval.argmax(1)

        metrics = {
            "td_error": (q_val - q_expected).abs().mean().item(),
            "td_loss":  self._td_loss(q_val, q_expected).item(),
            "q_std":    q_vals_all.std().item(),
            "q_policy": q_vals_all.max(dim=1)[0].mean().item(),
            "q_expert": q_val.mean().item(),
            "accuracy": (predicted_actions == expert_squeezed).float().mean().item(),
        }

        if candidate_grid is not None:
            heavy = self.policy.compute_heavy_metrics(
                predicted_actions, expert_squeezed, candidate_grid, self.policy.num_filters
            )
            metrics.update(heavy)
        return metrics

    # ----------------------------------------------------------------------- #
    # Target network update
    # ----------------------------------------------------------------------- #

    def _soft_update(self):
        for tgt, src in zip(self.target_net.parameters(), self.policy.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)


def calculate_distance_matrix(nside, is_azel):
    hpGrid = HealpixGrid(nside, is_azel)
    lons, lats = hpGrid.lon, hpGrid.lat
    dist = np.zeros((len(lons), len(lons)))
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        dist[i] = hpGrid.get_angular_separations(lon, lat)
    return dist