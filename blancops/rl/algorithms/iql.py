import torch
import torch.nn.functional as F

from blancops.configs.enums import Algorithm
from blancops.rl.algorithms.base import AlgorithmBase

import logging
logger = logging.getLogger(__name__)


class IQL(AlgorithmBase):
    """Implicit Q-Learning: Loss = Q_loss + V_loss + π_loss
    Learns the state-value function via expectile regression to
    estimate the Q-value function.
    See [https://arxiv.org/abs/2110.06169]
    """
    name = Algorithm.IQL
    def __init__(
        self,
        q_adapter, # QFlatPolicy adapting Q-net to flat (B, |A|) Q-values
        q_target_adapter, # QFlatPolicy wrapping frozen target Q-net
        v_net, # MLP state-value network: s -> (B, 1)
        policy_net, # QFlatPolicy wrapping policy net; saved by checkpointer
        optimizer, # optimizer over q, v, and policy params
        loss_function,
        gamma: float = 0.99,
        tau: float = 0.005, # target Q-net soft-update rate
        expectile: float = 0.7, # tau in IQL paper — V regression upper-expectile
        awr_beta: float = 3.0, # AWR temperature
        awr_clip: float = 100.0, # advantage-exp clip ceiling (prevents blowup)
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        lr_scheduler_epoch_start: int = 1,
        lr_scheduler_num_epochs: int = 50,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            policy=policy_net, # AlgorithmBase stores as self.policy; saved by checkpointer
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_epoch_start=lr_scheduler_epoch_start,
            lr_scheduler_num_epochs=lr_scheduler_num_epochs,
            device=device,
        )

        assert loss_function is not None, "loss_function must be provided for Q-regression"
        assert 0.5 < expectile < 1.0, (
            f"expectile must be in (0.5, 1.0); got {expectile}. "
            "0.5 recovers MSE; higher values upper-bound V toward max-Q."
        )

        self.q_adapter = q_adapter.to(device)
        self.v_net = v_net.to(device)
        self.policy_net = policy_net  # same object as self.policy; named alias for clarity

        self.q_target_adapter = q_target_adapter.to(device)
        self.q_target_adapter.eval()
        for p in self.q_target_adapter.parameters():
            p.requires_grad = False

        self.loss_function = loss_function
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.awr_beta = awr_beta
        self.awr_clip = awr_clip


    # ----------------------------------------------------------------------- #
    # Multi-net train/val overrides (AlgorithmBase only handles self.policy)
    # ----------------------------------------------------------------------- #

    def train_step(self, batch, epoch_num, step_num=None, candidate_grid=None, compute_metrics=False) -> dict:
        self.q_adapter.train()
        self.v_net.train()
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch_dict = self._unpack_batch(batch)
        with torch.amp.autocast(self.device_type_str, dtype=self.amp_dtype):
            loss, metrics = self._compute_loss(batch_dict, candidate_grid=candidate_grid, compute_metrics=compute_metrics)

        loss.backward()
        all_params = [*self.q_adapter.parameters(), *self.v_net.parameters(), *self.policy.parameters()]
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self.optimizer.step()
        self._scheduler_step(epoch_num)
        self._post_step()

        metrics["train_loss"] = loss.item()
        return metrics

    def val_step(self, batch, candidate_grid=None) -> dict:
        self.q_adapter.eval()
        self.v_net.eval()
        self.policy.eval()
        batch_dict = self._unpack_batch(batch)

        with torch.no_grad():
            with torch.amp.autocast(self.device_type_str, dtype=self.amp_dtype):
                loss, metrics = self._compute_loss(batch_dict, candidate_grid=candidate_grid, compute_metrics=True)

        metrics["val_loss"] = loss.item()
        return metrics

    # ----------------------------------------------------------------------- #
    # Hook implementations
    # ----------------------------------------------------------------------- #

    def _unpack_batch(self, batch) -> dict:
        (state, actions, rewards, next_state, dones,
         action_masks, next_action_masks, candidate_states, next_candidate_states, _) = batch

        return {
            "state":                 self._to_dev(state, torch.float32),
            "next_state":            self._to_dev(next_state, torch.float32),
            "candidate_states":      self._to_dev(candidate_states, torch.float32),
            "next_candidate_states": self._to_dev(next_candidate_states, torch.float32),
            "actions":               self._to_dev(actions, torch.long).unsqueeze(1),
            "rewards":               self._to_dev(rewards, torch.float32),
            "dones":                 self._to_dev(dones, torch.float32),
            "action_masks":          self._to_dev(action_masks, torch.bool),
            "next_action_masks":     self._to_dev(next_action_masks, torch.bool),
        }

    def _compute_loss(self, batch_dict, candidate_grid=None, compute_metrics=False):
        # Three separate loss terms, summed for a single backward pass.
        v_loss, v_pred, q_target_taken = self._compute_v_loss(batch_dict)
        q_loss, q_pred_taken, q_expected = self._compute_q_loss(batch_dict)
        pi_loss, advantages = self._compute_policy_loss(
            batch_dict, q_target_taken=q_target_taken, v_pred=v_pred
        )

        loss = q_loss + v_loss + pi_loss

        metrics: dict = {}
        if compute_metrics:
            metrics = self._build_metrics(
                batch_dict,
                q_loss=q_loss,
                v_loss=v_loss,
                pi_loss=pi_loss,
                q_pred_taken=q_pred_taken,
                q_expected=q_expected,
                v_pred=v_pred,
                advantages=advantages,
                candidate_grid=candidate_grid,
            )
        return loss, metrics

    def _post_step(self) -> None:
        self._soft_update_target_q()

    # ----------------------------------------------------------------------- #
    # IQL math, broken into separable pieces
    # ----------------------------------------------------------------------- #

    def _compute_v_loss(self, batch_dict):
        """V(s) regresses toward the upper τ-expectile of Q_target(s, a_dataset).

        Using the *target* Q here (not the online Q) is what makes the
        objective stable — V tracks a slowly-moving Q.
        """
        state = batch_dict["state"]
        candidate_states = batch_dict["candidate_states"]
        actions = batch_dict["actions"]

        with torch.no_grad():
            q_target_all = self.q_target_adapter.get_q_values(state, candidate_states)
            q_target_taken = q_target_all.gather(1, actions).squeeze(1)

        v_pred = self.v_net(x_glob=state, x_cand=candidate_states).squeeze(-1)
        diff = q_target_taken - v_pred  # advantage estimate
        v_loss = _expectile_loss(diff, self.expectile).mean()
        return v_loss, v_pred, q_target_taken

    def _compute_q_loss(self, batch_dict):
        """Q(s, a) regresses toward r + γ · V(s'). No max over next actions —
        this is what keeps IQL implicit and OOD-safe."""
        state = batch_dict["state"]
        candidate_states = batch_dict["candidate_states"]
        next_state = batch_dict["next_state"]
        next_candidate_states = batch_dict["next_candidate_states"]
        actions = batch_dict["actions"]
        rewards = batch_dict["rewards"]
        dones = batch_dict["dones"]

        q_all = self.q_adapter.get_q_values(state, candidate_states)
        q_pred_taken = q_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            v_next = self.v_net(x_glob=next_state, x_cand=next_candidate_states).squeeze(-1)
            q_expected = rewards + self.gamma * v_next * (1 - dones)

        q_loss = self.loss_function(q_pred_taken, q_expected)
        return q_loss, q_pred_taken, q_expected

    def _compute_policy_loss(self, batch_dict, *, q_target_taken, v_pred):
        """Advantage-Weighted Regression: weight the cross-entropy loss
        toward dataset actions by exp(β · advantage). High-advantage actions
        get higher weight; the clip prevents a few outliers from dominating.
        """
        state = batch_dict["state"]
        candidate_states = batch_dict["candidate_states"]
        actions = batch_dict["actions"]
        action_masks = batch_dict["action_masks"]

        with torch.no_grad():
            adv = q_target_taken - v_pred
            weights = torch.exp(self.awr_beta * adv).clamp(max=self.awr_clip)

        action_logits = self.policy_net.get_q_values(state, candidate_states)
        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        # Per-sample cross-entropy, weighted by AWR weights.
        ce_per_sample = F.cross_entropy(
            action_logits, actions.squeeze(1), reduction="none"
        )
        pi_loss = (weights * ce_per_sample).mean()
        return pi_loss, adv

    # ----------------------------------------------------------------------- #
    # Metrics
    # ----------------------------------------------------------------------- #

    def _build_metrics(
        self, batch_dict,
        q_loss, v_loss, pi_loss,
        q_pred_taken, q_expected,
        v_pred, advantages,
        candidate_grid,
    ):

        state = batch_dict["state"]
        candidate_states = batch_dict["candidate_states"]
        actions = batch_dict["actions"]
        action_masks = batch_dict["action_masks"]
        expert_squeezed = actions.squeeze(1)

        # Argmax over the *policy* (not Q) — this is what IQL actually deploys.
        with torch.no_grad():
            policy_logits = self.policy_net.get_q_values(state, candidate_states)
            policy_logits = policy_logits.masked_fill(
                ~action_masks, torch.finfo(policy_logits.dtype).min
            )
            predicted_actions = policy_logits.argmax(dim=1)

            # Q stats for diagnostics
            q_all = self.q_adapter.get_q_values(state, candidate_states)

        metrics = {
            "q_loss":   q_loss.item(),
            "v_loss":   v_loss.item(),
            "pi_loss":  pi_loss.item(),
            "td_error": (q_pred_taken - q_expected).abs().mean().item(),
            "q_expert": q_pred_taken.mean().item(),
            "q_policy": q_all.max(dim=1)[0].mean().item(),
            "q_std":    q_all.std().item(),
            "v_mean":   v_pred.mean().item(),
            "adv_mean": advantages.mean().item(),
            "adv_std":  advantages.std().item(),
            "accuracy": (predicted_actions == expert_squeezed).float().mean().item(),
        }

        if candidate_grid is not None:
            heavy = self.policy.compute_heavy_metrics(
                predicted_actions, expert_squeezed, candidate_grid, self.policy.num_filters
            )
            metrics.update(heavy)

        return metrics

    # ----------------------------------------------------------------------- #
    # Target update
    # ----------------------------------------------------------------------- #

    def _soft_update_target_q(self):
        for tgt, src in zip(
            self.q_target_adapter.parameters(), self.q_adapter.parameters()
        ):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)


# --------------------------------------------------------------------------- #
# Expectile loss
# --------------------------------------------------------------------------- #

def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric L2: penalizes positive residuals (diff > 0) with weight
    `expectile` and negative ones with `1 - expectile`. At τ=0.5 this is
    plain MSE / 2; at τ=0.7 it pushes V toward an upper expectile of Q,
    which approximates max-Q without ever evaluating OOD actions."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)