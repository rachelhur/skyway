from abc import ABC, abstractmethod

import torch

import logging
logger = logging.getLogger(__name__)


def q_value_metrics(q_all: torch.Tensor, q_taken: torch.Tensor, q_target: torch.Tensor,
                    action_masks: torch.Tensor) -> dict:
    """Q-scale diagnostics over valid actions, shared by the value-based algorithms.

    rel_td_error = mean|Q(s,a) - y| / mean|Q(s,a)|; q_gap = mean_s max_{valid a} Q(s,a) - mean Q(s,a_data).

    Parameters
    ----------
    q_all : torch.Tensor
        Q-values for every action, shape (batch, n_actions).
    q_taken : torch.Tensor
        Q of the dataset action, shape (batch,).
    q_target : torch.Tensor
        Bellman target y for the dataset action, shape (batch,).
    action_masks : torch.Tensor
        Valid-action mask, bool, shape (batch, n_actions).

    Returns
    -------
    dict
        rel_td_error, q_policy, q_gap, q_max, q_min, q_target_mean as floats.
    """
    q_all, q_taken, q_target = q_all.float(), q_taken.float(), q_target.float()                # reduce in fp32 under autocast
    q_valid_max = q_all.masked_fill(~action_masks, torch.finfo(q_all.dtype).min).max(dim=1)[0]  # [batch]
    q_valid = q_all[action_masks]                                                             # [n_valid]
    td_error = (q_taken - q_target).abs().mean()
    q_policy = q_valid_max.mean()
    return {
        "rel_td_error":  (td_error / q_taken.abs().mean().clamp_min(1e-8)).item(),
        "q_policy":      q_policy.item(),
        "q_gap":         (q_policy - q_taken.mean()).item(),
        "q_max":         q_valid.max().item(),
        "q_min":         q_valid.min().item(),
        "q_target_mean": q_target.mean().item(),
    }


class AlgorithmBase(ABC):
    """Owns the optimizer/scheduler lifecycle and the train/val step template.

    Subclasses fill in three hooks:
      * `_unpack_batch(batch) -> dict`            — algorithm-specific tensors
      * `_compute_loss(batch_dict, ...) -> (loss, metrics_dict)`
      * `_post_step()` (optional)                 — e.g. target-net soft update
    """
    def __init__(
        self, 
        policy, 
        optimizer, 
        lr_scheduler, 
        lr_scheduler_epoch_start=1, 
        lr_scheduler_num_epochs=50, 
        optimizer_kwargs=None, 
        lr_scheduler_kwargs=None, 
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.device_type_str = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        self.amp_dtype = torch.bfloat16
        
        self.policy = policy.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = self._initialize_scheduler(lr_scheduler, lr_scheduler_kwargs, self.optimizer)
        
        self.lr_scheduler_epoch_start = lr_scheduler_epoch_start
        self.lr_scheduler_num_epochs = lr_scheduler_num_epochs
    
    # ----------------------------------------------------------------------- #
    # Public API: template methods. Subclasses don't override these.
    # ----------------------------------------------------------------------- #

    def train_step(
        self, batch, epoch_num, step_num=None, candidate_grid=None, compute_metrics=False) -> dict:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch_dict = self._unpack_batch(batch)

        with torch.amp.autocast(self.device_type_str, dtype=self.amp_dtype):
            loss, metrics = self._compute_loss(
                batch_dict, candidate_grid=candidate_grid, compute_metrics=compute_metrics
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._scheduler_step(epoch_num)
        self._post_step()

        metrics["train_loss"] = loss.item()
        return metrics

    def val_step(self, batch, candidate_grid=None) -> dict:
        self.policy.eval()
        batch_dict = self._unpack_batch(batch)

        with torch.no_grad():
            with torch.amp.autocast(self.device_type_str, dtype=self.amp_dtype):
                loss, metrics = self._compute_loss(
                    batch_dict, candidate_grid=candidate_grid, compute_metrics=True
                )

        metrics["val_loss"] = loss.item()
        return metrics


    # ----------------------------------------------------------------------- #
    # Hooks: subclasses implement these.
    # ----------------------------------------------------------------------- #

    @abstractmethod
    def _unpack_batch(self, batch) -> dict:
        ...

    @abstractmethod
    def _compute_loss(
        self, batch_dict: dict, candidate_grid=None, compute_metrics: bool = False) -> tuple[torch.Tensor, dict]:
        """Return (loss_tensor, metrics_dict). metrics_dict may be empty if
        compute_metrics is False."""
        ...

    def _post_step(self) -> None:
        """Optional hook for things like target-network soft updates."""
        pass
    
    # ----------------------------------------------------------------------- #
    # Shared utilities
    # ----------------------------------------------------------------------- #

    def _initialize_scheduler(self, lr_scheduler, lr_scheduler_kwargs, optimizer):
        if lr_scheduler is None:
            return None
        if lr_scheduler in ("cosine_annealing", torch.optim.lr_scheduler.CosineAnnealingLR):
            assert lr_scheduler_kwargs is not None, (
                "Cosine annealing scheduler requires T_max and eta_min kwargs"
            )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, **lr_scheduler_kwargs
            )
        raise NotImplementedError(f"Scheduler {lr_scheduler!r} not implemented.")

    def _scheduler_step(self, epoch_num: int) -> None:
        if self.lr_scheduler is None:
            return
        in_window = (
            self.lr_scheduler_epoch_start
            <= epoch_num
            <= self.lr_scheduler_epoch_start + self.lr_scheduler_num_epochs
        )
        if in_window:
            self.lr_scheduler.step()

    def _to_dev(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=dtype)