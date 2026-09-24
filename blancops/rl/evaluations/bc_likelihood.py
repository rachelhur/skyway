"""Validation log-likelihood evaluation for trained BC policies.

Typical entry point::

    results = evaluate_run("/path/to/run_dir")

This loads ``checkpoints/model.pt``, ``checkpoints/<split>_dataset_cache.pt``,
and ``configs/resolved_config.yaml`` from the run directory and returns a dict
of log-likelihood metrics over the held-out split.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from blancops.configs.experiment_schema import load_and_validate
from blancops.data.feature_cache import DatasetCache
from blancops.configs.paths import RunPaths
from blancops.rl.agent_factory import AgentFactory

import logging
logger = logging.getLogger(__name__)


def evaluate_run(
    run_dir: Path,
    batch_size: int = 512,
    device: torch.device | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    split: str = 'val',
) -> dict:
    """Compute BC log-likelihood for a completed training run.

    Loads from the standard run-directory layout::

        <run_dir>/checkpoints/model.pt
        <run_dir>/checkpoints/<split>_dataset_cache.pt
        <run_dir>/configs/resolved_config.yaml

    Returns the same dict as compute_bc_loglikelihood.
    """
    run_paths = RunPaths(run_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_and_validate(run_paths.resolved_config)

    policy, _ = AgentFactory.load_policy(
        weights_path=run_paths.model_pt,
        cfg=cfg,
        device=device,
    )

    loader = loader_from_cache(
        run_paths.dataset_cache(split),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return compute_bc_loglikelihood(policy, loader, device=device)


def loader_from_cache(
    cache_path: Path,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader over one split's transitions from a saved DatasetCache.

    Expands per-state tensors to per-transition via curr_compact_idxs,
    mirroring the val_loader in OfflineDataset (shuffle=False, drop_last=False).
    """
    cache = DatasetCache.load(cache_path)
    states_tr, bin_states_tr, actions, masks_tr = cache.to_transition_tensors()

    dataset = TensorDataset(states_tr, actions, masks_tr, bin_states_tr)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def compute_bc_loglikelihood(
    policy,
    val_loader: DataLoader,
    device: torch.device | None = None,
) -> dict:
    """Mean action log-likelihood E_n[ log P(a_expert_n | s_n) ] over val_loader.

    val_loader must yield (state, action, action_mask, bin_state) batches,
    as produced by loader_from_cache.

    Returns
    -------
    dict:
        mean_ll  -- primary scalar metric (higher = better fit to expert)
        std_ll   -- std across observations
        per_obs  -- Tensor (N,) of per-observation log-likelihoods
        n_finite -- observations with finite LL (non-finite = horizon mask mismatch)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "to"):
        policy.to(device)

    chunks = []
    for state, actions, action_masks, bin_states in val_loader:
        g    = state.to(device, dtype=torch.float32)
        b    = bin_states.to(device, dtype=torch.float32)
        a    = actions.to(device).long()
        mask = action_masks.to(device)

        logits    = policy.core_net(g, b)                           # (B, n_bins * n_filters)
        mask_val  = torch.finfo(logits.dtype).min
        logits    = logits.masked_fill(~mask.bool(), mask_val)
        log_probs = F.log_softmax(logits, dim=-1)                   # (B, A)

        ll = log_probs[torch.arange(len(a), device=device), a]      # (B,)
        chunks.append(ll.cpu())

    per_obs = torch.cat(chunks)                                     # (N,)

    finite_mask = torch.isfinite(per_obs)
    n_nonfinite = int((~finite_mask).sum())
    if n_nonfinite > 0:
        N = len(per_obs)
        warnings.warn(
            f"{n_nonfinite}/{N} observations have non-finite log-likelihood. "
            "Check that chosen bins are not masked below-horizon in action_masks."
        )

    finite_ll = per_obs[finite_mask]
    return dict(
        mean_ll=float(finite_ll.mean()),
        std_ll=float(finite_ll.std()),
        per_obs=per_obs,
        n_finite=int(finite_mask.sum()),
    )


# Backwards-compatible alias
val_loader_from_cache = loader_from_cache
