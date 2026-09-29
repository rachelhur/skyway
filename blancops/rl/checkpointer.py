import random

import numpy as np
import torch
from pathlib import Path

import os
import json
import torch
import logging

from blancops.configs.paths import RunPaths

logger = logging.getLogger(__name__)


def _serialize_numpy_rng_state(state):
    # np.random.get_state() returns ('MT19937', np.array([...], uint32), pos, has_gauss, cached_gaussian)
    # Storing the numpy array directly breaks weights_only=True; convert to list.
    return (state[0], state[1].tolist(), state[2], state[3], float(state[4]))


class Checkpointer:
    def __init__(self, outdir: Path, top_k: int = 1, mode: str = 'min', overwrite=False, hard_overwrite=False):
        """
        Args:
            outdir: Directory to save checkpoints.
            top_k: Maximum number of best models to keep on disk.
            mode: 'min' if lower metric is better (e.g., loss, ang_sep), 'max' for accuracy/reward.
        """
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        self.top_k = top_k
        self.mode = mode
        
        # Tracks our best models: list of dicts [{'filepath': str, 'metric': float}]
        self.best_checkpoints = [] 
        self._history_file = self.outdir / RunPaths.CHECKPOINT_HISTORY

        if hard_overwrite:
            self._hard_reset()
        elif overwrite:
            self._soft_reset()
        else:
            self._load_history()

    def _load_history(self):
        """Loads previous checkpoint history if resuming a run."""
        if self._history_file.exists():
            try:
                with open(self._history_file, 'r') as f:
                    self.best_checkpoints = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Could not read checkpoint history. Starting fresh.")

    def _save_history(self):
        with open(self._history_file, 'w') as f:
            json.dump(self.best_checkpoints, f)


    def _is_better(self, a: float, b: float) -> bool:
        """Return True if metric a is better than metric b."""
        if self.mode == "min":
            return a < b
        elif self.mode == "max":
            return a > b
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


    def save_training_state(self, algorithm, epoch: int, metric_value: float, is_best: bool, norm_stats: dict = None):
        """Saves the state and manages the Top-K logic."""
        
        # Create the checkpoint dictionary
        checkpoint = {
            'policy_state_dict': algorithm.policy.state_dict(),
            'optimizer_state_dict': algorithm.optimizer.state_dict(),
            'epoch': epoch,
            'metric': metric_value,
            'norm_stats': norm_stats,
            'rng_states': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': _serialize_numpy_rng_state(np.random.get_state()),
                'python': random.getstate()
            }
        }
        
        # Always save latest for resume
        torch.save(checkpoint, self.outdir / RunPaths.LATEST_CHECKPOINT)
        
        if not is_best:
            return

        # File path for this candidate
        ckpt_path = self.outdir / f"checkpoint_epoch_{epoch:03d}_metric_{metric_value:.4f}.pt"

        # Store only the basename so the history stays portable across machines.
        entry = {
            "filepath": ckpt_path.name,
            "metric": float(metric_value),
        }


        # -------------------------------
        # CASE 1: fewer than K checkpoints
        # -------------------------------
        if len(self.best_checkpoints) < self.top_k:
            torch.save(checkpoint, ckpt_path)
            self.best_checkpoints.append(entry)

            self.best_checkpoints.sort(
                key=lambda x: x["metric"],
                reverse=(self.mode == "max"),
            )
            self._save_history()
            return

        # ------------------------------------
        # CASE 2: compare against current worst
        # ------------------------------------
        worst = self.best_checkpoints[-1]

        if not self._is_better(metric_value, worst["metric"]):
            # New checkpoint is not good enough → discard
            return

        # Replace worst checkpoint. Resolve by basename against outdir so this
        # tolerates both new basename entries and any legacy absolute paths.
        worst_path = self.outdir / Path(worst["filepath"]).name
        if worst_path.exists():
            worst_path.unlink()

        torch.save(checkpoint, ckpt_path)
        self.best_checkpoints[-1] = entry

        # Restore sorted order
        self.best_checkpoints.sort(
            key=lambda x: x["metric"],
            reverse=(self.mode == "max"),
        )

        self._save_history()
        assert 0 < len(self.best_checkpoints) <= self.top_k


    def save_periodic(self, algorithm, path: Path, epoch: int, val_metrics: dict, norm_stats: dict = None) -> None:
        """Save weights for one epoch regardless of the checkpoint metric, so the epoch can be re-picked later.

        Parameters
        ----------
        algorithm : AlgorithmBase
            Algorithm whose policy weights are saved.
        path : Path
            Output file (RunPaths.periodic_checkpoint).
        epoch : int
            Training epoch.
        val_metrics : dict
            Validation metrics of this epoch, metric name -> float.
        norm_stats : dict, optional
            Normalization stats needed to run the policy.
        """
        torch.save({
            'policy_state_dict': algorithm.policy.state_dict(),
            'epoch': epoch,
            'val_metrics': val_metrics,
            'norm_stats': norm_stats,
        }, path)

    def export_deployment_model(self, policy, norm_stats: dict, filename=RunPaths.MODEL_PT):
        """Saves a stripped-down, prefix-free state dict for deployment."""
        raw_state_dict = policy.state_dict()
        clean_state_dict = {}
        
        for key, value in raw_state_dict.items():
            clean_key = key.replace("policy.", "", 1) if key.startswith("policy.") else key
            clean_state_dict[clean_key] = value
        
        deployment_package = {
            'model_state_dict': clean_state_dict,
            'norm_stats': norm_stats
        }
        torch.save(deployment_package, self.outdir / filename)
        logger.info(f"Deployment model and norm stats saved to {filename}")
    
def resolve_weights_path(model_dir: Path, filename: str = None) -> Path:
    """Resolve which weights file to load for a model directory.
    Resolution order:
        1. explicit filename (model_dir/ or model_dir/checkpoints/)
        2. deployment artifact: model.pt (checkpoints/ then model_dir/)
        3. best from history: basename of history[0] joined to checkpoints/
    """
    model_dir = Path(model_dir)
    run_paths = RunPaths(model_dir)
    checkpoints_dir = run_paths.checkpoints

    # 1. User-specified file.
    if filename:
        if (model_dir / filename).exists():
            return model_dir / filename
        return checkpoints_dir / filename

    # 2. Deployment artifact (preferred for live/offline scheduling).
    for model_pt in (run_paths.model_pt, model_dir / RunPaths.MODEL_PT):
        if model_pt.exists():
            return model_pt

    # 3. Best checkpoint from training history (resolve by basename only).
    history_file = run_paths.checkpoint_history
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
        if history:
            # Checkpointer keeps the list sorted so index 0 is the best.
            best = history[0]
            weights_path = checkpoints_dir / Path(best["filepath"]).name
            logger.info(
                f"Auto-detected best checkpoint from history: {weights_path.name} "
                f"(Metric: {best['metric']:.4f})"
            )
            if weights_path.exists():
                return weights_path


    raise FileNotFoundError(f"Could not find any valid weights in {model_dir}")


def get_checkpoint(trained_model_dir, device):
    weights_path = resolve_weights_path(Path(trained_model_dir))

    try:
        checkpoint = torch.load(weights_path, map_location=device)
    except Exception as e:
        logger.warning(f"torch.load failed with default settings: {e}. Retrying with weights_only=False (legacy checkpoint support).")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    return checkpoint