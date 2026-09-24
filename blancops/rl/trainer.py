import operator
from random import random
import gymnasium as gym
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import time
from typing import Tuple
import os
import pickle
import random
from pathlib import Path

from blancops.configs.paths import RunPaths

from blancops.configs.constants import *
import logging

from blancops.configs.enums import Algorithm, CheckpointMetric
from blancops.rl.algorithms.base import AlgorithmBase
from blancops.rl.checkpointer import Checkpointer

logger = logging.getLogger(__name__)
from tqdm.contrib.logging import logging_redirect_tqdm

# Direction and operator for each checkpoint metric.
_CKPT_DIRECTION: dict[CheckpointMetric, str] = {
    CheckpointMetric.VAL_LOSS:           'min',
    CheckpointMetric.ANGULAR_SEPARATION: 'min',
    CheckpointMetric.MAX_Q_POLICY:       'max',
}
_CKPT_INIT_BEST: dict[str, float] = {
    'min': float('inf'),
    'max': float('-inf'),
}
_CKPT_OP = {
    'min': operator.lt,
    'max': operator.gt,
}
# Optional gate: (metric_key, comparator, threshold). Both gate and primary
# metric must pass before a checkpoint is saved. None = no gate.
_CKPT_GATE: dict[CheckpointMetric, tuple | None] = {
    CheckpointMetric.VAL_LOSS:           None,
    CheckpointMetric.ANGULAR_SEPARATION: None,
    CheckpointMetric.MAX_Q_POLICY:       ('td_error', operator.lt, 0.50),
}


class Trainer:
    """
    A simple, generic agent/wrapper for fitting and evaluating RL algorithms.

    This class abstracts training loops, evaluation, saving/loading, and interaction with environment.
    """
    def __init__(
            self,
            algorithm: AlgorithmBase,
            train_outdir: Path,
            top_k: int = 1,
            overwrite: bool = False,
            hard_overwrite: bool = False,
            ckpt_metric: CheckpointMetric = CheckpointMetric.VAL_LOSS,
            ):
        self.algorithm = algorithm
        self.device = algorithm.device
        if not os.path.exists(train_outdir):
            os.makedirs(train_outdir)
        self.train_outdir = Path(train_outdir)
        self.run_paths = RunPaths(self.train_outdir)
        self.overwrite = overwrite
        self.hard_overwrite = hard_overwrite
        self.ckpt_metric = ckpt_metric
        self.checkpointer = Checkpointer(
            self.run_paths.checkpoints,
            top_k=top_k,
            mode=_CKPT_DIRECTION[ckpt_metric],
            overwrite=overwrite,
            hard_overwrite=hard_overwrite,
        )

    def _validate_valloader(self, valloader):
        if len(valloader) == 0:
            raise ValueError("Validation dataloader is empty! Check dataset split logic.")

    def fit(self, num_epochs, batch_size, trainloader, valloader, patience=10,
            train_log_freq=10, hpGrid=None, norm_stats=None, start_epoch=None):

        if (self.overwrite or self.hard_overwrite) and start_epoch > 0:
            raise ValueError("Cannot overwrite checkpoints and resume from a previous epoch.")

        self._validate_valloader(valloader)

        val_metrics = defaultdict(list)
        train_metrics = defaultdict(list)
        train_metrics_filepath = self.run_paths.train_metrics
        val_metrics_filepath = self.run_paths.val_metrics

        # --- Reload previous metric histories if resuming ---
        if start_epoch > 0:
            if train_metrics_filepath.exists():
                with open(train_metrics_filepath, 'rb') as f:
                    train_metrics.update(pickle.load(f))
            if val_metrics_filepath.exists():
                with open(val_metrics_filepath, 'rb') as f:
                    val_metrics.update(pickle.load(f))

        # Set to train mode
        self.algorithm.policy.train()

        dataset_size = len(trainloader.dataset)
        steps_per_epoch = np.max([dataset_size // batch_size, 1])

        total_steps = int(num_epochs * steps_per_epoch)
        start_step = int(start_epoch * steps_per_epoch)

        loader_iter = iter(trainloader)

        metric_key = self.ckpt_metric.value                  # string key into val_metrics
        direction = _CKPT_DIRECTION[self.ckpt_metric]
        best_metric_val = _CKPT_INIT_BEST[direction]
        compare = _CKPT_OP[direction]

        best_epoch = start_epoch
        patience_cur = patience
        use_patience = patience != 0

        i_epoch = start_epoch

        logger.debug(f"Total number of training steps: {total_steps}")
        logger.debug(f"Steps per epoch: {steps_per_epoch}")
        logger.debug(f"Resuming from step: {start_step} (Epoch {start_epoch})")
        logger.debug(f"Number of transitions in dataset: {len(trainloader.dataset)}")

        with logging_redirect_tqdm():
            pbar = tqdm(initial=start_step, total=total_steps, dynamic_ncols=True, desc="Training")

            for i_step in range(start_step, total_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(trainloader)
                    batch = next(loader_iter)

                # Because of math, if we resume at epoch 10, start_step is an exact multiple
                # of steps_per_epoch. This will instantly bump i_epoch to 11, which is correct!
                if i_step % steps_per_epoch == 0:
                    i_epoch += 1

                pbar.update(1)
                pbar.set_description(f"Epoch {i_epoch}/{int(num_epochs)} (step {i_step}/{total_steps})")

                # Train step -- currently logs at each epoch
                log_metrics = i_step % steps_per_epoch == 0
                train_metrics_dict = self.algorithm.train_step(batch, epoch_num=i_epoch, hpGrid=hpGrid, compute_metrics=log_metrics)
                if log_metrics:
                    for k, v in train_metrics_dict.items():
                        train_metrics[k].append(v)

                    train_metrics['lr'].append(self.algorithm.optimizer.param_groups[0]["lr"])
                    train_metrics['epoch'].append(i_epoch)

                # Validation step
                with torch.no_grad():
                    if log_metrics:
                        val_metric_sums = defaultdict(float)
                        num_val_batches = len(valloader)

                        for eval_batch in valloader:
                            batch_metrics = self.algorithm.val_step(eval_batch, hpGrid)
                            for k, v in batch_metrics.items():
                                val_metric_sums[k] += v

                        # Average and save the metrics
                        val_log_str_parts = []
                        for k, total in val_metric_sums.items():
                            avg_val = total / num_val_batches
                            val_metrics[k].append(avg_val)
                            val_log_str_parts.append(f"{k} = {avg_val:.3f}")
                        val_metrics['epoch'].append(i_epoch)

                        # Log comparison
                        val_log_str = " | ".join(val_log_str_parts)
                        train_log_str = " | ".join(f"{k} = {v:.3f}" for k, v in train_metrics_dict.items())

                        logger.info(
                            f"\nValidation check at train step {i_step} \n"
                            f" (val set)      {val_log_str} \n"
                            f" (train batch)  {train_log_str}"
                        )

                        # Early stopping and model saving
                        ckpt_vals = val_metrics.get(metric_key)
                        if ckpt_vals is None:
                            logger.warning(
                                f"Checkpoint metric '{metric_key}' not in val metrics "
                                f"(hpGrid missing?); skipping improvement check."
                            )
                            improved = False
                        else:
                            ckpt_metric_val_cur = ckpt_vals[-1]
                            improved = compare(ckpt_metric_val_cur, best_metric_val)
                            gate = _CKPT_GATE[self.ckpt_metric]
                            if improved and gate is not None:
                                gate_key, gate_op, gate_thresh = gate
                                gate_vals = val_metrics.get(gate_key)
                                if gate_vals is None:
                                    logger.warning(f"Gate metric '{gate_key}' not in val metrics; gate skipped.")
                                else:
                                    gate_passed = gate_op(gate_vals[-1], gate_thresh)
                                    if not gate_passed:
                                        logger.debug(
                                            f"Gate failed: {gate_key}={gate_vals[-1]:.3f} not "
                                            f"{gate_op.__name__} {gate_thresh}. Skipping checkpoint."
                                        )
                                    improved = gate_passed
                        if improved:
                            best_metric_val = ckpt_metric_val_cur
                            best_epoch = i_epoch
                            metric_str = f"{metric_key}={best_metric_val:.3f}"
                            patience_cur = patience
                            logger.info(f'Improved model at step {i_step} (epoch {i_epoch}): {metric_str}. Saving weights')

                            self.checkpointer.save_training_state(
                                algorithm=self.algorithm,
                                epoch=i_epoch,
                                metric_value=best_metric_val,
                                is_best=True,
                                norm_stats=norm_stats
                            )
                            self.checkpointer.export_deployment_model(self.algorithm.policy, norm_stats=norm_stats)

                            with open(train_metrics_filepath, 'wb') as handle:
                                pickle.dump(train_metrics, handle)
                            with open(val_metrics_filepath, 'wb') as handle:
                                pickle.dump(val_metrics, handle)
                        elif use_patience:
                            patience_cur -= 1
                            logger.debug(f"Patience left: {patience_cur}")
                            if patience_cur == 0:
                                logger.info("No patience left. Ending training.")
                                break
        logger.info(f"Best {metric_key}={best_metric_val:.3f} at epoch {best_epoch}")
        with open(train_metrics_filepath, 'wb') as handle:
            pickle.dump(train_metrics, handle)
        with open(val_metrics_filepath, 'wb') as handle:
            pickle.dump(val_metrics, handle)

    def resume_from_checkpoint(self, checkpoint_path: Path):
        """Loads weights, optimizer states, and restores all random number generators."""
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
            return 0 # Return epoch 0

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 1. Load your model and optimizer weights
        # (You might need to adjust this depending on how algorithm.load() works)
        self.algorithm.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.algorithm.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 2. Restore all RNG states
        if 'rng_states' in checkpoint:
            rng = checkpoint['rng_states']
            try:
                # Restore PyTorch CPU RNG state. Torch expects a ByteTensor on CPU.
                torch_state = rng.get('torch')
                if torch_state is not None:
                    if not (isinstance(torch_state, torch.Tensor) and torch_state.dtype == torch.uint8):
                        # convert numpy array / list -> ByteTensor
                        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
                    torch_state = torch_state.cpu()
                    torch.set_rng_state(torch_state)

                # Restore CUDA RNG state if present and available
                torch_cuda_state = rng.get('torch_cuda', None)
                if torch_cuda_state is not None and torch.cuda.is_available():
                    if not (isinstance(torch_cuda_state, torch.Tensor) and torch_cuda_state.dtype == torch.uint8):
                        torch_cuda_state = torch.tensor(torch_cuda_state, dtype=torch.uint8)
                    # set on CUDA
                    torch_cuda_state = torch_cuda_state.cpu()
                    try:
                        torch.cuda.set_rng_state(torch_cuda_state)
                    except Exception:
                        # Newer torch may require different handling; skip if failing
                        logger.debug('Could not set CUDA RNG state from checkpoint; continuing.')

                # Restore numpy RNG state if available
                numpy_state = rng.get('numpy', None)
                if numpy_state is not None:
                    try:
                        if isinstance(numpy_state[1], list):  # new format: array serialized as list
                            numpy_state = (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                                           numpy_state[2], numpy_state[3], numpy_state[4])
                        np.random.set_state(numpy_state)
                    except Exception:
                        logger.debug('Failed to restore NumPy RNG state from checkpoint; skipping.')

                # Restore python random state if available
                python_state = rng.get('python', None)
                if python_state is not None:
                    try:
                        random.setstate(python_state)
                    except Exception:
                        logger.debug('Failed to restore python RNG state from checkpoint; skipping.')

                logger.info('Successfully restored available RNG states from checkpoint (best-effort).')
            except Exception as e:
                logger.warning(f'Failed to fully restore RNG states from checkpoint: {e}. Continuing without full RNG restoration.')

        # Return the epoch so your fit() loop knows where to pick up!
        return checkpoint.get('epoch', 0)



    def _setup_run(self, trainloader, batch_size, num_epochs, patience):
        raise NotImplementedError
        val_metrics = defaultdict(list)
        train_metrics = defaultdict(list)

        # Set to train mode
        self.algorithm.policy.train()
        save_filepath = self.train_outdir / 'best_weights.pt'
        train_metrics_filepath = self.run_paths.train_metrics
        val_metrics_filepath = self.run_paths.val_metrics

        dataset_size = len(trainloader.dataset)
        steps_per_epoch = np.max([dataset_size // batch_size, 1])
        total_steps = int(num_epochs * steps_per_epoch) # ie, total number of times dataset is sampled
        loader_iter = iter(trainloader)  # create iterator

        use_best_val_loss = self.algorithm.name == 'BC'
        use_best_ang_sep = self.algorithm.name in ["DDQN", "DQN", "CQL"]
        assert use_best_val_loss or use_best_ang_sep, "Algorithm name is not valid. Check config file."
        best_val_loss = 1e5
        best_ang_sep = 1e5
        best_epoch = 0
        patience_cur = patience
        use_patience = patience != 0
        i_epoch = 0

        # total_lr_scheduler_steps = int(args.lr_scheduler_max_epochs * iterations_per_epoch // args.lr_scheduler_step_freq)
        logger.info(f"Total number of training steps: {total_steps}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.debug(f"Total number of lr scheduler steps: {self.algorithm.lr_scheduler_num_epochs if self.algorithm.lr_scheduler is not None else None}")
        logger.info(f"Number of transitions in dataset: {len(trainloader.dataset)}")
