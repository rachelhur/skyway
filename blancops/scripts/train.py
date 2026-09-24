import numpy as np
import matplotlib
import torch

matplotlib.use('Agg')
import time

from blancops.rl.trainer import Trainer
from blancops.utils.sys_utils import get_system_device, seed_everything
from blancops.io.logger_utils import configure_logger
from blancops.data.dataset import TransitionDataset, OfflineDataset
from blancops.data.feature_cache import RawFeatureCache, DatasetCache, dataset_cache_path
from blancops.data.splits import resolve_night_split
from blancops.data.lookup_tables import TrainLookupTables
from blancops.plotting.training_viz import (
    plot_bin_feature_distributions, plot_bin_membership,
    plot_global_feature_distributions, plot_train_metrics,
)
from blancops.rl.registry import build_algorithm
from blancops.configs.experiment_schema import ExperimentConfig, load_and_validate, resolve_and_save
from blancops.configs.constants import DES_DATA_DIR, WORKSPACE
from blancops.configs.enums import Algorithm, CheckpointMetric

import argparse
import gc
import json
import logging
logger = logging.getLogger(__name__)

from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg', type=str, default=None, required=True,
                        help="Path to config file.")
    parser.add_argument('--data_dir', type=str, default=str(DES_DATA_DIR),
                        help="Data directory containing lookups/ and the feature cache.")
    parser.add_argument('-l', '--logging_level', type=str, default='info',
                        help='Logging level.')
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help='Resume training from a checkpoint.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Ignore existing history but keep files.')
    parser.add_argument('--hard_overwrite', action='store_true',
                        help='Completely overwrite existing results.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top runs to keep.')
    parser.add_argument('--dry_run_split', action='store_true',
                        help='Resolve the night split, write configs/split.json, and exit.')
    return parser.parse_args()


def setup_result_outdirs(cfg: ExperimentConfig):
    parent = Path(cfg.parent_dir)
    if not parent.is_absolute():
        parent = WORKSPACE / parent

    if cfg.outdir:
        outdir = Path(cfg.outdir)
        if not outdir.is_absolute():
            outdir = WORKSPACE / outdir
    else:
        outdir = parent / cfg.experiment_name

    outdir.mkdir(parents=True, exist_ok=True)
    for subdir in ('figures', 'checkpoints', 'metrics', 'configs', 'logs'):
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    cfg.outdir = str(outdir)
    return outdir


def _cache_dir(data_dir: Path, nside: int, is_azel: bool) -> Path:
    coord = 'azel' if is_azel else 'radec'
    return data_dir / f"feature_cache_nside{nside}_{coord}"


def main():
    args = get_args()
    cfg = load_and_validate(args.cfg)
    outdir = setup_result_outdirs(cfg)

    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=outdir / "logs",
        filename="train.log",
        use_tqdm=True,
    )

    seed_everything(cfg.train.seed)
    device = get_system_device()

    # --- LOAD FEATURE CACHE --- #
    data_dir = Path(args.data_dir)
    is_azel = 'azel' in cfg.data.action_space
    cache_dir = _cache_dir(data_dir, cfg.data.nside, is_azel)

    if not RawFeatureCache.exists(cache_dir):
        raise FileNotFoundError(
            f"Feature cache not found at {cache_dir}. "
            f"Run `precompute-features --outdir {cache_dir} ...` first."
        )
    if args.dry_run_split:
        night_split = resolve_night_split(
            unique_nights=RawFeatureCache.nights_in_range(
                cache_dir,
                start_date=cfg.data.start_date,
                end_date=cfg.data.end_date,
            ),
            seed=int(cfg.train.seed),
            val_nights=cfg.data.val_nights,
            val_frac=cfg.data.effective_val_frac,
            test_nights=cfg.data.test_nights,
            test_frac=cfg.data.test_frac,
        )
        night_split.save(outdir / "configs" / "split.json")
        logger.info(f"Dry run split: {night_split.counts}")
        return

    logger.info(f"Loading feature cache from {cache_dir}")
    cache = RawFeatureCache.load(
        cache_dir, mmap_bin=True,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    train_lookups = TrainLookupTables.load_from_dir(data_dir / "lookups")

    # --- CONSTRUCT TRAIN DATASET --- #
    train_dataset = TransitionDataset(
        cache=cache,
        cfg=cfg,
        lookups=train_lookups,
    )
    norm_stats = train_dataset.norm_stats
    norm_stats_path = outdir / "checkpoints" / "normalization_stats.json"
    norm_stats_path.write_text(json.dumps(norm_stats.to_dict(), indent=4))
    logger.info(f"Normalization stats saved to {norm_stats_path}")
    logger.info(
        f"Train dataset: {train_dataset.n_nights} nights, "
        f"{train_dataset.num_transitions} transitions"
    )
    cache.log_transition_filter_stats(train_dataset.train_nights, label='Train')

    # --- BUILD SPLIT DATASET CACHES --- #
    night_split = train_dataset.night_split
    night_split.save(outdir / "configs" / "split.json")

    for split in ('val', 'test'):
        split_nights = night_split.nights_for(split)
        if not split_nights:
            logger.info(f"No {split} nights; skipping {split} dataset cache.")
            continue
        split_raw_cache = cache.filter_nights(split_nights, label=split.capitalize())
        split_dataset = TransitionDataset(
            cache=split_raw_cache,
            cfg=cfg,
            lookups=train_lookups,
            norm_stats=norm_stats,
            split_role=split,
        )
        split_cache_path = dataset_cache_path(outdir, split)
        DatasetCache.from_transition_dataset(split_dataset, split=split).save(split_cache_path)
        logger.info(f"{split} dataset cache saved to {split_cache_path}")
        del split_raw_cache, split_dataset
        gc.collect()

    del cache
    gc.collect()
    logger.info("Released feature cache from memory.")

    # --- DEFAULT PLOTS --- #
    plot_bin_membership(train_dataset, outdir / "figures")
    plot_global_feature_distributions(train_dataset, outdir / "figures")
    plot_bin_feature_distributions(train_dataset, outdir / "figures")

    # --- DATALOADERS --- #
    offline_dataset = OfflineDataset(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == 'cuda'),
        seed=cfg.train.seed,
    )
    trainloader = offline_dataset.train_loader
    valloader = offline_dataset.val_loader

    # --- COSINE ANNEALING SCHEDULER KWARGS --- #
    steps_per_epoch = max(int(len(train_dataset) // cfg.train.batch_size), 1)
    num_lr_steps = int(max(1, int(cfg.train.lr_sched_epoch_duration * steps_per_epoch)))
    lr_scheduler_kwargs = (
        {'T_max': num_lr_steps, 'eta_min': float(cfg.train.lr_final)}
        if cfg.train.lr_scheduler == 'cosine_annealing'
        else {}
    )

    # --- BUILD TRAINER --- #
    cfg = resolve_and_save(
        cfg=cfg,
        dataset_dims=train_dataset.dataset_dims,
        dataset_feature_names=train_dataset.dataset_feature_names,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        night_split=night_split,
        outdir=outdir / "configs",
    )
    algorithm = build_algorithm(cfg, device=device)
    latest_ckpt_path = outdir / "checkpoints" / "latest_checkpoint.pt"

    trainer = Trainer(
        algorithm=algorithm,
        train_outdir=outdir,
        top_k=args.top_k,
        overwrite=args.overwrite,
        hard_overwrite=args.hard_overwrite,
        ckpt_metric=cfg.train.checkpoint_metric,
    )

    if latest_ckpt_path.exists() and args.resume_from_checkpoint:
        start_epoch = trainer.resume_from_checkpoint(latest_ckpt_path)
    else:
        start_epoch = 0

    logger.info("Starting training…")
    start_time = time.time()

    trainer.fit(
        start_epoch=start_epoch,
        num_epochs=cfg.train.max_epochs,
        trainloader=trainloader,
        valloader=valloader,
        batch_size=cfg.train.batch_size,
        patience=cfg.train.patience,
        hpGrid=train_dataset.hpGrid,
        norm_stats=norm_stats.to_dict(),
    )
    end_time = time.time()
    logger.info(f"Total train time = {end_time - start_time:.1f}s on {device}")
    logger.info("Training complete.")
    logger.info(f"Results saved in {outdir}")

    logger.info("Plotting metrics…")
    plot_train_metrics(outdir, dataset=train_dataset)

    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"Peak GPU memory: {max_memory:.2f} GB")


if __name__ == "__main__":
    main()
