#%%

import argparse
import os
from pathlib import Path

from matplotlib import pyplot as plt
from blancops.configs import paths
from blancops.io.logger_utils import configure_logger
from blancops.rl.evaluations.evaluator import build_evaluators, plot_metric_distributions_with_ss_overlay
from blancops.configs.experiment_schema import load_and_validate
from blancops.configs.paths import RunPaths
import logging

from blancops.utils.sys_utils import get_system_device


def main():

    # ------------------------------
    # ArgParse
    # ------------------------------

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('c', '--cfg_paths', type=str, nargs='+', help="Paths to config files for each model. ")
    parser.add_argument('-l', '--logging_level', type=str, default='debug', help='Logging level. Options: info, debug')
    parser.add_argument('-f', '--force_overwrite', action='store_true', help='Whether to force overwrite previous rollout files.')
    parser.add_argument('--action_decoding', type=str, default='joint', choices=['joint', 'filter_first'], help='Action decoding strategy to use.')
    parser.add_argument('--save_movies', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--save_mollweides', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Which split to evaluate.')

    args = parser.parse_args()


    # ------------------------------
    # Load config and device
    # ------------------------------

    cfg_paths = args.cfg_paths
    cfg_list = []
    for p in cfg_paths:
        cfg_list.append(load_and_validate(p))
    device = get_system_device()

    # Resolve the model dir from where the config was loaded (machine-portable).
    outdir = paths.workspace().model_comparison()

    # ------------------------------
    # Initialize logger
    # ------------------------------
    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=outdir,
        filename='model_comparison.log',
        use_tqdm=True
    )

    logger.info(f"Comparing models \n {("\t" +  cfg.experiment_name for cfg in cfg_list)}")

    # ------------------------------
    # Build and run evaluators
    # ------------------------------
    logger.info("Building evaluators...")
    ss_list = []
    ms_list = []

    logger.info("Running evaluators...")
    for cfg in cfg_list:
        ss, ms = build_evaluators(
            cfg,
            device=device,
            save_movie=args.save_movies,
            save_mollweide=args.save_mollweides,
            action_decoding=args.action_decoding,
            split=args.split,
        )
        ss.run()
        ms.run(overwrite=args.force_overwrite)

        ss_list.append(ss)
        ms_list.append(ms)

