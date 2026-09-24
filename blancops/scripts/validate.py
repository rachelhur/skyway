#%%

import argparse
import os
from pathlib import Path

from matplotlib import pyplot as plt
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
    parser.add_argument('-c', '--cfg_path', type=str, default=None, help="Path to config file. If passed, all other arguments are ignored")
    parser.add_argument('-l', '--logging_level', type=str, default='debug', help='Logging level. Options: info, debug')
    parser.add_argument('-f', '--force_overwrite', action='store_true', help='Whether to force overwrite previous rollout files.')
    parser.add_argument('--action_decoding', type=str, default='joint', choices=['joint', 'filter_first'], help='Action decoding strategy to use.')
    parser.add_argument('--save_movies', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--save_mollweides', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Which split to evaluate.')

    args = parser.parse_args()

    # ------------------------------
    # Load config and device
    # ------------------------------

    cfg_path = args.cfg_path

    cfg = load_and_validate(cfg_path)
    device = get_system_device()

    # Resolve the model dir from where the config was loaded (machine-portable).
    suffix = '_filter_first' if args.action_decoding == 'filter_first' else ''
    base_eval_dir = RunPaths.from_config(cfg).eval_dir(args.split)
    eval_subdir = f'{base_eval_dir.name}{suffix}'
    outdir = base_eval_dir.with_name(eval_subdir)

    # ------------------------------
    # Initialize logger
    # ------------------------------
    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=outdir,
        filename='validation.log',
        use_tqdm=True
    )

    # ------------------------------
    # Build and run evaluators
    # ------------------------------
    logger.info("Building evaluators...")
    s_eval, m_eval = build_evaluators(
        cfg,
        device=device,
        eval_outdir=eval_subdir,
        save_movie=args.save_movies,
        save_mollweide=args.save_mollweides,
        action_decoding=args.action_decoding,
        split=args.split,
    )

    logger.info("Running evaluators...")
    s_eval.run()
    m_eval.run(overwrite=args.force_overwrite)

    # ------------------------------
    # Default plots
    # ------------------------------
    os.makedirs(outdir / 'ss', exist_ok=True)
    os.makedirs(outdir / 'ms', exist_ok=True)
    fig, ax = s_eval.plot_2dhist_res('ra', 'dec', normalization='probability')
    fig.savefig(outdir / 'ss' / 'ra_vs_dec_res.png')
    fig, ax =m_eval.plot_2dhist_res('ra', 'dec', normalization='probability')
    fig.savefig(outdir / 'ms' / 'ra_vs_dec_res.png')

    m_eval.plot_violin_per_filter('moon_el')
    plt.savefig(outdir / 'ms' / 'filter_strategy_violins_moon_el.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_metric_distributions_with_ss_overlay(ms_evaluator=m_eval, ss_evaluator=s_eval)
    plt.savefig(outdir / 'ms' / 'survey_quality_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    s_eval.plot_filter_confusion()
    plt.savefig(outdir / 'ss' / 'filter_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    filter_figs = m_eval.plot_2dhist_per_filter('ra', 'dec', normalization='probability')
    for filt, filter_fig in filter_figs.items():
        filter_fig.savefig(outdir / 'ms' / f'ra_vs_dec_{filt}.png', dpi=300, bbox_inches='tight')
        plt.close(filter_fig)

    # m_eval.plot_layer1_weights()
    # plt.savefig(outdir / 'ms' / 'layer1_weights.png', dpi=300, bbox_inches='tight')
    # plt.close()

    s_eval.plot_cdf_pointing_error(per_filter=True)
    plt.savefig(outdir / 'ss' / 'cdf_pointing_error_per_filter.png', dpi=300, bbox_inches='tight')
    plt.close()

    s_eval.plot_cdf_pointing_error()
    plt.savefig(outdir / 'ss' / 'cdf_pointing_error.png', dpi=300, bbox_inches='tight')
    plt.close()





