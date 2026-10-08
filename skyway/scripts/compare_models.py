#%%

import argparse

from matplotlib import pyplot as plt
from skyway.configs import paths
from skyway.io.logger_utils import configure_logger
from skyway.rl.evaluations.evaluator import build_evaluators
from skyway.rl.evaluations import survey_metrics as sm
from skyway.configs.experiment_schema import load_and_validate
from skyway.configs.paths import RunPaths
import logging

from skyway.utils.sys_utils import get_system_device


def main():

    # ------------------------------
    # ArgParse
    # ------------------------------

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_paths', type=str, nargs='+', required=True, help="Paths to config files for each model. ")
    parser.add_argument('-l', '--logging_level', type=str, default='debug', help='Logging level. Options: info, debug')
    parser.add_argument('-f', '--force_overwrite', action='store_true', help='Whether to force overwrite previous rollout files.')
    parser.add_argument('--action_decoding', type=str, default='joint', choices=['joint', 'filter_first'], help='Action decoding strategy to use.')
    parser.add_argument('--save_movies', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--save_mollweides', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--plot_bins', action='store_true',
                        help='Also draw HEALPix bins in movies of field-level models (bin-level models always draw them).')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Which split to evaluate.')
    parser.add_argument('--baselines', action='store_true',
                        help='Also run the random and min_slew heuristics in the depth-vs-uniformity figure.')

    args = parser.parse_args()

    # ------------------------------
    # Load configs and device
    # ------------------------------

    cfg_list = [load_and_validate(p) for p in args.cfg_paths]
    run_paths = [RunPaths.from_config(cfg) for cfg in cfg_list]
    roots = [rp.root.resolve() for rp in run_paths]
    if len(set(roots)) < len(roots):
        raise ValueError(f"The same run folder was passed more than once: {roots}")
    device = get_system_device()

    labels = sm.unique_labels([cfg.model.algorithm.name for cfg in cfg_list],
                              [cfg.experiment_name for cfg in cfg_list], roots)
    names = [r.name for r in roots]
    run_names = sorted(f"{r.parent.name}-{r.name}" if names.count(r.name) > 1 else r.name for r in roots)
    suffix = ('_filter_first' if args.action_decoding == 'filter_first' else '') + ('_baselines' if args.baselines else '')
    outdir = paths.workspace().model_comparison / f"{args.split}{suffix}__{'__'.join(run_names)}"

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

    logger.info("Comparing models:\n" + "\n".join(
        f"\t{label}: {root} ({args.split}, {args.action_decoding})" for label, root in zip(labels, roots)))

    # ------------------------------
    # Build evaluators and check they are comparable
    # ------------------------------
    logger.info("Building evaluators...")
    evaluators = [
        build_evaluators(
            cfg,
            device=device,
            eval_outdir=rp.eval_dir(args.split, args.action_decoding).name,
            save_movie=args.save_movies,
            save_mollweide=args.save_mollweides,
            plot_bins=args.plot_bins,
            action_decoding=args.action_decoding,
            split=args.split,
        )
        for cfg, rp in zip(cfg_list, run_paths)
    ]
    ms_list = [ms for _, ms in evaluators]
    sm.check_comparable(ms_list, labels)

    # ------------------------------
    # Run evaluators
    # ------------------------------
    logger.info("Running evaluators...")
    for ss, ms in evaluators:
        ss.run()
        ms.run(overwrite=args.force_overwrite)

    # ------------------------------
    # Depth vs. uniformity against DES
    # ------------------------------
    rollouts = {label: sm.policy_rollout(ms) for label, ms in zip(labels, ms_list)}
    legacy = [label for label, r in rollouts.items() if r is None]
    if legacy:
        raise ValueError(f"No manifest-listed night CSVs for {legacy}; rerun them with -f.")
    lookups = ms_list[0].data.lookups
    record = sm.load_des_record(sm.resolve_fits_path(cfg_list[0]), sm.evaluated_nights(ms_list[0]))
    scored, report = sm.score_schedulers(rollouts, record, lookups, heuristics=args.baselines)
    report.log()
    table = sm.save_depth_uniformity(scored, report, lookups, {label: label for label in labels}, outdir)
    for label, root in zip(labels, roots):
        table.loc[table['key'] == label, 'run_dir'] = str(root)
    table['split'] = args.split
    table['action_decoding'] = args.action_decoding
    table.to_csv(outdir / 'depth_uniformity.csv', index=False)
    logger.info(f"Wrote {outdir / 'depth_uniformity.png'}")


if __name__ == '__main__':
    main()
