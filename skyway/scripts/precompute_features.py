"""Precompute all raw features from a FITS file and save to disk.

Run this once before training to populate the feature cache that
``run-train`` and ``run-validate`` load from.

Usage::

    precompute-features \\
        --fits_path  <path/to/fits>       \\
        --lookups_dir <path/to/lookups>   \\
        --outdir     <cache_dir>          \\
        --nside      16                   \\
        --action_space_type [radec|azel]
"""
import argparse
import logging
from pathlib import Path

from skyway.configs.paths import feature_cache_dir, field_feature_cache_dir, lookups_dir, resolve_data_dir, workspace
from skyway.data.feature_cache import FieldFeatureCache, BinFeatureCache
from skyway.data.lookup_tables import TrainLookupTables
from skyway.data.preprocessing import find_interruptions, load_and_process_historic_data, preprocess_fits
from skyway.ephemerides import ephemerides
from skyway.io.logger_utils import configure_logger
from skyway.configs.constants import _FIELD_FEATURES
from skyway.configs.experiment_schema import ActionConstraints
from skyway.data.features.field_features import label_mask_report

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Precompute all raw features from a FITS file and save to disk.",
    )
    parser.add_argument(
        '--fits_path', type=str, default=str(workspace().des_fits),
        help='Path to the FITS observations file.'
    )
    parser.add_argument(
        '--data_dir', type=str, default=workspace().des_data,
        help='Data directory containing lookups and output dir for feature cache (relative paths are under the workspace root).'
    )
    parser.add_argument(
        '--nside', type=int, default=16,
        help='HEALPix nside parameter.'
    )
    parser.add_argument(
        '--action_space_type', type=str, default='radec',
        choices=['radec', 'azel'],
        help='Coordinate frame for the HEALPix grid.'
    )
    parser.add_argument(
        '-l', '--logging_level', type=str, default='info',
        help='Logging level (info or debug).'
    )

    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced data.')
    parser.add_argument('--field_features', action='store_true',
                        help='Build the standalone field feature cache (field_filter runs) from the FITS archive.')
    parser.add_argument('--interruptions_only', action='store_true',
                        help='Only write the interruptions file into an existing cache directory.')
    return parser.parse_args()


def compute_field_cache(outdir: Path, df, lookups, interruptions) -> None:
    """Build the standalone field feature cache and report expert labels outside the field mask.

    Parameters
    ----------
    outdir : Path
        Field feature cache directory.
    df : pd.DataFrame
        Processed survey exposures.
    lookups : LookupTables
        Survey lookups.
    interruptions : pd.DataFrame
        Interrupted survey exposures.
    """
    logger.info(f"Computing the field feature cache into {outdir}")
    FieldFeatureCache.compute(outdir, df, lookups, _FIELD_FEATURES, interruptions=interruptions)
    cache = FieldFeatureCache.load(outdir, mmap=True)
    constraints = ActionConstraints()
    report = label_mask_report(
        cache.global_df, cache.current_state_idxs, cache.next_state_idxs, cache.field_features,
        cache.field_feature_names, lookups.fields['dec'].to_numpy(), lookups.survey.telescope,
        min(constraints.airmass_limit, constraints.airmass_failsafe), survey=lookups.survey,
    )
    logger.info(f"Expert labels outside their own field-level mask: {report}")


def main():
    args = get_args()
    configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=False,
        use_tqdm=True,
    )

    fits_path = Path(args.fits_path)
    is_azel = 'azel' in args.action_space_type
    data_dir = resolve_data_dir(args.data_dir)
    lookup_dir = lookups_dir(data_dir)
    outdir = feature_cache_dir(data_dir, args.nside, is_azel)

    logger.info(f"Loading the full exposure archive from {fits_path}")
    archive_df = preprocess_fits(fits_path)
    df = load_and_process_historic_data(df=archive_df.copy())
    interruptions = find_interruptions(df, archive_df)
    logger.info(f"Found {len(interruptions)} survey exposures preceded by other archived exposures.")

    if args.field_features:
        field_dir = field_feature_cache_dir(data_dir)
        compute_field_cache(field_dir, df, TrainLookupTables.load_from_dir(lookup_dir), interruptions)
        return

    if args.interruptions_only:
        if not BinFeatureCache.exists(outdir):
            raise FileNotFoundError(f"No feature cache at {outdir}; run the full precompute first.")
        BinFeatureCache.save_interruptions(outdir, interruptions)
        logger.info(f"Wrote interruptions to {outdir}")
        return

    if args.test:
        logger.info("Running in test mode: using only the first 1000 rows of data.")
        df = df.head(1000)

    logger.info(f"Loading lookup tables from {lookup_dir}")
    lookups = TrainLookupTables.load_from_dir(lookup_dir)

    logger.info(f"Building HEALPix grid  nside={args.nside}  is_azel={is_azel}")
    hpGrid = ephemerides.HealpixGrid(nside=args.nside, is_azel=is_azel)

    logger.info("Computing feature cache…")
    cache = BinFeatureCache.compute(df=df, lookups=lookups, hpGrid=hpGrid)
    cache.interruptions = interruptions


    if not args.test:
        logger.info(f"Saving feature cache to {outdir}")
        cache.save(outdir)
    logger.info("Done.")


if __name__ == '__main__':
    main()
