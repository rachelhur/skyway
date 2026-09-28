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

from blancops.configs.paths import feature_cache_dir, lookups_dir, workspace
from blancops.data.feature_cache import RawFeatureCache
from blancops.data.lookup_tables import TrainLookupTables
from blancops.data.preprocessing import load_and_process_historic_data
from blancops.ephemerides import ephemerides
from blancops.io.logger_utils import configure_logger

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
        help='Data directory containing lookups and output dir for feature cache.'
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
                        help='Compute field-level features into an existing cache directory (field_filter runs).')
    parser.add_argument('--interruptions_only', action='store_true',
                        help='Only write the interruptions file into an existing cache directory.')
    return parser.parse_args()


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
    lookup_dir = lookups_dir(args.data_dir)
    outdir = feature_cache_dir(args.data_dir, args.nside, is_azel)

    logger.info(f"Loading and processing historical data from {fits_path}")
    df = load_and_process_historic_data(fits_path=fits_path)

    if args.test:
        logger.info("Running in test mode: using only the first 1000 rows of data.")
        df = df.head(1000)

    logger.info(f"Loading lookup tables from {lookup_dir}")
    lookups = TrainLookupTables.load_from_dir(lookup_dir)

    logger.info(f"Building HEALPix grid  nside={args.nside}  is_azel={is_azel}")
    hpGrid = ephemerides.HealpixGrid(nside=args.nside, is_azel=is_azel)

    logger.info("Computing feature cache…")
    cache = RawFeatureCache.compute(df=df, lookups=lookups, hpGrid=hpGrid)


    if not args.test:
        logger.info(f"Saving feature cache to {outdir}")
        cache.save(outdir)
    logger.info("Done.")


if __name__ == '__main__':
    main()
