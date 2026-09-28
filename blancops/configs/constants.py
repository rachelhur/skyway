from collections import OrderedDict
from enum import IntEnum
from typing import Dict, List, Literal

import numpy as np

"""
Feature names
"""

_CYCLICAL_FEATURE_NAMES = ["ra", "az", "ha", "lst"]
_FILTER_DEP_FEATURE_NAMES = [
     # global features
    'global_mean_tiling',
    'urgency', 'sky_brightness', 'is_filter', 'survey_progress',
    # bin features
    'min_tiling', 'num_unvisited_fields', 'num_incomplete_fields', 't_since_last_visit',
    'rel_min_tiling', 'rel_num_unvisited_fields', 'rel_num_incomplete_fields', 'rel_t_since_last_visit',
    # field features
    'completion', 'rel_completion',
    ]


_DEFAULT_BC_AZEL_GLOB_FEATURES = [
    't_night', 'moon_phase', 'moon_distance',
    'airmass', 'ha', 'lst', 'el', 'az',
    'sun_ra', 'sun_dec', 'sun_az', 'sun_el',
    'moon_ra', 'moon_dec', 'moon_az', 'moon_el',
    'is_filter', 'sky_brightness', 'global_mean_tiling',
    'fwhm'
]
_DEFAULT_BC_AZEL_BIN_FEATURES = [
    'moon_distance', 'airmass', 'el',
    'delta_az', 'delta_el',
    'pointing_distance',
    'rel_ha', 'rel_moon_distance', 'rel_t_since_last_visit',
    'rel_min_tiling', 'rel_num_unvisited_fields', 'rel_num_incomplete_fields',
    't_until_set'
]

_GLOBAL_FEATURES = [
    "t_night",	# use, but maybe refactor?
    "t_survey",  # DO NOT USE loss of generality
    "lst",  # azel: always use
    "ha",   # azel: always use
    "el",   # azel: always use
    "ra",   # azel: don't use - footprint-specific memorization
    "az",   # azel: don't use - redundant with ha, el
    "dec",  # azel: don't use - redundant with el, airmass, ha, and also footprint-specific memorization
    "airmass", # always use
    "sun_ra", "sun_dec",    # always use
    "sun_az", "sun_el",    # always use
    "moon_ra", "moon_dec",   # always use
    "moon_az", "moon_el",       # always use
    "moon_distance",  # always use
    "sun_distance",  # always use
    # "num_unvisited_fields",
    # "num_incomplete_fields",
    # "min_tiling",
    "filter_wave", "filter_idx", "is_filter", # is_filter one-hot encoded is probably best
    # "urgency",    #  loss of generality
    # "survey_progress",   # mostly parallel to mean_tiling
    "global_mean_tiling",   # always use
    "sky_brightness",   # always use
    "moon_phase",     # always use
    "fwhm",
]


_BIN_FEATURES = [
    "ha",
    "airmass",   # always use
    "moon_distance",   # always use
    "sun_distance",   # always use
    "rel_ha", "rel_moon_distance",
    "delta_az",    # always use
    "delta_el",    # always use
    "az",
    "el",
    "ra",		# don't use - specific to DES footprint and could cause memorization
    "dec",      # test
    "pointing_distance",       # test
    "num_unvisited_fields",      # test
    "num_incomplete_fields",      # test
    "min_tiling",       # test
    "rel_num_unvisited_fields", # The number of univisited fields in this bin divided by total number of fields in this bin
    "rel_num_incomplete_fields",  # The number of incomplete fields in this bin divided by total number of fields in this bin
    "rel_min_tiling",	# The minimum tiling amongst all fields in this bin, divided by that field's target tiling.
    "rel_t_since_last_visit", # The last time since this bin has been visited in the survey minus the mean "last time since last visit" across all bins at this timestamp
    "t_until_set", # always use
    "t_since_last_visit" # use rel_t_since_last_visit instead
                            # the z-score norm bakes in an assumed survey cadence
                            # resulting in loss of generality for future surveys
                            # rel_t_since_last_visit still suffers from a different spread,
                            # but much better off
]


# Per-field features for the field_filter action space (base names; completion and
# t_since_last_visit expand per filter). Positional features are evaluated at field centers.
_FIELD_FEATURES = [
    "el", "airmass", "ha", "moon_distance", "sun_distance", "pointing_distance",
    "delta_az", "delta_el", "t_until_set", "rel_ha", "rel_moon_distance",
    "completion", "rel_completion", "t_since_last_visit",
]


_NORM_TYPES = Literal[
    'cyclical', # cos/sin - this normalization is run before all others
    'sin', # sin only
    'log',
    'fractional', # for values bound between 0 and 1 -- performs 2*(val - .5)
    'z_score', # standard z-score normalization
    'local_mean_z', # divides by global std (local mean already in rel_ features)
    'local_z_score', # divides by per-timestamp std across bins (local mean already in rel_ features)
    None
    ]

_ALLOWED_NORMS_PER_FEATURE = {
    # Telescope coords
    'ra': {'cyclical'},
    'az': {'cyclical'},
    'ha': {'cyclical'},
    'lst': {'cyclical'},

    # Sun coords
    'sun_ra': {'cyclical'},
    'sun_az': {'cyclical'},
    'sun_el': {'z_score'},
    'sun_dec': {'z_score'},

    # Moon coords
    'moon_ra': {'cyclical'},
    'moon_az': {'cyclical'},
    'moon_el': {'z_score'},
    'moon_dec': {'z_score'},

    'moon_distance': {'z_score'},
    'sun_distance': {'z_score'},
    'airmass': {'log', 'z_score'},
    'pointing_distance': {'z_score'},
    'sky_brightness': {'z_score'},
    'delta_az': {'z_score'},
    'delta_el': {'z_score'},
    'el': {'z_score'},
    'dec': {'z_score'},

    'fwhm': {'log', 'z_score'},
    'urgency': {'log', 'z_score'},
    'survey_progress': {'fractional', 'sin', 'z_score'},

    'pointing_distance': ['z_score'],
    'num_unvisited_fields': ['z_score'],
    'num_incomplete_fields': ['z_score'],
    'min_tiling': ['z_score'],

    'rel_num_unvisited_fields': {'local_mean_z', 'local_z_score'},
    'rel_num_incomplete_fields': {'local_mean_z', 'local_z_score'},
    'rel_min_tiling': {'local_mean_z', 'local_z_score'},
    'rel_t_since_last_visit': {'local_mean_z', 'log', 'local_z_score'},
    'rel_moon_distance': {'local_mean_z', 'local_z_score'},
    'rel_ha': {'local_mean_z', 'local_z_score'},

    't_night': {'fractional'},
    't_survey': {'fractional'},
    'moon_phase': {'fractional'},
    'survey_num_visits_done': {'fractional'},
    't_until_set': {'fractional'},
    't_since_last_visit': {'fractional', 'z_score', 'log', None},
    'global_mean_tiling': {'fractional'},
}

_DEFAULT_NORM_MAPPING = {
    # Telescope coords
    'ra': ['cyclical'],
    'az': ['cyclical'],
    'ha':['cyclical'],
    'lst': ['cyclical'],
    'el': ['z_score'],
    'dec': ['z_score'],

    # Sun coords
    'sun_ra': ['cyclical'],
    'sun_az': ['cyclical'],
    'sun_el': ['z_score'],
    'sun_dec': ['z_score'],

    # Moon coords
    'moon_ra': ['cyclical'],
    'moon_az': ['cyclical'],
    'moon_el': ['z_score'],
    'moon_dec': ['z_score'],

    # Image quality
    'moon_distance': ['z_score'],
    'sun_distance': ['z_score'],
    'airmass': ['log', 'z_score'],
    'sky_brightness': ['z_score'],
    'delta_az': ['z_score'],
    'delta_el': ['z_score'],
    'fwhm': ['log', 'z_score'],
    'urgency': ['z_score'],

    # Bin features
    'pointing_distance': ['z_score'],
    'num_unvisited_fields': ['z_score'],
    'num_incomplete_fields': ['z_score'],
    'min_tiling': ['z_score'],

    'rel_num_unvisited_fields': ['local_mean_z'],
    'rel_num_incomplete_fields': ['local_mean_z'],
    'rel_min_tiling': ['local_mean_z'],
    'rel_moon_distance': ['local_mean_z'],
    'rel_ha': ['local_mean_z'],
    'rel_t_since_last_visit': ['local_mean_z'],

    't_night': ['fractional'],
    't_survey': ['fractional'],
    'moon_phase': ['fractional'],
    'survey_num_visits_done': ['fractional'],
    't_until_set': ['fractional'],
    't_since_last_visit': ['z_score'],
    'global_mean_tiling': ['fractional'],
}

"""

ENVIRONMENT SENTINEL VALS

"""
WAIT_SIGNAL = -2
NO_FILTER_SIGNAL = -1 # if action space is just bins, not filters
AZEL_BIN_FEAT_SENTINEL = -1.0 # no fields now, might be later
RADEC_BIN_FEAT_SENTINEL = -1.0 # no fields ever


class EnvSignal(IntEnum):
    WAIT = -2
    NO_FILTER = -1

"""
BLANCO CONSTS
"""

# BLANCO_LAT = -30.169
BLANCO_LON = "-70:48:23.49"
BLANCO_ELEV = 2200

"""

ZENITH CONSTANTS

"""

ZENITH_AZ = 0
ZENITH_EL = np.pi/2
ZENITH_AIRMASS = 1
ZENITH_ZD = 0
ZENITH_HA = 0
ZENITH_OBJECT = 'zenith'
ZENITH_FIELD_ID = -1
ZENITH_BIN_NUM = -1
ZENITH_WAVELENGTH = 0
ZENITH_FILTER_IDX = -1
ZENITH_FILTER = 'null'

"""

FILTER INFO

"""

# Filter wavelengths (nm) according to obztak https://github.com/kadrlica/obztak/blob/c28fab23b09bcff1cf46746eae4ec7e40aeb7f7a/obztak/seeing.py#L22
FILTER2WAVE = {
    # 'u': 380, # not present in train data,
    'g': 480,
    'r': 640,
    'i': 780,
    'z': 920,
    'Y': 990
}

_NUM_FILTERS = len(FILTER2WAVE)
IDX2WAVE = {i: FILTER2WAVE[k] for i, k in enumerate(FILTER2WAVE.keys())}
FILTERWAVENORM = 1000.

FILTER2IDX = {k: i for i, k in enumerate(FILTER2WAVE.keys())}
IDX2FILTER = {v: k for k, v in FILTER2IDX.items()}

# Reference band for seeing (FWHM) projection. Used when a pointing carries no
# filter (zenith / WAIT) so the wavelength term drops out, and as the default
# band for seeding forward-sim seeing in OfflineBlancoEnv. r-band by convention
# (matches the obztak seeing reference above).
FWHM_REF_FILTER = 'r'
FWHM_REF_WAVELENGTH = FILTER2WAVE[FWHM_REF_FILTER]


# SIN_NORM_FEATURE_NAMES = []
# LOG_NORM_FEATURE_NAMES = ["fwhm", "urgency"]
# FRACTIONAL_FEATURE_NAMES = ["moon_phase", "t_night", "t_survey", "survey_num_visits_done"]
# Z_SCORE_NORM_FEATURE_NAMES = ["airmass", "pointing_distance", "moon_distance", "sky_brightness", "fwhm", "delta_az", "delta_el", "urgency", "el", "dec"]
# LOCAL_MEAN_Z_SCORE_FEATURE_NAMES = ["rel_num_unvisited_fields", "rel_num_incomplete_fields", "rel_min_tiling", "rel_moon_distance", "rel_ha"]

"""

TEST SUITE CONSTS

"""

TEST_SUITE_NAMES = ['magic-spring', 'magic-spring-1', 'healpix-grid', 'delve', 'gw-followup']

MS_OBSERVING_DATES = ['2026-04-09-half1', '2026-04-10-half1', '2026-04-11-half1', '2026-04-12-half1', \
                      '2026-05-09-half1', '2026-05-10-half1', '2026-06-08-half1', '2026-06-09-half1']
HP_OBSERVING_DATES = ['2026-04-09-full', '2026-04-10-full', '2026-04-11-full', '2026-04-12-full', \
                      '2026-05-09-full', '2026-05-10-full', '2026-06-08-full', '2026-06-09-full']
DD_NIGHT = ['2026-04-10-half2']
GW_OBSERVING_DATES_GOOD = ['2026-02-10-full', '2026-02-11-full', '2026-02-12-full', '2026-02-13-full', \
                           '2026-02-14-full', '2026-02-15-full', '2026-02-16-full', '2026-02-17-full']
GW_OBSERVING_DATES_BAD = ['2026-05-09-full', '2026-05-10-full', '2026-05-11-full', '2026-05-12-full', \
                          '2026-05-13-full', '2026-05-14-full', '2026-05-15-full', '2026-05-16-full']
# DELVE_OBSERVING_DATES = ['2026-04-09-half1', '2026-04-10-half1', '2026-04-11-half1', '2026-04-12-half1', '2026-05-09-half1', '2026-05-10-half1', '2026-06-08-half1', '2026-06-09-half1']

"""

DEPLOYMENT CONSTS

"""

DEPLOYMENT_OBSERVING_DATES = ['2026-06-23-half1', '2026-06-24-half1']


"""

MODEL CONSTANTS

"""

import numpy as np

# Focal loss weights
FILTER_COUNTS_ORDERED = np.array([20574, 18450, 17312, 15984, 16221]) # entire train dataset
FILTER_ALPHA_WEIGHTS = 1 / FILTER_COUNTS_ORDERED * len(FILTER_COUNTS_ORDERED) / np.sum(1/FILTER_COUNTS_ORDERED)
