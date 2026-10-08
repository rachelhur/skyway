import numpy as np


from skyway.configs.constants import *
import logging

logger = logging.getLogger(__name__)

import pandas as pd


def read_schedule_csv(path) -> pd.DataFrame:
    """Read a schedule CSV (columns timestamp, field_id, filter_idx, filter, bin_id, reward).

    Strips the `agent_` column prefix written by older runs.

    Parameters
    ----------
    path : str or Path
        Schedule CSV path.

    Returns
    -------
    pd.DataFrame
        One row per scheduled exposure.
    """
    return pd.read_csv(path).rename(columns=lambda c: c.removeprefix('agent_'))


# -------------------------------------------------------------- #
# -------------------- FITS <-> PD.DATAFRAME -------------------- #
# ------------------------------------------------------------ #

import fitsio
import pandas as pd

from astropy.time import Time
import pandas as pd


def fits_to_df(fits_path):
    d = fitsio.read(fits_path)
    df = pd.DataFrame(d.astype(d.dtype.newbyteorder('='))) # Big-endian/little-endian error
    return df

def _replace_with_pd_dt(df):
    df['datetime'] = pd.to_datetime(
        df['datetime'],
        format='%Y-%m-%d %H:%M:%S',
        utc=True,
        errors='coerce'
    )
    return df

def _drop_nan_dts(df):
    df = df.dropna(subset=['datetime'])
    return df

def _add_timestamp(df):
    t_array = Time(df['datetime'].dt.tz_localize(None).values, scale='utc')
    # .assign() creates and returns a new df with the added column
    return df.assign(timestamp=t_array.unix.astype(np.int64))
def _add_night(df):
    return df.assign(night=(df['datetime'] - pd.Timedelta(hours=12)).dt.date)
