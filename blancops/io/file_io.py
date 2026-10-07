from collections import OrderedDict

import numpy as np

from blancops.configs.constants import WAIT_SIGNAL, ZENITH_BIN_NUM
from blancops.math import units
from collections import defaultdict
from pathlib import Path

from blancops.configs.constants import *
import logging

logger = logging.getLogger(__name__)

SCHEDULE_KEYS = {
    "timestamp": 'agent_timestamp',
    'field_id': 'agent_field_id',
    'bin_id': 'agent_bin_id',
    'filter': 'agent_filter',
    'filter_idx': 'agent_filter_idx'
}
import pandas as pd


import json
_EMPTY_SISPI_DICT = OrderedDict([
    ("object",  None),
    ("seqnum",  None), # 1-indexed
    ("seqtot",  1),
    ("seqid",   ""),
    ("expTime", 90),
    ("RA",      None),
    ("dec",     None),
    ("filter",  None),
    ("count",   1),
    ("expType", "object"),
    ("program", None),
    ("wait",    "False"),
    ("propid",  None),
    ("comment", ""),
])


def write_SISPI_from_df(schedule_df, out_fn, save_dir, lookups, filter_override_val='CaHK', proposer='Cerny', program='magic-spring', dt_series=None,
                              propid='test', use_date_prefix=True):
    obs_night_str = dt_series.dt.strftime('%Y-%m-%d').values[0]
    outpath = save_dir / (f"{obs_night_str}_{out_fn}" if use_date_prefix else out_fn)

    ordered_field_ids = schedule_df[SCHEDULE_KEYS['field_id']].values

    input_dict = {}
    input_dict['object'] = [str(lookups.fields['field'].values[fid]) for fid in ordered_field_ids]
    input_dict['RA'] = [round(float(lookups.fields['ra'].values[fid] / units.deg), 5) for fid in ordered_field_ids]
    input_dict['dec'] = [round(float(lookups.fields['dec'].values[fid] / units.deg), 5) for fid in ordered_field_ids]
    if filter_override_val is not None:
        input_dict['filter'] = filter_override_val
    else:
        filter_col = 'filter' if 'filter' in schedule_df.columns else SCHEDULE_KEYS['filter']
        input_dict['filter'] = schedule_df[filter_col].to_list()
    filter_val = input_dict['filter']
    if isinstance(filter_val, list):
        filter_indices = np.array([lookups.survey.filter2idx[f] for f in filter_val])
    else:
        filter_indices = np.full(len(ordered_field_ids), lookups.survey.filter2idx[filter_val])
    input_dict['expTime'] = lookups.fidfilt_exptime[ordered_field_ids, filter_indices].astype(int).tolist()
    input_dict['program'] = program
    input_dict['proposer'] = proposer
    input_dict['count'] = 1
    input_dict['expType'] = "object"
    input_dict['wait'] = "False"
    input_dict['propid'] = propid
    input_dict['comment'] = "" #[f"datetime: {dt}" for dt in dt_series]
    input_dict['seqid'] = [f"datetime: {dt}" for dt in dt_series]

    seqtot = len(ordered_field_ids)

    sispi_list = []

    for i in range(seqtot):
        obs = _EMPTY_SISPI_DICT.copy()

        current_obs = {}

        # Iterate through your config and check the type of each value
        for key, value in input_dict.items():
            if isinstance(value, (list, tuple)):
                # If it is a list or tuple, grab the i-th element
                current_obs[key] = value[i]
            else:
                # If it is a constant (string, int, float), use it as-is
                current_obs[key] = value

        # Add the sequence numbers explicitly
        current_obs["seqnum"] = 1 #i + 1
        current_obs["seqtot"] = 1

        # Update and append
        obs.update(current_obs)
        sispi_list.append(obs)

    with open(outpath, 'w') as f:
        json.dump(sispi_list, f, indent=4)

def save_survey_schedule(eval_metrics, save_dir, field_lookup, multinight_movie=True, ep_num=0, save_SISPI=False,
                         SISPI_fn="survey_schedule.json"):
    eval_metrics = eval_metrics[f'ep-{ep_num}']
    if multinight_movie:
        schedule_path = Path(save_dir) / "full_survey_schedule.csv"
        collected_metrics = defaultdict(list)
        schedule_keys = ['bin', 'field_id', 'filter_idx', 'timestamp']

        # Extract the arrays from each night
        for night_key, metrics_dict in eval_metrics.items():
            for metric_name, array_values in metrics_dict.items():
                if metric_name in schedule_keys:
                    collected_metrics[metric_name].append(array_values)

        # Concatenate the collected arrays for each metric
        full_schedule = {}
        for k, list_of_arrays in collected_metrics.items():
            # np.concatenate joins the arrays end-to-end
            if k == 'bin':
                key = SCHEDULE_KEYS['bin_id']
            elif k == 'field_id':
                key = SCHEDULE_KEYS['field_id']
            elif k == 'filter_idx':
                key = SCHEDULE_KEYS['filter']
            elif k == 'timestamp':
                key = SCHEDULE_KEYS['timestamp']

            full_schedule[key] = np.concatenate(list_of_arrays)

        # Filter out zenith and wait states
        sel_valid_obs = full_schedule[SCHEDULE_KEYS['bin_id']] != ZENITH_BIN_NUM
        sel_valid_obs &= full_schedule[SCHEDULE_KEYS['bin_id']] != WAIT_SIGNAL
        for k, v in full_schedule.items():
            full_schedule[k] = v[sel_valid_obs]

        # Save schedule
        df = pd.DataFrame(data={k: pd.Series(v) for k, v in full_schedule.items()})
        df[SCHEDULE_KEYS['filter']] = df[SCHEDULE_KEYS['filter']].map(field_lookup.survey.idx2filter)
        df.to_csv(schedule_path, index=False)

    if save_SISPI:
        for night_key, night_dict in eval_metrics.items():
            if 'night' not in night_key:
                continue
            collected_metrics = defaultdict(list)
            schedule_keys = ['bin', 'field_id', 'filter_idx', 'timestamp']
            # Extract the arrays from each night
            for metric_name, array_values in night_dict.items():
                if metric_name in schedule_keys:
                    collected_metrics[metric_name].append(array_values)

            valid_mask = (night_dict['bin'] != -1) & (night_dict['bin'] != -2)
            # Concatenate the collected arrays for each metric
            schedule = {}
            for k, list_of_arrays in collected_metrics.items():
                # np.concatenate joins the arrays end-to-end
                if k == 'bin':
                    key = SCHEDULE_KEYS['bin_id']
                elif k == 'field_id':
                    key = SCHEDULE_KEYS['field_id']
                elif k == 'filter_idx':
                    key = SCHEDULE_KEYS['filter']
                elif k == 'timestamp':
                    key = SCHEDULE_KEYS['timestamp']

                schedule[key] = np.concatenate(list_of_arrays)

            # Filter out zenith and wait states
            sel_valid_obs = schedule[SCHEDULE_KEYS['bin_id']] != ZENITH_BIN_NUM
            sel_valid_obs &= schedule[SCHEDULE_KEYS['bin_id']] != WAIT_SIGNAL
            assert all(sel_valid_obs == valid_mask)
            for k, v in schedule.items():
                schedule[k] = v[sel_valid_obs]

            # Save schedule
            dt_series = pd.Series(pd.to_datetime(schedule[SCHEDULE_KEYS['timestamp']], utc=True, unit='s') - pd.Timedelta(12, "h"))
            if len(dt_series) < 1:
                continue
            obs_night_str = dt_series.dt.strftime('%Y-%m-%d').values[0]

            schedule_path = Path(save_dir) / f"survey_schedule_{obs_night_str}.csv"
            df = pd.DataFrame(data={k: pd.Series(v) for k, v in schedule.items()})
            df[SCHEDULE_KEYS['filter']] = df[SCHEDULE_KEYS['filter']].map(field_lookup.survey.idx2filter)
            df.to_csv(schedule_path, index=False)
            write_SISPI_from_df(df, SISPI_fn, save_dir, lookups=field_lookup, dt_series=dt_series)
    return full_schedule

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
