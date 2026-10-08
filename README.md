# Skyway

This repo contains a reinforcement-learning based autonomous scheduling agent for the Blanco 4-m telescope. It operates in three primary modes: **offline training/evaluation** for developing scheduling policies, the **Offline Scheduler** for pre-generating schedules for future nights, and the **Live Scheduler** for real-time, human-in-the-loop autonomous observation scheduling with live telescope telemetry.

## Setup

```bash
conda env create -f environment.yml
conda activate skyway
pip install -e .
```

For instructions on training and evaluating policies, see the [training and evaluating documentation](./skyway/rl/README.md).

For instructions on creating offline schedules for future nights, see [below](#creating-offline-pre-generated-schedules-for-the-blanco-4-m-telescope).

For instructions on deploying the real-time observation scheduling agent, see the [live scheduler documentation](./skyway/live_scheduler/README.md).

# Creating offline (pre-generated) schedules for the Blanco 4-m Telescope

`run-offline-scheduler` simulates one or more future nights with a trained scheduling policy and writes the schedule
as observing script(s) that can be passed to Blanco's telescope control system, SISPI.

## Feeding a list of targets

Your set of targets can be passed as a CSV or JSON list of records, with one row per (field, filter). Column names are case insensitive.

| Column | Required | Meaning |
|---|---|---|
| `ra`, `dec` | yes | pointing **degrees** by default (pass `--radians` if the file is in radians). Negative RA is fine. |
| `filter` | yes | One of `g`, `r`, `i`, `z`, `Y`. |
| `count` | yes | Number of exposures wanted in this filter (> 0). The scheduler does not assume these (or any other fields) need to happen sequentially.|
| `exptime` | yes | Exposure time in seconds (> 0). |
| `field_name` | no | A unique identifier for a target pointing; a target observed in several filters must use the same `field_name` on each of its rows. Two different positions may not share a name within a `propid`. Gets passed into the `object` column in SISPI. If none passed, a default `field_<n>` is assigned|
| `priority` | no | Not used by the offline scheduler (only by the live scheduler). |
| `propid` | no | Only scopes the `field_name` check above. The `propid` written to the SISPI file comes from `--propid`. |

For details on preparing observing scripts, see NOIRLab's documentation [here](https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/User-Guide/Preparing-Observing-Scripts)

Here is an example file:

```csv
ra,dec,filter,count,exptime,field_name
150.1,-30.2,g,2,90,TargetA
150.1,-30.2,r,1,90,TargetA
-10.0,-45.0,i,1,120,TargetB
```

The run stops with an error message for a missing column, an unknown filter, a Dec outside [-90, 90], a `count` or
`exptime` that is not a positive number, a (field, filter) pair listed twice, or a name clash as described above. It
warns if every coordinate looks like radians while degrees were expected.

## Running the offline scheduler

### Scheduling for a full/half night

The offline scheduler can be run with the command:

```bash
run-offline-scheduler \
    --fields my_fields.csv \
    -d 2026-11-03-full \
    -o runs/2026-11-03 \
    -s \
    --propid <your proposal id> --proposer <name> --program <program>
```

### Scheduling an exact time window

Instead of `-d`, give `--start_time` and/or `--stop_time` as a date and time in UTC. For example, 01:30 to 06:00 UTC on
2026-11-04 (the night of 2026-11-03 at CTIO):

```bash
run-offline-scheduler \
    --fields my_fields.csv \
    --start_time 2026-11-04T01:30 --stop_time 2026-11-04T06:00 \
    -o runs/2026-11-03-window \
    -s \
    --propid <your proposal id> \
    --proposer <name> --program <program> \
    --save_movie
```

Accepted formats (every example below means the same moment):

| Format | Example |
|---|---|
| UTC date and time (ISO 8601) | `2026-11-04T01:30`, `2026-11-04T01:30:00Z` |
| Local time with its UTC offset | `2026-11-03T22:30-03:00`, `2026-11-03T22:30-03` |
| Unix seconds | `1793755800` |

- Without `--stop_time` the window ends at that night's sunrise; without `--start_time` it starts at that night's sunset.
- The window must lie within one night, while the Sun is below `--sun_el_limit`; otherwise the run stops with an error
  that shows that night's sunset and sunrise.
- It cannot be combined with `-d`. Output files are named `<evening date>-window`, e.g. `nights/2026-11-03-window.csv`.


### Models

Three trained models ship with the package; choose one with `-m`:

| `-m` | Model | Approximate behavior |
|---|---| --- |
| `cql_field` (default) | Field-level conservative Q-learning (CQL) | Visits every field once, then revisits about hourly, taking several filters per visit. |
| `bc_field` | Field-level behavior cloning | -- |
| `bc_v1_nside32` | Behavior cloning that picks HEALPix sky bins (nside 32), then the best field inside the chosen bin. | -- |

`-m` also accepts the path to a trained run directory (one holding `configs/resolved_config.yaml` and
`checkpoints/model.pt`).

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `-d/--observing_nights` | none | One or more nights, `YYYY-MM-DD-full`, `-half1` or `-half2` (the date is the evening's date). |
| `--start_time`, `--stop_time` | none | Instead of `-d`: one exact window within a single night, as a UTC date and time such as `2026-11-04T01:30`; see [Scheduling an exact time window](#scheduling-an-exact-time-window). |
| `--sun_el_limit` | -12 | Highest Sun elevation (degrees) for observing. Determines each night's start and end times; `--start_time`/`--stop_time` must fall inside that night. |
| `--airmass_limit` | 1.8 | Only fields with airmass below this are scheduled (1 < limit <= 3). |
| `--initial_fwhm` | 0.9 | Assumed zenith seeing (arcsec, r band) for the simulation, scaled per pointing by airmass and filter. |
| `-s/--save_observing_script` | off | Write SISPI files; requires `--propid` and `--program`. |
| `--save_movie` | off | Per-night movie in `plots/`. |
| `--overwrite` | off | Replace the results of an earlier run in `-o` (see below). |
| `-l/--logging_level` | info | `debug` also logs every run argument. |


If `-o` already holds results from an earlier run, the scheduler refuses to start. Pass `--overwrite` to replace them
(this deletes the earlier `nights/`, `observing_scripts/`, `plots/` and `rollout_info.pkl`; `lookups/` is kept), or
choose another `-o`.

To schedule another window with the same targets, repeat the command with a new `-o`, or reuse the tables
the first run built with `--field_lookup_dir runs/2026-11-03/lookups` instead of `--fields`.

## Output

```text
runs/2026-11-03/
  lookups/                                       tables built from your fields file
  nights/2026-11-03-full.csv                     the schedule (timestamp, field_id, filter_idx, bin_id, reward, filter)
  observing_scripts/2026-11-03-full_sispi.json   SISPI script for that night
  observing_scripts/all_nights_sispi.json        all nights in one script
  plots/2026-11-03-full_movie.gif                with --save_movie
  rollout_info.pkl                               per-night diagnostics
  offline_scheduler.log
```

The log prints how many exposures each night got. A night where none of your fields is observable (below the
airmass limit while the Sun is down) gets a warning and no files.

Before handing a script to the observers, check each night's exposure count in the log, and that `propid` and
`program` in the SISPI file are the real values.
