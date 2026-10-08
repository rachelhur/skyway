"""Live scheduling visualization utilities."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from skyway.plotting import plotting
from skyway.ephemerides import ephemerides, time_utils
from skyway.math import units

DEFAULT_FILTER_COLORS = {
    "u": "darkviolet",
    "g": "royalblue",
    "r": "forestgreen",
    "i": "lawngreen",
    "z": "gold",
    "Y": "red",
    "unknown": "darkgray",
}


def _is_empty(df):
    if df is None:
        return True
    elif isinstance(df, pd.DataFrame):
        return df.empty
    elif isinstance(df, (pd.Series, dict)):
        return len(df) == 0
    else:
        raise ValueError(f"Unsupported type for emptiness check: {type(df)}")


def _filter_colors(df, color_map=DEFAULT_FILTER_COLORS, alpha=1):
    labels = df["filter"].fillna("unknown").astype(str).values
    colors = [color_map.get(lbl, color_map["unknown"]) for lbl in labels]
    return [to_rgba(color, alpha) for color in colors]


def plot_live_schedule_snapshot(
    completed_df=None,
    candidate_df=None,
    proposed_df=None,
    current=None,
    outfile=None,
    time=None,
):
    """
    Plot the sky in the middle of an observing night.

    Arguments
    ---------
    completed_df : pd.DataFrame [None]
        The fields completed so far in the night.
    candidate_df : pd.DataFrame [None]
        The fields which the model may choose to pick later in the night.
    proposed_df : pd.DataFrame [None]
        The current proposed chunk of fields under consideration.
    current : pd.Series or dict [None]
        The current telescope pointing and filter.
    outfile : str [None]
        If provided, saves the image to this path/filename.
    time : float [None]
        Time (Unix timestamp, in UTC) to center the coordinates; default is now.
    """

    # ensure at least one plottable list of fields
    if (
        _is_empty(completed_df)
        and _is_empty(candidate_df)
        and _is_empty(proposed_df)
        and _is_empty(current)
    ):
        raise ValueError(
            "At least one dataframe must be provided with at least one row."
        )

    # initialize figure at requested time
    observer = ephemerides.blanco_observer(time=time)
    zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", observer=observer)
    skymap = plotting.SkyMap(center_ra=zenith_ra, center_dec=zenith_dec)

    # current pointing: thick outline, top layer, black face color
    if not _is_empty(current):
        skymap.scatter(
            ra=current["ra"],
            dec=current["dec"],
            marker="h",
            s=80,
            facecolor="none",
            edgecolor=DEFAULT_FILTER_COLORS[current.get("filter", "unknown")],
            linewidth=2,
            alpha=1,
            zorder=10,
        )
        skymap.scatter(
            ra=current["ra"],
            dec=current["dec"],
            marker="x",
            s=50,
            color=DEFAULT_FILTER_COLORS[current.get("filter", "unknown")],
            linewidth=1.5,
            alpha=1,
            zorder=10,
        )

    # proposed chunk: thick outline, top layer, order-based face colors
    if not _is_empty(proposed_df):
        order = list(range(len(proposed_df)))
        omin, omax = min(order), max(order)
        omax = omax if omax != omin else omax + 1
        norm = Normalize(vmin=omin, vmax=omax)
        cmap = plt.cm.gist_grey
        facecolors = cmap(norm(order))
        skymap.scatter(
            ra=proposed_df["ra"].values,
            dec=proposed_df["dec"].values,
            marker="h",
            s=80,
            facecolor=facecolors,
            edgecolor=_filter_colors(proposed_df, DEFAULT_FILTER_COLORS),
            linewidth=2,
            alpha=1,
            zorder=9,
        )

        # draw a connecting line
        line_df = proposed_df
        if not _is_empty(current):
            line_df = pd.concat([pd.DataFrame([current]), line_df], ignore_index=True)
        if len(line_df) > 1:
            skymap.plot(
                ra=line_df["ra"],
                dec=line_df["dec"],
                linewidth=0.8,
                linestyle="dotted",
                color="black",
                zorder=8,
            )

        # order-based colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = skymap.fig.colorbar(
            sm, ax=skymap.ax, fraction=0.055, pad=0.07, anchor=(0, 0.95), shrink=0.4
        )
        cbar.locator = MaxNLocator(integer=True)
        cbar.set_label("Proposed Chunk Order")

    # completed fields: clear color, mid layer, filled with filter color
    if not _is_empty(completed_df):
        skymap.scatter(
            ra=completed_df["ra"].values,
            dec=completed_df["dec"].values,
            marker="h",
            s=80,
            facecolor=_filter_colors(completed_df, DEFAULT_FILTER_COLORS, alpha=0.1),
            edgecolor=_filter_colors(completed_df, DEFAULT_FILTER_COLORS),
            linewidth=1,
            zorder=8,
        )

    # candidate fields: dimmed, background layer, unfilled, dotted border
    if not _is_empty(candidate_df):
        skymap.scatter(
            ra=candidate_df["ra"].values,
            dec=candidate_df["dec"].values,
            marker="h",
            s=80,
            facecolor="none",
            edgecolor=_filter_colors(candidate_df, DEFAULT_FILTER_COLORS),
            linewidth=1,
            linestyle="dotted",
            alpha=0.8,
            zorder=7,
        )

    # plot lines through the galactic plane
    l = np.linspace(0, 360, 100) * units.deg
    b = np.zeros_like(l)
    ra, dec = ephemerides.galactic_to_equatorial(l=l, b=b)
    skymap.plot(ra=ra, dec=dec, color="gray", zorder=10, linewidth=0.8)
    for offset in [5 * units.deg, -5 * units.deg]:
        ra, dec = ephemerides.galactic_to_equatorial(l=l, b=b + offset)
        skymap.plot(
            ra=ra, dec=dec, color="gray", zorder=10, linestyle="--", linewidth=0.8
        )

    # plot requested airmass
    az = np.linspace(0, 360, 100) * units.deg
    el = np.ones_like(az) * 90 * units.deg - np.arccos(1 / 1.4)
    ra, dec = ephemerides.topographic_to_equatorial(az=az, el=el, observer=observer)
    skymap.plot(
        ra=ra,
        dec=dec,
        color="gray",
        zorder=10,
        linewidth=0.8,
    )

    # plot the moon
    ra, dec = ephemerides.get_source_ra_dec(source="moon", observer=observer)
    skymap.scatter(
        ra=ra,
        dec=dec,
        facecolor="darkgrey",
        edgecolor="black",
        marker="o",
        s=300,
    )

    # filter color legend
    handles = []
    for filt in "ugrizY":
        handles.append(
            Line2D(
                [0],
                [0],
                marker="h",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor=DEFAULT_FILTER_COLORS[filt],
                markeredgewidth=2.0,
                markersize=8,
                label=filt,
            )
        )
    legend = skymap.ax.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(1.16, 0.35),
        frameon=False,
        title="Filter",
    )
    skymap.ax.add_artist(legend)

    # observation completion legend
    handles = []
    handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label=f"current",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="h",
            linestyle="dotted",
            color="black",
            linewidth=1,
            markerfacecolor="grey",
            markeredgecolor="black",
            markeredgewidth=2.0,
            markersize=8,
            label=f"proposed",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="h",
            linestyle="None",
            markerfacecolor=to_rgba("darkgray", 0.1),
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8,
            label=f"completed",
        )
    )
    handles.append(
        plt.scatter(
            [],
            [],
            marker="h",
            facecolors="white",
            edgecolors="black",
            linestyle="dotted",
            linewidth=1.0,
            s=60,
            label=f"candidate",
        )
    )
    legend = skymap.ax.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(1.13, 0.09),
        frameon=False,
        title="Status",
    )
    skymap.ax.add_artist(legend)

    # set title with date and time
    time = time if time is not None else time_utils.utc_now()
    time = int(time)
    dt_utc = time_utils.unix_to_datetime(time).strftime("%Y-%m-%d %H:%M:%S %Z")
    dt_local = time_utils.unix_to_local_datetime(time).strftime("%Y-%m-%d %H:%M:%S %Z")
    plt.title(
        "Live Scheduling Snapshot\n"
        f"UTC: {dt_utc} | Local: {dt_local} | Unix UTC: {time}"
    )

    # save the file if requested
    if outfile is not None:
        plt.savefig(outfile)
    
    return skymap
