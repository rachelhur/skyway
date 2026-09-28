from datetime import datetime, timezone
from blancops.math import units 
from blancops.ephemerides import ephemerides
import tempfile, os, shutil, glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from tqdm import tqdm
from scipy.stats import circmean
from collections import Counter


class SkyMap:
    """
    A class for making sky maps in RA and Dec using Cartopy. Cartopy expects lon and lat
    coordinates, while RA increases opposite to lon. This class supports orthographic,
    Mollweide, Hammer, and Aitoff projections, and it can plot scatters and lines.
    """

    def __init__(
        self, center_ra=0, center_dec=0, projection="ortho", figsize=(10.5, 8.5), dpi=80
    ):
        """
        Initialize the class.

        Arguments
        ---------
        center_ra, center_dec : float
            Center RA, Dec coordinates in radians.
        projection : str
            Map projection to use
        figsize : tuple of floats
            figsize to pass to matplotlib
        dpi : int
            dpi to pass to matplotlib
        """

        # convert RA center to cartopy longitude
        self.center_lon = -center_ra / units.deg % 360
        self.center_lat = center_dec / units.deg

        # choose a projection
        if projection.lower() in ["ortho", "orthographic"]:
            self.projection = ccrs.Orthographic(
                central_longitude=self.center_lon, central_latitude=self.center_lat
            )
        elif projection.lower() in ["moll", "mollweide"]:
            self.projection = ccrs.Mollweide(central_longitude=self.center_lon)
        elif projection.lower() in ["hammer"]:
            self.projection = ccrs.Hammer(central_longitude=self.center_lon)
        elif projection.lower() in ["aitoff"]:
            self.projection = ccrs.Aitoff(central_longitude=self.center_lon)
        else:
            raise ValueError(f"Unknown projection: {projection}")

        # set up figure and axes
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=self.projection)
        self.ax.set_global()

        # CRS for RA/Dec inputs
        self.input_crs = ccrs.PlateCarree()

        # initialize plot gridlines
        self.ax.gridlines(color="gray", linestyle="dotted", linewidth=0.8)

        # add labels to ra gridlines
        for ra in np.arange(0, 360, 60) * units.deg:
            self.text(
                ra=ra,
                dec=-7 * units.deg,
                label=f"{ra / units.deg:.0f}°",
                outline="white",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                rotation=90,
            )

        # add labels to dec gridlines
        for dec in np.arange(-80, 80 + 1e-5, 20) * units.deg:
            for ra in np.array([10, 190]) * units.deg:
                self.text(
                    ra=ra,
                    dec=dec,
                    label=f"{dec / units.deg:.0f}°",
                    outline="white",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    rotation=0,
                )

        # add East and West labels
        self.ax.annotate(
            "West", xy=(1.01, 0.5), ha="left", xycoords="axes fraction", rotation=-90
        )
        self.ax.annotate(
            "East", xy=(-0.01, 0.5), ha="right", xycoords="axes fraction", rotation=90
        )

    # convert RA (in rad) to the longitude convention (in rad) Cartopy expects
    @staticmethod
    def ra_to_lon(ra_deg):
        return np.unwrap(-np.atleast_1d(ra_deg))

    # public plotting API ---------------------------------

    def scatter(self, ra, dec, **kwargs):
        """
        Plots points on map as a scatter plot.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs
            Options to pass to ax.scatter
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        self.ax.scatter(
            lon / units.deg, lat / units.deg, transform=self.input_crs, **kwargs
        )

    def plot(self, ra, dec, **kwargs):
        """
        Plots points on map as a line plot.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs
            Options to pass to ax.plot
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        self.ax.plot(
            lon / units.deg, lat / units.deg, transform=self.input_crs, **kwargs
        )

    def poly(self, ra, dec, **kwargs):
        """
        Plots a polygon on map.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs
            Options to pass to maplotlib.patches.Polygon
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        verts = np.column_stack([lon, lat]) / units.deg
        patch = Polygon(verts, closed=True, transform=self.input_crs, **kwargs)
        self.ax.add_patch(patch)

    def text(self, ra, dec, label, outline=None, **kwargs):
        """
        Adds text to the map.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to add text
        label : str
            test to add
        outline : str
            If provided, the color to outline the text in
        kwargs
            Options to pass to ax.text
        """
        # draw the text
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        lon = float(np.ravel(lon)[0])
        lat = float(np.ravel(lat)[0])
        txt = self.ax.text(
            lon / units.deg, lat / units.deg, label, transform=self.input_crs, **kwargs
        )

        # outline the text
        if outline is not None:
            txt.set_path_effects([pe.withStroke(linewidth=5, foreground=outline)])

    def line_collection(self, ra, dec, **kwargs):
        """
        Plots a collection of lines on the map.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coords to plot, expects 2d numpy arrays where each row is a line
        kwargs
            Options to pass to matplotlib.collections.LineCollection
        """
        lons = self.ra_to_lon(ra) / units.deg
        lats = np.asarray(dec) / units.deg
        lines = [np.column_stack([lon, lat]) for lon, lat in zip(lons, lats)]
        lc = LineCollection(lines, transform=self.input_crs, **kwargs)
        self.ax.add_collection(lc)


def plot_fields(
    time,
    current_radec,
    completed_radec,
    future_radec,
    plot_zenith=True,
    plot_airmass=1.4,
    plot_galaxy=True,
    plot_moon=True,
    observer=None,
    skymap=None,
    current_kwargs={},
    completed_kwargs={},
    future_kwargs={},
    schedule_label="",
):
    """
    Initialize a sky view and plot of current and completed fields at a selected time.

    Arguments
    ---------
    time : float
        Time (Unix timestamp, in UTC) of current observation.
    current_radec : float tuple
        (ra, dec) tuple in radians for current field to plot.
    completed_radec : list of float tuples
        List of (ra, dec) tuples in radians for completed fields to plot.
    future_radec : list of float tuples
        List of (ra, dec) tuples in radians for future fields to plot.
    plot_zenith : bool [True]
        Whether to plot a circle at zenith
    plot_airmass : float [1.4]
        Plots a circle at constant airmass. Pass 0 for no circle
    plot_galaxy : bool [True]
        Whether to plot a line through the galactic plane +/-1deg
    plot_moon : bool [True]
        Whether to plot a circle at the moon's position
    observer : ephem.Observer [None]
        An observer instance. Defaults to blanco_observer at specified time.
    skymap : SkyMap [None]
        A SkyMap plot instance on which to plot. Defaults to creating a new instance.
    current_kwargs, completed_kwargs, future_kwargs : dict [{}]
        kwargs to pass to SkyMap.scatter while plotting fields, updating the defaults.
    schedule_label : str [""]
        Optional label to append to the plot title (e.g., "Expert Schedule").

    Returns
    -------
    skymap : SkyMap instance
        The created figure.
    """

    # initialize figure at selected time
    observer = ephemerides.blanco_observer(time=time) if observer is None else observer
    zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", observer=observer)
    skymap = (
        SkyMap(center_ra=zenith_ra, center_dec=zenith_dec) if skymap is None else skymap
    )

    # set title to selected time
    title_str = (
        datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC"
    )
    if schedule_label:
        title_str = f"{schedule_label}" + "\n" + title_str
    plt.title(title_str)

    # plot current field
    if not np.any(np.isnan(np.asarray(current_radec, dtype=float))):
        kwargs = dict(
            c="0.6",
            edgecolor="k",
            zorder=10,
            marker="H",
            s=80,
        )
        kwargs.update(current_kwargs)
        skymap.scatter(ra=current_radec[0], dec=current_radec[1], **kwargs)

    # plot completed fields
    kwargs = dict(
        c="0.8",
        edgecolor=None,
        zorder=9,
        marker="H",
        s=80,
    )
    kwargs.update(completed_kwargs)
    completed_radec = np.asarray(completed_radec, dtype=float)
    keep = np.all(~np.isnan(completed_radec), axis=1)
    if len(completed_radec) > 0:
        skymap.scatter(
            ra=completed_radec[keep, 0], dec=completed_radec[keep, 1], **kwargs
        )

    # plot future fields
    kwargs = dict(
        facecolor="none",
        edgecolor="gainsboro",
        zorder=8,
        marker="H",
        s=80,
    )
    kwargs.update(future_kwargs)
    future_radec = np.asarray(future_radec, dtype=float)
    keep = np.all(~np.isnan(future_radec), axis=1)
    if len(future_radec) > 1:
        skymap.scatter(ra=future_radec[:, 0], dec=future_radec[:, 1], **kwargs)

    # plot zenith marking
    if plot_zenith:
        skymap.scatter(
            ra=zenith_ra,
            dec=zenith_dec,
            facecolor="none",
            edgecolor="green",
            zorder=10,
            marker="o",
            s=80,
        )

    # plot lines through the galactic plane
    if plot_galaxy:
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
    if plot_airmass != 0:
        az = np.linspace(0, 360, 100) * units.deg
        el = np.ones_like(az) * 90 * units.deg - np.arccos(1 / plot_airmass)
        ra, dec = ephemerides.topographic_to_equatorial(az=az, el=el, observer=observer)
        skymap.plot(
            ra=ra,
            dec=dec,
            color="green",
            zorder=10,
            linewidth=0.8,
        )

    # plot the moon
    if plot_moon:
        ra, dec = ephemerides.get_source_ra_dec(source="moon", observer=observer)
        skymap.scatter(
            ra=ra,
            dec=dec,
            facecolor="darkgrey",
            edgecolor="black",
            marker="o",
            s=300,
        )

    return skymap


# plot_fields options for drawing fields on top of a plot_bins map
_FIELD_OVERLAY_KWARGS = dict(
    plot_zenith=False,
    plot_airmass=0,
    plot_galaxy=False,
    plot_moon=False,
    current_kwargs={"edgecolor": "darkgreen", "c": "forestgreen", "s": 60},
    completed_kwargs={"edgecolor": "seagreen", "c": "none", "s": 60},
    future_kwargs={"edgecolor": "silver", "s": 60},
)


def plot_bins(
    time,
    current_idx,
    alternate_idx,
    completed_idxs,
    future_idxs,
    nside,
    is_azel=False,
    plot_zenith=True,
    plot_airmass=1.4,
    plot_galaxy=True,
    plot_moon=True,
    sky_bin_mapping=None,
    observer=None,
    skymap=None,
    schedule_label="",
):
    """
    Initialize a sky view binned into healpix grid. Optionally highlight select pixels.

    Arguments
    ---------
    time : float
        Time (Unix timestamp, in UTC) of current observation.
    current_idx : int
        Index of current sky bin, highlighted in green.
    alternate_idx : int
        Index of an alternate sky bin, highlighted in red if different from current. For
        instance, a policy that is being compared to an expert can be compared in red.
    completed_idxs : list of int
        List of sky bin indices with observations yet to be observed.
    future_idxs : list of int
        List of sky bin indices already completed.
    nside : int
        nside used to instantiate the ephemerides.HealpixGrid
    is_azel : bool [False]
        whether the ephemerides.HealpixGrid uses az/el (True) or RA/Dec coords (False)
    plot_zenith : bool [True]
        Whether to plot a circle at zenith
    plot_airmass : float [1.4]
        Plots a circle at constant airmass. Pass 0 for no circle
    plot_galaxy : bool [True]
        Whether to plot a line through the galactic plane +/-1deg
    plot_moon : bool [True]
        Whether to plot a circle at the moon's position
    sky_bin_mapping : dict [None]
        If provided, is used to validate that the recreated healpix grid matches the
        provided grid.
    observer : ephem.Observer [None]
        An observer instance. Defaults to blanco_observer at specified time.
    skymap : SkyMap [None]
        A SkyMap plot instance on which to plot. Defaults to creating a new instance.
    schedule_label : str [""]
        Optional label to append to the plot title (e.g., "Expert Schedule").

    Returns
    -------
    skymap : SkyMap instance
        The created figure.
    """

    # initialize figure at selected time
    observer = ephemerides.blanco_observer(time=time) if observer is None else observer
    zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", observer=observer)
    skymap = (
        SkyMap(center_ra=zenith_ra, center_dec=zenith_dec) if skymap is None else skymap
    )

    # set title to selected time
    title_str = (
        datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC"
    )
    if schedule_label:
        title_str = f"{schedule_label}" + "\n" + title_str
    plt.title(title_str)

    # re-create the healpix grid
    hpgrid = ephemerides.HealpixGrid(nside=nside, is_azel=is_azel)
    if sky_bin_mapping is not None:
        assert len(sky_bin_mapping) == len(hpgrid.lon)
        for idx, (lon, lat) in sky_bin_mapping.items():
            assert int(idx) == hpgrid.ang2idx(lon=lon, lat=lat)

    # plot the healpix grid lines
    ra, dec = hpgrid.get_pixel_boundaries(step=2, loop=True)
    if is_azel:
        ra, dec = ephemerides.topographic_to_equatorial(
            az=ra, el=dec, observer=observer
        )
    keep = (
        hpgrid.get_source_angular_separations("zenith", observer=observer)
        < 90 * units.deg
    )  # save time: plot bins above horizon
    skymap.line_collection(
        ra=ra[
            keep, : ra.shape[1] // 2 + 1
        ],  # save time, no overlap: plot half the boundary
        dec=dec[keep, : ra.shape[1] // 2 + 1],
        color="pink",
        linewidths=0.8,
        linestyles=":",
    )

    # mark the current sky bin
    skymap.poly(
        ra=ra[current_idx],
        dec=dec[current_idx],
        facecolor="none",
        edgecolor="darkgreen",
        zorder=10,
        linewidth=2,
    )

    # mark the alternate sky bin
    if alternate_idx != current_idx:
        skymap.poly(
            ra=ra[alternate_idx],
            dec=dec[alternate_idx],
            facecolor="none",
            edgecolor="red",
            hatch="xxx",
            linewidth=2,
            zorder=10,
        )

    # count up completed and total visits per bin
    completed_counts = Counter(completed_idxs)
    completed_counts[current_idx] += 1
    total_counts = completed_counts + Counter(future_idxs)

    # shade in the sky bins to be visited
    for idx in set(future_idxs) - set(completed_counts):
        skymap.poly(
            ra=ra[idx],
            dec=dec[idx],
            facecolor="silver",
            alpha=0.6,
            zorder=9,
        )

    # plot counts for completed bins
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        return colors.LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n)),
        )

    cmap = truncate_colormap(cm.Greens, 0.3, 1.0)
    norm = colors.Normalize(vmin=1, vmax=max(total_counts.values()))
    for idx, count in completed_counts.items():
        skymap.poly(
            ra=ra[idx],
            dec=dec[idx],
            facecolor=cmap(norm(count)),
            alpha=0.6,
            zorder=9,
        )

    # colorbar for completed bins
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = skymap.fig.colorbar(
        sm, ax=skymap.ax, orientation="vertical", fraction=0.046, pad=0.07
    )
    cbar.locator = MaxNLocator(integer=True)
    cbar.set_label("Completed observations")

    # plot zenith marking
    if plot_zenith:
        skymap.scatter(
            ra=zenith_ra,
            dec=zenith_dec,
            facecolor="none",
            edgecolor="green",
            zorder=10,
            marker="o",
            s=80,
        )

    # plot lines through the galactic plane
    if plot_galaxy:
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
    if plot_airmass != 0:
        az = np.linspace(0, 360, 100) * units.deg
        el = np.ones_like(az) * 90 * units.deg - np.arccos(1 / plot_airmass)
        ra, dec = ephemerides.topographic_to_equatorial(az=az, el=el, observer=observer)
        skymap.plot(
            ra=ra,
            dec=dec,
            color="green",
            zorder=10,
            linewidth=0.8,
        )

    # plot the moon
    if plot_moon:
        ra, dec = ephemerides.get_source_ra_dec(source="moon", observer=observer)
        skymap.scatter(
            ra=ra,
            dec=dec,
            facecolor="darkgrey",
            edgecolor="black",
            marker="o",
            s=300,
        )

    return skymap


def plot_schedule_movie(
    outfile: str,
    times: np.ndarray,
    field_pos: np.ndarray | None = None,
    bin_idxs: np.ndarray | None = None,
    alternate_bin_idxs: np.ndarray | None = None,
    nside: int | None = None,
    is_azel: bool = False,
    sky_bin_mapping: dict | None = None,
    schedule_label: str = "",
) -> None:
    """
    Creates a gif of fields observed over the course of a night.

    Arguments
    ---------
    outfile : str
        Path to output gif file.
    times : list of float
        List of times (Unix timestamps, in UTC) of observations.
    field_pos : list of float tuples [None]
        List of field (ra, dec) for each observation. If provided, plots specific fields
        overlaid on the bins
    bin_idxs : list of float [None]
        List of bin indices for each observation. If provided, plots the sky bins
    alternate_bin_idxs : list of float [None]
        List of alternate bin indices for each observation. Defaults to copy bins
    nside : int [None]
        nside used to make the ephemerides.HealpixGrid. Required if bin_idxs provided.
    is_azel : bool [False]
        whether the ephemerides.HealpixGrid uses az/el (True) or RA/Dec coords (False)
    sky_bin_mapping : dict [None]
        If provided, is used to validate that the recreated healpix grid matches the
        provided grid.
    schedule_label : str [""]
        Optional label to append to the plot title (e.g., "Expert Schedule").
    """

    # check required arguments
    if field_pos is None and bin_idxs is None:
        raise ValueError("Must provide field_pos and/or bin_idxs")
    if bin_idxs is not None and nside is None:
        raise ValueError("Must specify nside if plotting bin_idxs")

    # ensure output file is gif
    if os.path.splitext(outfile)[-1] not in [".gif"]:
        raise NotImplementedError("Only animated gif currently supported.")

    # create temporary directory for temporary png files
    tmpdir = tempfile.mkdtemp()

    # duplicate bins if alternate bins is not given
    alternate_bin_idxs = alternate_bin_idxs if alternate_bin_idxs is not None else bin_idxs

    # make sure field pos is numpy array
    field_pos = None if field_pos is None else np.asarray(field_pos)

    # plot each observation successively, saving pngs
    plt.ioff()
    for i, time in enumerate(tqdm(times)):
        # plot the sky bins
        skymap = None
        if bin_idxs is not None:
            skymap = plot_bins(
                time,
                current_idx=bin_idxs[i],
                alternate_idx=alternate_bin_idxs[i],
                completed_idxs=bin_idxs[:i],
                future_idxs=bin_idxs[i + 1 :],
                nside=nside,
                sky_bin_mapping=sky_bin_mapping,
                is_azel=is_azel,
                schedule_label=schedule_label,
            )

        # plot the sky fields on the sky map
        if field_pos is not None:
            skymap = plot_fields(
                time,
                current_radec=field_pos[i, :],
                completed_radec=field_pos[:i, :],
                future_radec=field_pos[i + 1 :, :],
                skymap=skymap,
                schedule_label=schedule_label,
                **(_FIELD_OVERLAY_KWARGS if bin_idxs is not None else {}),
            )

        # save the figure
        plt.savefig(os.path.join(tmpdir, "field_%08i.png" % i))
        plt.close(skymap.fig)
    plt.ion()

    # convert pngs to gif
    pngs = sorted(glob.glob(os.path.join(tmpdir, "*.png")))
    if not pngs:
        shutil.rmtree(tmpdir)
        raise RuntimeError("No PNG frames were generated for plot_schedule_movie()")
    images = [imageio.imread(p) for p in pngs]
    imageio.mimsave(outfile, images, duration=0.10, loop=0)
    shutil.rmtree(tmpdir)

    return


def plot_schedule_whole(
    outfile,
    times,
    field_pos=None,
    bin_idxs=None,
    alternate_bin_idxs=None,
    nside=None,
    sky_bin_mapping=None,
    projection="mollweide",
    center_pos=(None, None),
    schedule_label="",
):
    """
    Creates an image of the areas visited at any point in the schedule on a Mollweide
    sky projection. Optionally plots bins or fields.

    Arguments
    ---------
    outfile : str
        Path to output image file.
    times : list of float
        List of times (Unix timestamps, in UTC) of observations.
    field_pos : list of float tuples [None]
        List of field (ra, dec) for each observation. If provided, plots specific fields
        visited at any point during the schedule.
    bin_idxs : list of int [None]
        List of bin indices for each observation. If provided, expects nside. Plots the
        sky bins visited at any point during the schedule, colored by number of times
        visited.
    alternate_bin_idxs : list of int [None]
        List of bin indices for each observation in an alternate schedule. If provided,
        expects nside. Plots the difference in the sky bins visited at any point during
        the provided schedules, colored by the difference in number of times visited.
    nside : int [None]
        nside used to make the ephemerides.HealpixGrid. Required if bin_idxs provided.
    sky_bin_mapping : dict [None]
        If provided, is used to validate that the recreated healpix grid matches the
        provided grid.
    projection : str ["mollweide"]
        Map projection. Options: ortho/orthographic, moll/mollweide, hammer, aitoff
    center_pos : float tuple (None, None)
        Center (ra, dec) for the map. Default plots the average (ra, dec) for the
        schedule, using field positions instead of bin positions if available.
    schedule_label : str [""]
        Optional label to append to the plot title (e.g., "Expert Schedule").
    """

    # check required arguments
    if field_pos is None and bin_idxs is None:
        raise ValueError("Must provide field_pos and/or bin_idxs")
    if bin_idxs is not None and nside == 0:
        raise ValueError("Must specify nside if plotting bin_idxs")
    if alternate_bin_idxs is not None and bin_idxs is None:
        raise ValueError("Provide bin_idxs when also specifying an alternate schedule")

    # make sure field pos is numpy array
    field_pos = None if field_pos is None else np.asarray(field_pos, dtype=float)

    # re-create the healpix grid
    hpgrid = None
    if bin_idxs is not None:
        hpgrid = ephemerides.HealpixGrid(nside=nside, is_azel=False)
        if sky_bin_mapping is not None:
            assert len(sky_bin_mapping) == len(hpgrid.lon)
            for idx, (lon, lat) in sky_bin_mapping.items():
                assert int(idx) == hpgrid.ang2idx(lon=lon, lat=lat)

    # calculate center ra, dec
    center_ra, center_dec = center_pos
    if center_ra is None:
        ra = field_pos[:, 0] if field_pos is not None else hpgrid.lon[bin_idxs]
        center_ra = circmean(ra, nan_policy="omit")
    if center_dec is None:
        dec = field_pos[:, 1] if field_pos is not None else hpgrid.lat[bin_idxs]
        center_dec = np.nanmean(dec)

    # initialize figure with selected central positioning
    observer = ephemerides.blanco_observer(time=np.mean(times))
    skymap = SkyMap(center_ra=center_ra, center_dec=center_dec, projection=projection)

    # plot sky bins
    if bin_idxs is not None:
        # plot the healpix grid lines
        ra, dec = hpgrid.get_pixel_boundaries(step=2, loop=True)
        skymap.line_collection(
            ra=ra[
                :, : ra.shape[1] // 2 + 1
            ],  # save time, no overlap: plot half the boundary
            dec=dec[:, : ra.shape[1] // 2 + 1],
            color="pink",
            linewidths=0.8,
            linestyles=":",
        )

        # determine colors for counts of visits to each bin_idx
        if alternate_bin_idxs is None:
            counts = Counter(bin_idxs)

            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
                return colors.LinearSegmentedColormap.from_list(
                    f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
                    cmap(np.linspace(minval, maxval, n)),
                )

            cmap = truncate_colormap(cm.Greys, 0.3, 1.0)
            norm = colors.Normalize(vmin=1, vmax=max(counts.values()))
            clabel = "Number of observations"

        # determine colors for differences between provided schedules
        else:
            counts = Counter(bin_idxs)
            counts.subtract(Counter(alternate_bin_idxs))
            cmap = cm.PRGn
            vscale = max(abs(max(counts.values())), abs(min(counts.values())))
            norm = colors.Normalize(vmin=-vscale, vmax=vscale)
            clabel = "Difference in number of observations"

        # shade in the sky bin counts
        for idx, count in counts.items():
            if idx != -1 and count != 0:
                skymap.poly(
                    ra=ra[idx],
                    dec=dec[idx],
                    facecolor=cmap(norm(count)),
                    alpha=0.6,
                    zorder=9,
                )

        # colorbar for sky bin counts
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = skymap.fig.colorbar(
            sm, ax=skymap.ax, orientation="vertical", fraction=0.046, pad=0.07
        )
        cbar.locator = MaxNLocator(integer=True)
        cbar.set_label(clabel)

    # plot fields
    if field_pos is not None:
        # color the fields by time
        keep = np.all(~np.isnan(field_pos), axis=1)
        cmap = cm.viridis
        norm = colors.Normalize(vmin=min(times[keep]), vmax=max(times[keep]))
        c = cmap(norm(times[keep]))

        # plot the actual fields
        skymap.scatter(
            ra=field_pos[keep, 0],
            dec=field_pos[keep, 1],
            edgecolor=c,
            facecolor=c if bin_idxs is None else "none",
            alpha=0.6 if bin_idxs is None else 1.0,
            zorder=10,
            marker="H",
            s=60 if "ortho" in projection else 20,
        )

        # colorbar for fields
        ts = np.linspace(min(times[keep]), max(times[keep]), 5)
        ls = [
            datetime.fromtimestamp(t, tz=timezone.utc).strftime("%m/%d %H:%M")
            for t in ts
        ]
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = skymap.fig.colorbar(sm, ax=skymap.ax, label="Datetime")
        cbar.set_ticks(ticks=ts, labels=ls, rotation=45, va="bottom")

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

    # plot the moon
    ra, dec = ephemerides.get_source_ra_dec(source="moon", observer=observer)
    skymap.scatter(
        ra=ra,
        dec=dec,
        facecolor="darkgrey",
        edgecolor="black",
        marker="o",
        s=300 if "ortho" in projection else 100,
    )

    # set plot title
    title_str = (
        datetime.fromtimestamp(times[0], tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC -- "
        + datetime.fromtimestamp(times[-1], tz=timezone.utc).strftime(
            "%Y/%m/%d %H:%M:%S"
        )
        + " UTC"
    )
    if schedule_label:
        title_str = f"{schedule_label}" + "\n" + title_str
    plt.title(title_str)

    # save the figure
    plt.savefig(outfile)
    plt.close(skymap.fig)

    return


def _lookup_field_pos(field_ids, field_id2pos):
    """
    Map a column of field ids to (ra, dec) positions.

    Field ids read from a schedule csv are floats when the column holds missing
    values, so they are matched against the mapping both as-written and coerced
    to an integer key.

    Arguments
    ---------
    field_ids : pandas.Series
        Field ids for each observation.
    field_id2pos : dict
        Maps field id (as a string key) to (ra, dec) in radians.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_obs, 2) holding (ra, dec) in radians, with NaN where
        the field id was missing from the mapping.
    """
    positions = []
    n_missing = 0
    for fid in field_ids.values:
        pos = field_id2pos.get(str(fid))
        if pos is None:
            try:
                pos = field_id2pos.get(str(int(float(fid))))
            except (TypeError, ValueError):
                pos = None
        if pos is None:
            n_missing += 1
            pos = [np.nan, np.nan]
        positions.append(pos)

    if n_missing == len(positions):
        raise KeyError(
            "None of the %i field ids were found in the field mapping; "
            "check that the mapping matches the schedule." % len(positions)
        )
    if n_missing:
        print(
            "Warning: %i of %i field ids missing from the field mapping."
            % (n_missing, len(positions))
        )

    return np.asarray(positions, dtype=float)


def plot_schedule_from_file(
    outfile,
    schedule_file,
    plot_type,
    nside=None,
    fields_file=None,
    bins_file=None,
    whole=False,
    compare=False,
    expert=False,
    is_azel=False,
    mollweide=False,
):
    """
    Load a schedule from file and create plots (movie or whole-schedule view).

    This is a convenience function that wraps the plotting logic for programmatic use.
    For command-line usage, use the script directly (see `if __name__ == "__main__"`).

    Arguments
    ---------
    outfile : str
        Path to output file (expects .gif for movies, .png etc for whole-schedule plots)
    schedule_file : str
        Path to the schedule CSV file with keys "expert_timestamp" and/or
        "agent_timestamp", plus some combination of "expert_field_id", "agent_field_id",
        "expert_bin_id", "agent_bin_id".
    plot_type : str
        Type of plot: "field", "bin", or "fieldbin".
    nside : int [None]
        nside for healpix sky bins. Required if plot_type is "bin" or "fieldbin".
    fields_file : str [None]
        Path to field mapping JSON file (maps field_id to (ra, dec) in radians).
    bins_file : str [None]
        Path to bin mapping JSON file (maps bin_id to (ra, dec) in radians). Not
        required if nside is provided, but used for validation if present.
    whole : bool [False]
        If True, plot whole schedule on flat sky. If False, create movie.
    compare : bool [False]
        If True, plot comparison of expert vs agent schedules.
    expert : bool [False]
        If True, plot expert schedule as primary. If False, plot agent schedule.
    is_azel : bool [False]
        If True, use az/el healpix grid instead of RA/Dec for bin plots.
    mollweide : bool [False]
        If True, use Mollweide projection. If False, use orthographic projection.

    Returns
    -------
    None
        Creates output file(s) as side effect.
    """
    import json
    import pandas as pd

    # check argument compatibility
    if plot_type not in ["field", "bin", "fieldbin"]:
        raise ValueError(
            f'plot_type must be "field", "bin", or "fieldbin", got "{plot_type}"'
        )
    if plot_type in ["field", "fieldbin"] and fields_file is None:
        raise ValueError("fields_file required to plot fields.")
    if plot_type in ["bin", "fieldbin"] and nside is None:
        raise ValueError("nside required to plot bins.")
    if plot_type == "field" and compare and not whole:
        raise NotImplementedError("Comparing fields for 2 schedules not implemented.")
    if mollweide and not whole:
        raise NotImplementedError("Currently, can only plot ortho for movies.")
    if is_azel and plot_type not in ["bin", "fieldbin"]:
        raise NotImplementedError("Az/el healpix only implemented for bin plots.")
    if is_azel and whole:
        raise NotImplementedError("Az/el healpix not implemented for whole-sky plots.")

    # load schedule file and check validity
    schedule = pd.read_csv(schedule_file)

    # determine primary and alternate schedule prefixes
    primary_prefix = "expert" if expert else "agent"
    alternate_prefix = "agent" if expert else "expert"
    primary_label = "Expert Schedule" if expert else "Agent Schedule"
    alternate_label = "Agent Schedule" if expert else "Expert Schedule"
    primary_time_key = f"{primary_prefix}_timestamp"
    alternate_time_key = f"{alternate_prefix}_timestamp"

    # check for required columns
    if primary_time_key not in schedule.columns:
        raise KeyError(f'Missing "{primary_time_key}" required to make requested plot.')
    if compare and alternate_time_key not in schedule.columns:
        raise KeyError(f'Missing "{alternate_time_key}" required to compare schedules.')
    if plot_type in ["field", "fieldbin"]:
        if f"{primary_prefix}_field_id" not in schedule.columns:
            raise KeyError(
                f'Missing "{primary_prefix}_field_id" required to make requested plot.'
            )
        if compare and f"{alternate_prefix}_field_id" not in schedule.columns:
            raise KeyError(
                f'Missing "{alternate_prefix}_field_id" required to compare schedules.'
            )
    if plot_type in ["bin", "fieldbin"]:
        if f"{primary_prefix}_bin_id" not in schedule.columns:
            raise KeyError(
                f'Missing "{primary_prefix}_bin_id" required to make requested plot.'
            )
        if compare and f"{alternate_prefix}_bin_id" not in schedule.columns:
            raise KeyError(
                f'Missing "{alternate_prefix}_bin_id" required to compare schedules.'
            )

    # read in times and filter out unphysical entries
    times = np.asarray(schedule[primary_time_key].values, dtype=int)
    valid = times != 0
    schedule = schedule[valid].reset_index(drop=True)
    times = times[valid]

    # load field and bin mappings
    field_id2pos = None
    if fields_file is not None:
        with open(fields_file) as f:
            field_id2pos = json.load(f)
    bin_id2pos = None
    if bins_file is not None:
        with open(bins_file) as f:
            bin_id2pos = json.load(f)

    # parse field, bin positions
    field_ids_1 = schedule.get(f"{primary_prefix}_field_id", None)
    field_ids_2 = (
        schedule.get(f"{alternate_prefix}_field_id", None) if compare else None
    )
    if field_ids_1 is None or field_id2pos is None:
        field_pos_1 = None
    else:
        field_pos_1 = _lookup_field_pos(field_ids_1, field_id2pos)
    if field_ids_2 is None or field_id2pos is None:
        field_pos_2 = None
    else:
        field_pos_2 = _lookup_field_pos(field_ids_2, field_id2pos)
    bin_ids_1 = schedule.get(f"{primary_prefix}_bin_id", None)
    bin_ids_2 = schedule.get(f"{alternate_prefix}_bin_id", None) if compare else None

    # plot whole schedules
    if whole:
        if not compare:  # 1 plot of plot_type style
            ofs = [outfile]
            fps = [field_pos_1 if plot_type in ["field", "fieldbin"] else None]
            bis = [bin_ids_1 if plot_type in ["bin", "fieldbin"] else None]
            abis = [None]
            slabels = [primary_label]
        else:  # 2 (field) or 3 (bin/fieldbin) plots when comparing
            base, ext = os.path.splitext(outfile)
            if "bin" in plot_type:
                ofs = [base + "_%s" % i + ext for i in range(3)]
                fps = (
                    [field_pos_1, field_pos_2, None]
                    if "field" in plot_type
                    else [None, None, None]
                )
                bis = [bin_ids_1, bin_ids_2, bin_ids_1]
                abis = [None, None, bin_ids_2]
                slabels = [
                    primary_label,
                    alternate_label,
                    f"{primary_label} vs {alternate_label}",
                ]
            else:
                ofs = [base + "_%s" % i + ext for i in range(2)]
                fps = [field_pos_1, field_pos_2]
                bis = [None, None]
                abis = [None, None]
                slabels = [primary_label, alternate_label]
        for i, (of, fp, bi, abi, slabel) in enumerate(
            zip(ofs, fps, bis, abis, slabels)
        ):
            plot_schedule_whole(
                outfile=of,
                times=times,
                field_pos=fp,
                bin_idxs=bi,
                alternate_bin_idxs=abi,
                nside=nside,
                sky_bin_mapping=bin_id2pos,
                projection="mollweide" if mollweide else "ortho",
                center_pos=(None, None),
                schedule_label=slabel,
            )

    # plot movies of fields, bins, or combined field+bin
    else:
        do_bins = plot_type in ["bin", "fieldbin"]
        plot_schedule_movie(
            outfile=outfile,
            times=times,
            field_pos=field_pos_1 if plot_type in ["field", "fieldbin"] else None,
            bin_idxs=bin_ids_1.values if do_bins else None,
            alternate_bin_idxs=bin_ids_2.values if (do_bins and compare) else None,
            nside=nside,
            is_azel=is_azel,
            sky_bin_mapping=bin_id2pos,
            schedule_label=primary_label,
        )


if __name__ == "__main__":
    import argparse as ap

    # command line arguments
    parser = ap.ArgumentParser(
        description="Create a gif of observed fields over the course of a night."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help="Path to output file (expects .gif for movies).",
    )
    parser.add_argument(
        "-s",
        "--schedule",
        type=str,
        required=True,
        help=(
            'Path to the schedule file, a csv file with keys "expert_timestamp" and/or '
            '"agent_timestamp", plus some combination of "expert_field_id", '
            '"agent_field_id", "expert_bin_id", "agent_bin_id".'
        ),
    )
    parser.add_argument(
        "-t",
        "--plot-type",
        type=str,
        required=True,
        choices=["field", "bin", "fieldbin"],
        help=(
            'Whether to plot schedule of "field", "bin", or combined "fieldbin". '
            'Requires "(expert_/agent_)field_id" and/or "(expert_/agent_)bin_id" keys, '
            "respectively."
        ),
    )
    parser.add_argument(
        "-w",
        "--whole",
        action="store_true",
        help="Switch to plot the whole schedule on a flat sky. Default plots movies.",
    )
    parser.add_argument(
        "-c",
        "--compare",
        action="store_true",
        help=(
            "Switch to plot a comparison of two schedules (expert vs agent). "
            "Requires both expert_* and agent_* columns in the schedule file."
        ),
    )
    parser.add_argument(
        "-e",
        "--expert",
        action="store_true",
        help=(
            'Switch to plot "expert_*" schedule keys as the primary schedule. '
            'Default is to plot "agent_*" schedule keys as the primary schedule. '
            "When comparing, the alternate schedule will be the other one."
        ),
    )
    parser.add_argument(
        "-f",
        "--fields",
        type=str,
        help=(
            "Path to field file, a json file that maps field_id to (ra, dec) in rad. "
            "Used for making field plots."
        ),
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=str,
        help=(
            "Path to bin file, a json file that maps bin_id to (ra, dec) in rad. Not "
            "needed if nside provided, in which case this is used to verify grid "
            "reconstruction. Used for making bin plots."
        ),
    )
    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        help="nside used to make healpix sky bin for bin schedules",
    )
    parser.add_argument(
        "-a",
        "--azel",
        action="store_true",
        help=(
            "Switch to use az/el healpix grid instead of RA/Dec grid for bin plots. "
            "Expects the schedule bin_ids and bin file to map id to (az, el) in rad."
        ),
    )
    parser.add_argument(
        "-m",
        "--mollweide",
        action="store_true",
        help="Use Mollweide projection for instead of default Ortho.",
    )
    args = parser.parse_args()

    # call the main plotting function
    plot_schedule_from_file(
        outfile=args.outfile,
        schedule_file=args.schedule,
        plot_type=args.plot_type,
        nside=args.nside,
        fields_file=args.fields,
        bins_file=args.bins,
        whole=args.whole,
        compare=args.compare,
        expert=args.expert,
        is_azel=args.azel,
        mollweide=args.mollweide,
    )
