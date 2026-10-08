import ephem
from datetime import datetime, timezone
import numpy as np
import healpy as hp
from skyway.math import units
from skyway.ephemerides import time_utils


def blanco_observer(time=None):
    """
    Define an ephem.Observer for the Blanco telescope location. Location info from:
    https://www.ctio.noirlab.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon

    Arguments
    ---------
    time : float [None]
        Time (Unix timestamp, in UTC) at which to define observer. Default: now.

    Returns
    -------
    observer : ephem.Observer
        Ephem observer object for Blanco location at specified time.
    """

    # define the location
    observer = ephem.Observer()
    observer.lat = "-30:10:10.78"  # -30.169661 deg
    observer.lon = "-70:48:23.49"  # -70.806525 deg
    observer.elevation = 2206.8  # m

    # initialize time
    time = time if time is not None else time_utils.utc_now()
    observer.date = time_utils.unix_to_datetime(time).strftime("%Y/%m/%d %H:%M:%S")

    return observer


def equatorial_to_topographic(ra, dec, time=None, observer=None):
    """
    Convert RA/Dec to Az/El for the Blanco telescope location.

    Arguments
    ---------
    ra : float or array of floats
        Right ascension in radians
    dec : float or array of floats
        Declination in radians
    time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.
    observer : ephem.Observer [None]
        Observer object. If not provided, defaults to Blanco observer at chosen time.

    Returns
    -------
    az, el : float
        Azimuth and elevation in radians
    """

    # initialize observer location and time
    observer = observer if observer is not None else blanco_observer(time=time)

    # ephem is not vectorizable, so compute each conversion separately
    az, el = [], []
    for r, d in zip(np.atleast_1d(ra), np.atleast_1d(dec)):
        # define position in equatorial coordinates
        source = ephem.FixedBody()
        source._ra = r
        source._dec = d

        # compute topographic position for the observer
        source.compute(observer)
        az.append(float(source.az))
        el.append(float(source.alt))

    # return outputs
    if len(az) == 1:
        return az[0], el[0]
    else:
        return np.asarray(az), np.asarray(el)


def topographic_to_equatorial(az, el, time=None, observer=None):
    """
    Convert Az/El to RA/Dec for the Blanco telescope location.

    Arguments
    ---------
    az : floats or arrays of floats
        Azimuth in radians
    el : float or arrays of floats
        Elevation in radians
    time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.
    observer : ephem.Observer [None]
        Observer object. If not provided, defaults to Blanco observer at chosen time.

    Returns
    -------
    ra, dec : float
        Right ascension and declination in radians
    """

    # initialize observer location and time
    observer = observer if observer is not None else blanco_observer(time=time)

    # compute topographic position for the observer
    if not np.iterable(az):
        return observer.radec_of(az, el)

    # compute Nd arrays by iterating through flattened 1d arrays
    else:
        shape = np.asarray(az).shape
        ra, dec = np.array(
            [
                observer.radec_of(a, e)
                for a, e in zip(np.asarray(az).flatten(), np.asarray(el).flatten())
            ]
        ).T
        return ra.reshape(shape), dec.reshape(shape)


def galactic_to_equatorial(l, b):
    """
    Convert galactic longitude l / latitude b to equatorial Ra/Dec.

    Arguments
    ---------
    l, b : floats or arrays of floats
        Galactic longitude and latitude in radians

    Returns
    -------
    ra, dec : floats or arrays of floats
        Right ascension and declination in radians
    """
    from astropy.coordinates import SkyCoord

    # convert units
    gal = SkyCoord(
        l=np.asarray(l) / units.deg,
        b=np.asarray(b) / units.deg,
        frame="galactic",
        unit="deg",
    )
    icrs = gal.icrs
    ra, dec = icrs.ra.rad, icrs.dec.rad

    return ra if np.iterable(ra) else ra.item(), dec if np.iterable(dec) else dec.item()


def equatorial_to_galactic(ra, dec):
    """
    Convert equatorial Ra/Dec to galactic longitude l / latitude b.

    Arguments
    ---------
    ra, dec : floats or arrays of floats
        Right ascension and declination in radians

    Returns
    -------
    l, b : floats or arrays of floats
        Galactic longitude and latitude in radians
    """
    from astropy.coordinates import SkyCoord

    icrs = SkyCoord(
        ra=np.asarray(ra) / units.deg,
        dec=np.asarray(dec) / units.deg,
        frame="icrs",
        unit="deg",
    )
    gal = icrs.galactic
    l, b = gal.l.rad, gal.b.rad

    return l if np.iterable(l) else l.item(), b if np.iterable(b) else b.item()


def equatorial_to_hour_angle(ra, dec, time=None, observer=None):
    """
    Compute hour angle of specified RA/Dec coordinate at a specified time. Uses the
    convention bounded to (-pi, pi].

    Arguments
    ---------
    ra : float or array of floats
        Right ascension in radians
    dec : float or array of floats
        Declination in radians
    time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.
    observer : ephem.Observer [None]
        Observer object. If not provided, defaults to Blanco observer at chosen time.

    Returns
    -------
    hour_angle : float or array of floats
        Hour angle in radians of the requested coordinates.
    """

    # initialize observer location and time
    observer = observer if observer is not None else blanco_observer(time=time)

    # ephem is not vectorizable, so compute each conversion separately
    hour_angle = []
    for r, d in zip(np.atleast_1d(ra), np.atleast_1d(dec)):
        # define position in equatorial coordinates
        source = ephem.FixedBody()
        source._ra = r
        source._dec = d

        # compute topographic position for the observer
        source.compute(observer)
        hour_angle.append(source.ha.znorm)

    # return outputs
    return hour_angle[0] if len(hour_angle) == 1 else np.asarray(hour_angle)


def get_source_ra_dec(source, time=None, observer=None):
    """
    Get the astrophysical coordinates of a known source using pyephem.

    Arguments
    ---------
    source : str
        Source name. Options: "moon", "sun", "zenith"
    time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.
    observer : ephem.Observer [None]
        Observer object. If not provided, defaults to Blanco observer at chosen time.

    Returns
    -------
    ra, dec : float
        Source position in radians
    """

    # check validity of source
    source = source.lower()
    if source not in {"zenith", "moon", "sun"}:
        raise NotImplementedError("Getting ephemerides for invalid source: " + source)

    # create observer at Blanco telescope
    observer = observer if observer is not None else blanco_observer(time=time)

    # compute zenith location
    if source == "zenith":
        return topographic_to_equatorial(0, 90 * units.deg, observer=observer)

    # compute locations for pyephem sources
    elif source == "moon":
        body = ephem.Moon()
    elif source == "sun":
        body = ephem.Sun()
    body.compute(observer)
    return float(body.ra), float(body.dec)


def get_source_set_time(source, time=None, observer=None, horizon=0):
    """
    Get the Unix timestamp of the next setting of a known source.

    Arguments
    ---------
    source : str
        Source name. Options: "moon", "sun".
    time : float [None]
        Time (Unix timestamp, in UTC) from which to search forward. Default: now.
    observer : ephem.Observer [None]
        Observer object. If not provided, defaults to Blanco observer at chosen time.
    horizon : float [0]
        Horizon elevation (deg) defining the set event.

    Returns
    -------
    set_time : float
        Unix timestamp (UTC) of the first setting after `time`.
    """

    source = source.lower()
    if source not in {"moon", "sun"}:
        raise NotImplementedError("Getting set time for invalid source: " + source)

    observer = observer if observer is not None else blanco_observer(time=time)
    observer.horizon = str(horizon)

    if source == "moon":
        body = ephem.Moon()
    elif source == "sun":
        body = ephem.Sun()
    body.compute(observer)

    ephem_date = observer.next_setting(body)
    return ephem_date.datetime().replace(tzinfo=timezone.utc).timestamp()


class HealpixGrid:
    """
    A class for creating a grid over az/el or ra/dec using healpix.
    """

    def __init__(self, nside, is_azel=True):
        """
        Initialize the grid.

        Arguments
        ---------
        nside : int
            The healpix resolution parameter. npix = 12 * nside^2
        is_azel : bool [True]
            Whether grid points (labeled as lon/lat) represent az/el (True) or ra/dec
            (False). If True, only pixels in the visible hemisphere (el > 0) are kept.
        """

        # store initial arguments
        self.nside = nside
        self.is_azel = is_azel

        # track native healpix indices, which may change when pixels are sorted/filtered
        self.npix = hp.nside2npix(self.nside)
        self.heal_idx = np.arange(self.npix)
        self.idx_lookup = {heal_i: i for i, heal_i in enumerate(self.heal_idx)}

        # get pixel centers in spherical coords (supplied in deg)
        self.lon, self.lat = hp.pix2ang(self.nside, self.heal_idx, lonlat=True)
        self.lon *= units.deg
        self.lat *= units.deg

        # keep only bins above the horizon
        if self.is_azel:
            keep = self.lat > 0
            self.lon = self.lon[keep]
            self.lat = self.lat[keep]
            self.heal_idx = self.heal_idx[keep]
            self.idx_lookup = {heal_i: i for i, heal_i in enumerate(self.heal_idx)}

    def ang2idx(self, lon, lat):
        """
        Wrapper for healpy.pixelfunc.ang2pix that returns the pixel index corresponding
        to the given lon/lat position according to pix order in current class instance.

        Arguments
        ---------
        lon, lat : float, scalars or array-like
            Angular coordinates of a point on the sphere. Provide in the same coordinate
            system as the native pixel system (i.e. az/el if self.is_azel, else ra/dec)

        Returns
        -------
        idx : int or array of int
            Index of currently stored lon/lat pixels corresponding the given positions.
            Returns None for coords outside the grid
        """

        # get healpix indices for requested positions
        heal_idx = hp.ang2pix(
            nside=self.nside,
            theta=np.asarray(lon) / units.deg,
            phi=np.asarray(lat) / units.deg,
            lonlat=True,
            latauto=True,
            latbounce=False,
        )

        # map healpix indices onto the indices for the current grid order
        if np.iterable(heal_idx):
            return np.array([self.idx_lookup.get(i, None) for i in heal_idx])
        else:
            return self.idx_lookup.get(heal_idx, None)

    def get_pixel_boundaries(self, idx=None, step=1, loop=True):
        """
        For each pixel stored in the grid, computes the pixel boundaries of the pixel in
        the grid's native lon/lat coordinates (az/el if is_azel is True, else RA/Dec).

        Arguments
        ---------
        idx : int or list-like of ints [None]
            The pixel indices to get the boundaries for. Default returns all pixels.
        step : int [1]
            Number of elements for each side of the pixel. Default 1 returns only the
            corners of the pixels
        loop : bool [True]
            Whether to loop back to the starting point for each boundary.

        Returns
        -------
        boundaries_lon, boundaries_lat : 2d np.array
            Arrays of size n_pixels by 4*step: the lon/lat coordinates of the pixel
            boundaries.
        """

        # format requested index choice
        idx = self.heal_idx if idx is None else np.atleast_1d(idx)

        # containers for outputs
        vertices_ra = []
        vertices_dec = []

        # get boundaries for pixels, specified as xyz points on unit sphere
        # shape: n_pixels x 3 (xyz) x 4*step (points along boundary)
        boundaries = hp.boundaries(self.nside, idx, step)

        # convert xyz vector positions to lon, lat vector positions
        # shape: 2 (lon/lat) x n_pixels*4*nstep
        boundaries = hp.vec2ang(np.transpose(boundaries, axes=(0, 2, 1)), lonlat=True)

        # resize boundaries into lon/lat arrays of expected format
        boundaries_lon, boundaries_lat = np.reshape(boundaries, (2, len(idx), 4 * step))

        # loop back to starting point if requested
        # shape: n_pixels x (4*step + 1)
        if loop:
            boundaries_lon = np.hstack([boundaries_lon, boundaries_lon[:, 0:1]])
            boundaries_lat = np.hstack([boundaries_lat, boundaries_lat[:, 0:1]])

        # apply units
        boundaries_lon *= units.deg
        boundaries_lat *= units.deg

        return boundaries_lon, boundaries_lat

    def get_angular_separations(self, lon, lat):
        """
        For each pixel stored in the grid, calculate the distance from the pixel's
        center to the provided position.

        Arguments
        ---------
        lon, lat : float
            Position from which to calculate angular separation for each pixel

        Returns
        -------
        ang_seps : array of floats
            Angular separations between pixel centers and the requested lon/lat position
        """
        from skyway.math.geometry import angular_separation

        return angular_separation((lon, lat), (self.lon, self.lat))

    def get_source_idx(self, source, time=None, observer=None):
        """
        Determine the pixel of a source's location (sun, moon, etc.) at a specified time

        Arguments
        ---------
        src : str
            Source name. Options: "moon", "sun", "zenith"
        time : float [None]
            Time (Unix timestamp, in UTC) at which to determine position. Default: now.
        observer : ephem.Observer [None]
            Observer object. If not provided, defaults to Blanco observer at chosen time

        Returns
        -------
        idx : int
            Index of pixel the source is in. Returns None for pos outside the grid
        """

        # get the source position
        lon, lat = get_source_ra_dec(source=source, time=time, observer=observer)
        if self.is_azel:
            lon, lat = equatorial_to_topographic(
                ra=lon, dec=lat, time=time, observer=observer
            )

        # get sky bin index
        return self.ang2idx(lon=lon, lat=lat)

    def get_source_angular_separations(self, source, time=None, observer=None):
        """
        For each pixel stored in the grid, calculate the distance from the pixel's
        center to a source's location (sun, moon, etc.) at a specified time

        Arguments
        ---------
        src : str
            Source name. Options: "moon", "sun", "zenith"
        time : float [None]
            Time (Unix timestamp, in UTC) at which to determine position. Default: now.
        observer : ephem.Observer [None]
            Observer object. If not provided, defaults to Blanco observer at chosen time

        Returns
        -------
        ang_seps : array of floats
            Angular separations between pixel centers and the requested lon/lat position
        """

        # get the source position
        lon, lat = get_source_ra_dec(source=source, time=time, observer=observer)
        if self.is_azel:
            lon, lat = equatorial_to_topographic(
                ra=lon, dec=lat, time=time, observer=observer
            )

        # get sky bin distances
        return self.get_angular_separations(lon=lon, lat=lat)

    def get_airmass(self, time=None, observer=None):
        """
        For each pixel stored in the grid, calculate the airmass.

        Arguments
        ---------
        time : float [None]
            Time (Unix timestamp, in UTC) at which to determine position. Default: now.
            Ignored if is_azel is True.
        observer : ephem.Observer [None]
            Observer object. If not provided, default to Blanco observer at chosen time.
            Ignored if is_azel is True.

        Returns
        -------
        airmass : array of floats
            Airmass for each pixel. Pixels at or below the horizon default to np.inf
        """
        # get elevation of each pixel
        if self.is_azel:
            el = self.lat
        else:
            az, el = equatorial_to_topographic(
                ra=self.lon, dec=self.lat, time=time, observer=observer
            )

        # calculate airmass
        airmass = np.ones_like(el)
        airmass[el <= 0] = np.inf
        airmass[el > 0] = 1 / np.cos(90 * units.deg - el[el > 0])

        return airmass

    def get_hour_angle(self, time=None, observer=None):
        """
        For each pixel stored in the grid, calculate the hour angle.

        Arguments
        ---------
        time : float [None]
            Time (Unix timestamp, in UTC) at which to determine position. Default: now.
        observer : ephem.Observer [None]
            Observer object. If not provided, default to Blanco observer at chosen time.

        Returns
        -------
        hour_angle : array of floats
            Hour angle for each pixel.
        """

        # get equatorial coordinates of each pixel
        if self.is_azel:
            ra, dec = topographic_to_equatorial(
                az=self.lon, el=self.lat, time=time, observer=observer
            )
        else:
            ra, dec = self.lon, self.lat

        # calculate hour angle
        hour_angle = equatorial_to_hour_angle(
            ra=ra, dec=dec, time=time, observer=observer
        )

        return hour_angle

    def get_time_until_set(self, time=None, observer=None):
        """
        Time until the sky patch currently in each pixel sets below the horizon.

        For az/el grids, this means:
        - Convert each pixel's current az/el to RA/Dec
        - Compute time until that RA/Dec reaches el=0

        Returns
        -------
        t_until_set : ndarray
            Time until set in seconds.
            np.inf for circumpolar or already-below-horizon pixels.
        """
        OMEGA_EARTH = 2 * np.pi / 86164.0905  # rad/s (sidereal)
        observer = observer if observer is not None else blanco_observer(time=time)

        if self.is_azel:
            ra, dec = topographic_to_equatorial(
                az=self.lon, el=self.lat, time=time, observer=observer
            )
        else:
            ra, dec = self.lon, self.lat

        # Current ha
        ha = equatorial_to_hour_angle(
            ra=ra, dec=dec, time=time, observer=observer
        )

        # Observer latitude
        phi = float(observer.lat)

        # cos(ha) at set
        cos_ha_set = -np.tan(phi) * np.tan(dec)

        # Check for circumpolar objects and calculate hour angle at set
        mask = np.abs(cos_ha_set) <= 1
        ha_set = np.arccos(cos_ha_set[mask])

        # ha difference between current ha and ha_set (correcting for wrapping)
        dH = (ha_set - ha[mask] + 2 * np.pi) % (2 * np.pi)

        # Convert to seconds
        t_until_set = np.full_like(dec, np.inf, dtype=float)
        t_until_set[mask] = dH / OMEGA_EARTH

        # Already below horizon → infinite
        if self.is_azel:
            below_horizon = self.lat <= 0
        else:
            _, el = equatorial_to_topographic(
                ra=ra, dec=dec, time=time, observer=observer
            )
            below_horizon = el <= 0
        t_until_set[below_horizon] = np.inf
        return t_until_set