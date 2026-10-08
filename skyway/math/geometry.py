import numpy as np
from skyway.math import units


def angular_separation(pos1, pos2):
    """
    Calculate the angular distance between two points on a sphere using the special case
    of the Vincenty formula for a sphere.

    Arguments
    ---------
    pos1, pos2 : float tuple
        The (horizontal, vertical) positions of the two points on the sphere, provided
        in radians. For example: (ra, dec), (lat, lon), (az, el), etc.

    Returns
    -------
    distance : float
        The angular distance in radians between the two specified points.
    """

    # extract az and el from tuples
    az1, el1 = pos1
    az2, el2 = pos2

    # cast to arrays if not already
    scalar = (
        np.isscalar(az1) and np.isscalar(el1) and np.isscalar(az2) and np.isscalar(el2)
    )
    if not scalar:
        az1 = np.asarray(np.atleast_1d(az1), dtype=float)
        el1 = np.asarray(np.atleast_1d(el1), dtype=float)
        az2 = np.asarray(np.atleast_1d(az2), dtype=float)
        el2 = np.asarray(np.atleast_1d(el2), dtype=float)

    # precalculate repeated terms
    daz = np.abs(az2 - az1)
    cos_daz = np.cos(daz)
    cos_el1 = np.cos(el1)
    sin_el1 = np.sin(el1)
    cos_el2 = np.cos(el2)
    sin_el2 = np.sin(el2)

    # use Vincenty formula
    distance = np.arctan2(
        np.sqrt(
            (cos_el2 * np.sin(daz)) ** 2
            + (cos_el1 * sin_el2 - sin_el1 * cos_el2 * cos_daz) ** 2
        ),
        sin_el1 * sin_el2 + cos_el1 * cos_el2 * cos_daz,
    )

    # return scalar if inputs were scalar, otherwise return array of distances
    if scalar:
        return distance.item()
    else:
        return distance


def blanco_slew_time(distance):
    """
    Calculate the approximate time to slew some angular distance in between observations
    with the Blanco telescope, approximated from historic data.

    Arguments
    ---------
    distance : float
        The angular distance on the sky between two fields.

    Returns
    -------
    time : float
        The overhead + slew time in between observations separated by distance.
    """

    # hardcoded linear fit parameters
    rate = 2.3342091388075774 * (units.sec / units.deg)
    intercept = 27.20906999686867 * units.sec

    return rate * distance + intercept
