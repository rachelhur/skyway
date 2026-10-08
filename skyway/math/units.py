import numpy as np


# ======================================================================================
# unit conversion constants
# intended use:
# - skyway assumes native units throughout code (for instance, all angles are rad)
# - multiply by units to convert to the native units (ex: my_ang = 30 * units.deg)
# - divide by units to convert from native unit (ex: my ang_in_deg = my_ang / units.deg)
# ======================================================================================

# angle units
rad = 1.0
radian = rad
deg = np.pi / 180.0
degree = deg
arcmin = deg / 60.0
arcsec = arcmin / 60.0

# time units
sec = 1.0
second = sec
min = 60.0 * sec
minute = min
hr = 60.0 * min
hour = hr
day = 24.0 * hr
yr = 365.25 * day
year = yr

# brightness units
mag = 1.0

# ======================================================================================
# conversion functions
# intended use: for particularly annoying conversions not handled by a simple constant
# ======================================================================================


def rad_to_hms(ang):
    """
    Convert angle in radians to hours, minutes, seconds.

    Arguments
    ---------
    ang : float
        Angle in radians

    Returns
    -------
    hours, minutes, seconds : int
        Hours, minutes, and seconds components of angle
    """

    ang = np.mod(ang, 2.0 * np.pi)  # wrap to [0, 2pi)
    hours = ang / (15.0 * deg)
    minutes = (hours - int(hours)) * 60.0
    seconds = (minutes - int(minutes)) * 60.0
    return int(hours), int(minutes), seconds


def rad_to_dms(ang):
    """
    Convert angle in radians to degrees, arcminutes, arcseconds.

    Arguments
    ---------
    ang : float
        Angle in radians

    Returns
    -------
    degrees, arcminutes, arcseconds : int
        Degrees, arcminutes, and arcseconds components of angle
    """

    sign = 1
    if ang < 0.0:
        sign = -1
    ang = np.abs(ang)

    ang = np.mod(ang, 2.0 * np.pi)  # wrap to [0, 2pi)
    degrees = ang / deg
    arcminutes = (degrees - int(degrees)) * 60.0
    arcseconds = (arcminutes - int(arcminutes)) * 60.0
    return sign * int(degrees), int(arcminutes), arcseconds


def hms_to_rad(hours, minutes, seconds):
    """
    Convert hours, minutes, seconds to angle in radians.

    Arguments
    ---------
    hours : int
        Hours component of angle
    minutes : int
        Minutes component of angle
    seconds : float
        Seconds component of angle

    Returns
    -------
    ang : float
        Angle in radians
    """

    ang = (hours + minutes / 60.0 + seconds / 3600.0) * (15.0 * deg)
    return ang


def dms_to_rad(degrees, arcminutes, arcseconds):
    """
    Convert degrees, arcminutes, arcseconds to angle in radians.

    Arguments
    ---------
    degrees : int
        Degrees component of angle
    arcminutes : int
        Arcminutes component of angle
    arcseconds : float
        Arcseconds component of angle

    Returns
    -------
    ang : float
        Angle in radians
    """

    sign = 1
    if degrees < 0:
        sign = -1
    degrees = np.abs(degrees)

    ang = sign * (degrees + arcminutes / 60.0 + arcseconds / 3600.0) * deg
    return ang
