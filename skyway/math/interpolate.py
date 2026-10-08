import numpy as np
from scipy.interpolate import RBFInterpolator


def interpolate_on_sphere(az, el, az_data, el_data, values, kernel="quintic", **kwargs):
    """
    Function to interpolate between values sampled on a sphere. Uses
    scipy.RBFInterpolator to interpolate over the surface over a sphere, chosen to
    ensure that interpolated values approach the provided data points near the supplied
    data coordinates.

    Arguments:
    az, el : arrays of floats
        the azimuths and elevations at which to interpolate new values
    az_data, el_data : arrays of floats:
        the azimuths and elevations of the provided data points to be interpolated
    values : array of floats:
        the values at coords az_data, el_data to be interpolated
    kernel : str
        the type of scipy.RBFInterpolator kernel to use
    kwargs
        other keyword arguments to be passed to scipy.RBFInterpolator

    Returns:
    array
        Interpolated values at the requested coordinates az, el
    """

    # helper function to convert lat/lon-like coords to xyz on a 3d-sphere with r=1
    def sph_to_xyz(lon, lat):
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.column_stack((x, y, z))

    # convert coordinates to cartesian
    xyz_data = sph_to_xyz(az_data, el_data)
    xyz_pred = sph_to_xyz(az, el)

    # initialize interpolator
    rbf = RBFInterpolator(xyz_data, values, kernel=kernel, **kwargs)

    # interpolate on requested values
    return rbf(xyz_pred)
