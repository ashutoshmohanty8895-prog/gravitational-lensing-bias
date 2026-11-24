import numpy as np
from .constants import G, c, Mpc_to_m
from .cosmology import angular_diameter_distance, adm_diff



def point_mass_magnification(y):
    """
    Calculate the total magnification produced by a point-mass lens.

    Parameters
    ----------
    y : float or array-like
        Source-lens angular separation in units of the Einstein radius.
        Must be strictly positive.

    Returns
    -------
    float or numpy.ndarray
        Magnification corresponding to the input.
    """

    y = np.asarray(y, dtype=float)

    if np.any(y <= 0):
        raise ValueError("Source-lens separation y must be positive.")

    mu = (y**2 + 2) / (y * np.sqrt(y**2 + 4))

    return mu

from .constants import G, c, Mpc_to_m
from .cosmology import angular_diameter_distance, adm_diff


def einstein_radius(M_L, z_L, z_S):
    """
    Calculate the angular Einstein radius of a point-mass lens.

    Parameters
    ----------
    M_L : float
        Lens mass in kg.
    z_L : float
        Lens redshift.
    z_S : float
        Source redshift.

    Returns
    -------
    float
        Angular Einstein radius in radians.
    """

    if M_L <= 0:
        raise ValueError("Lens mass must be positive.")

    if z_L < 0 or z_S <= z_L:
        raise ValueError("Require 0 <= z_L < z_S.")

    D_L = angular_diameter_distance(z_L) * Mpc_to_m
    D_S = angular_diameter_distance(z_S) * Mpc_to_m
    D_LS = adm_diff(z_L, z_S) * Mpc_to_m

    theta_E = np.sqrt(
        (4 * G * M_L / c**2) * (D_LS / (D_L * D_S))
    )

    return theta_E