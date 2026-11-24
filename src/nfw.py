"""
NFW dark matter halo lensing.

Implements the lensing properties of the Navarro-Frenk-White
(NFW) density profile following Wright & Brainerd (2000).

All projected lensing functions use the dimensionless radius:

    x = R / r_s

Units:
    r, r_s       : metres
    rho_s        : kg / m^3
    Sigma        : kg / m^2
    Sigma_crit   : kg / m^2
    x            : dimensionless
"""

import numpy as np

from .constants import G, c, Mpc_to_m
from .cosmology import angular_diameter_distance, adm_diff
from scipy.optimize import brentq


# ----------------------------------------------------------------------
# Density profile
# ----------------------------------------------------------------------

def rho_nfw(r, rho_s, r_s):
    """
    Calculate the 3D NFW density profile.

    rho(r) = rho_s / [x(1+x)^2]

    where x = r/r_s.

    Parameters
    ----------
    r : float or array-like
        Three-dimensional radius in metres.
    rho_s : float
        Characteristic density in kg/m^3.
    r_s : float
        Scale radius in metres.

    Returns
    -------
    float or numpy.ndarray
        Density in kg/m^3.
    """

    r = np.asarray(r, dtype=float)

    if np.any(r <= 0):
        raise ValueError("Radius r must be positive.")

    if rho_s <= 0:
        raise ValueError("Characteristic density must be positive.")

    if r_s <= 0:
        raise ValueError("Scale radius must be positive.")

    x = r / r_s

    return rho_s / (x * (1.0 + x)**2)


def M_enclosed(r, rho_s, r_s):
    """
    Calculate the mass enclosed within radius r.

    M(<r) = 4*pi*rho_s*r_s^3 *
            [ln(1+x) - x/(1+x)]

    where x = r/r_s.

    Parameters
    ----------
    r : float or array-like
        Three-dimensional radius in metres.
    rho_s : float
        Characteristic density in kg/m^3.
    r_s : float
        Scale radius in metres.

    Returns
    -------
    float or numpy.ndarray
        Enclosed mass in kg.
    """

    r = np.asarray(r, dtype=float)

    if np.any(r < 0):
        raise ValueError("Radius r cannot be negative.")

    if rho_s <= 0:
        raise ValueError("Characteristic density must be positive.")

    if r_s <= 0:
        raise ValueError("Scale radius must be positive.")

    x = r / r_s

    mass = 4.0 * np.pi * rho_s * r_s**3 * (
        np.log1p(x) - x / (1.0 + x)
    )

    return mass


# ----------------------------------------------------------------------
# Projected quantities
# Wright & Brainerd (2000)
# ----------------------------------------------------------------------

def _f(x):
    """
    Dimensionless surface-density function.

    Sigma(x) = 2*rho_s*r_s*f(x)

    Evaluated separately for x < 1, x = 1 and x > 1.
    """

    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Dimensionless radius x must be positive.")

    out = np.empty_like(x)

    # x < 1
    mask = x < 1.0
    xm = x[mask]

    out[mask] = (
        1.0 / (xm**2 - 1.0)
        * (
            1.0
            - 2.0 / np.sqrt(1.0 - xm**2)
            * np.arctanh(
                np.sqrt((1.0 - xm) / (1.0 + xm))
            )
        )
    )

    # x = 1
    out[x == 1.0] = 1.0 / 3.0

    # x > 1
    mask = x > 1.0
    xm = x[mask]

    out[mask] = (
        1.0 / (xm**2 - 1.0)
        * (
            1.0
            - 2.0 / np.sqrt(xm**2 - 1.0)
            * np.arctan(
                np.sqrt((xm - 1.0) / (1.0 + xm))
            )
        )
    )

    return out


def _g(x):
    """
    Dimensionless mean-surface-density function.

    Sigma_bar(x) = 4*rho_s*r_s*g(x)/x^2

    Evaluated separately for x < 1, x = 1 and x > 1.
    """

    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Dimensionless radius x must be positive.")

    out = np.empty_like(x)

    # x < 1
    mask = x < 1.0
    xm = x[mask]

    out[mask] = (
        np.log(xm / 2.0)
        + 2.0 / np.sqrt(1.0 - xm**2)
        * np.arctanh(
            np.sqrt((1.0 - xm) / (1.0 + xm))
        )
    )

    # x = 1
    out[x == 1.0] = 1.0 + np.log(0.5)

    # x > 1
    mask = x > 1.0
    xm = x[mask]

    out[mask] = (
        np.log(xm / 2.0)
        + 2.0 / np.sqrt(xm**2 - 1.0)
        * np.arctan(
            np.sqrt((xm - 1.0) / (1.0 + xm))
        )
    )

    return out


# ----------------------------------------------------------------------
# Surface mass density
# ----------------------------------------------------------------------

def sigma_nfw(x, rho_s, r_s):
    """
    Calculate the projected surface mass density.

    Sigma(x) = 2*rho_s*r_s*f(x)

    Parameters
    ----------
    x : float or array-like
        Dimensionless projected radius R/r_s.
    rho_s : float
        Characteristic density in kg/m^3.
    r_s : float
        Scale radius in metres.

    Returns
    -------
    float or numpy.ndarray
        Surface mass density in kg/m^2.
    """

    if rho_s <= 0 or r_s <= 0:
        raise ValueError("rho_s and r_s must be positive.")

    return 2.0 * rho_s * r_s * _f(x)


def sigma_bar_nfw(x, rho_s, r_s):
    """
    Calculate the mean surface mass density inside radius x.

    Sigma_bar(x) = 4*rho_s*r_s*g(x)/x^2

    Parameters
    ----------
    x : float or array-like
        Dimensionless projected radius R/r_s.
    rho_s : float
        Characteristic density in kg/m^3.
    r_s : float
        Scale radius in metres.

    Returns
    -------
    float or numpy.ndarray
        Mean surface mass density in kg/m^2.
    """

    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Dimensionless radius x must be positive.")

    if rho_s <= 0 or r_s <= 0:
        raise ValueError("rho_s and r_s must be positive.")

    return 4.0 * rho_s * r_s * _g(x) / x**2


# ----------------------------------------------------------------------
# Critical surface mass density
# ----------------------------------------------------------------------

def sigma_crit(z_L, z_S):
    """
    Calculate the critical surface mass density.

    Sigma_crit = c^2/(4*pi*G) * D_S/(D_L*D_LS)

    Parameters
    ----------
    z_L : float
        Lens redshift.
    z_S : float
        Source redshift.

    Returns
    -------
    float
        Critical surface mass density in kg/m^2.
    """

    if z_L < 0 or z_S <= z_L:
        raise ValueError("Require 0 <= z_L < z_S.")

    D_L = angular_diameter_distance(z_L) * Mpc_to_m
    D_S = angular_diameter_distance(z_S) * Mpc_to_m
    D_LS = adm_diff(z_L, z_S) * Mpc_to_m

    if D_L <= 0 or D_S <= 0 or D_LS <= 0:
        raise ValueError("Angular diameter distances must be positive.")

    return (
        c**2 / (4.0 * np.pi * G)
        * D_S / (D_L * D_LS)
    )


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------

def kappa_nfw(x, rho_s, r_s, sigma_c):
    """
    Calculate the convergence.

    kappa = Sigma / Sigma_crit
    """

    if sigma_c <= 0:
        raise ValueError("Critical surface density must be positive.")

    return sigma_nfw(x, rho_s, r_s) / sigma_c


# ----------------------------------------------------------------------
# Shear
# ----------------------------------------------------------------------

def gamma_nfw(x, rho_s, r_s, sigma_c):
    """
    Calculate the tangential shear.

    gamma = (Sigma_bar - Sigma) / Sigma_crit
    """

    if sigma_c <= 0:
        raise ValueError("Critical surface density must be positive.")

    return (
        sigma_bar_nfw(x, rho_s, r_s)
        - sigma_nfw(x, rho_s, r_s)
    ) / sigma_c


# ----------------------------------------------------------------------
# Magnification
# ----------------------------------------------------------------------

def mu_nfw(x, rho_s, r_s, sigma_c):
    """
    Calculate the signed local image magnification.

    mu = 1 / [(1-kappa)^2 - gamma^2]

    Note
    ----
    This is the magnification at an image position.
    It is not necessarily the total magnification of a source.
    Total magnification requires solving the lens equation
    and summing the absolute magnifications of all images.
    """

    kappa = kappa_nfw(x, rho_s, r_s, sigma_c)

    gamma = gamma_nfw(x, rho_s, r_s, sigma_c)

    denominator = (1.0 - kappa)**2 - gamma**2

    return 1.0 / denominator
def M_projected_nfw(x, rho_s, r_s):
    """
    Projected mass enclosed within projected radius R.
    """

    x = np.asarray(x, dtype=float)

    if np.any(x < 0):
        raise ValueError("x cannot be negative.")

    result = np.zeros_like(x, dtype=float)

    positive = x > 0

    if np.any(positive):
        R = x[positive] * r_s

        result[positive] = (
            np.pi * R**2
            * sigma_bar_nfw(x[positive], rho_s, r_s)
        )

    if result.ndim == 0:
        return float(result)

    return result

def alpha_hat_nfw(x, rho_s, r_s):
    """
    Physical deflection angle for an NFW lens.

    Returns angle in radians.
    """

    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("x must be positive.")

    R = x * r_s

    mass = M_projected_nfw(x, rho_s, r_s)

    return 4.0 * G * mass / (c**2 * R)


def lens_equation_nfw(theta, beta, rho_s, r_s, z_L, z_S):

    D_L = angular_diameter_distance(z_L) * Mpc_to_m
    D_S = angular_diameter_distance(z_S) * Mpc_to_m
    D_LS = adm_diff(z_L, z_S) * Mpc_to_m

    if theta == 0:
        return -beta

    R = D_L * abs(theta)
    x = R / r_s

    alpha = alpha_hat_nfw(x, rho_s, r_s)

    return theta - (D_LS / D_S) * np.sign(theta) * alpha - beta

def solve_images_nfw(beta, rho_s, r_s, z_L, z_S,
                     theta_max, n_grid=10000):
    """
    Find image positions by numerically solving the NFW lens equation.

    All angular quantities are in radians.
    """

    if beta <= 0 or theta_max <= 0:
        raise ValueError("beta and theta_max must be positive.")

    theta_values = np.linspace(-theta_max, theta_max, n_grid)

    roots = []

    for i in range(len(theta_values) - 1):

        a = theta_values[i]
        b = theta_values[i + 1]

        fa = lens_equation_nfw(
            a, beta, rho_s, r_s, z_L, z_S
        )

        fb = lens_equation_nfw(
            b, beta, rho_s, r_s, z_L, z_S
        )

        if fa * fb < 0:

            root = brentq(
                lens_equation_nfw,
                a, b,
                args=(beta, rho_s, r_s, z_L, z_S)
            )

            if not roots or all(abs(root - r) > 1e-10 for r in roots):
                roots.append(root)

    return np.array(sorted(roots))

def total_magnification_nfw(images, rho_s, r_s,
                            z_L, z_S):

    D_L = angular_diameter_distance(z_L) * Mpc_to_m

    sigma_c = sigma_crit(z_L, z_S)

    magnifications = []

    for theta in images:

        R = D_L * abs(theta)
        x = R / r_s

        mu = mu_nfw(x, rho_s, r_s, sigma_c)

        magnifications.append(abs(float(mu)))

    return np.sum(magnifications)