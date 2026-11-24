
"""
Main pipeline for gravitational-lensing mass-bias calculations.

Supported lens models:
    1. Point-mass lens
    2. NFW lens

The pipeline connects the lens models to the mass-bias calculation.
"""

import numpy as np

from .point_mass import (
    einstein_radius,
    point_mass_magnification,
)

from .nfw import (
    solve_images_nfw,
    mu_nfw,
    sigma_crit,
)

from .cosmology import (
    luminosity_distance,
    angular_diameter_distance,
)

from .constants import Mpc_to_m

from .bias import calculate_mass_bias


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_redshifts(z_L, z_S):
    """Validate lens and source redshifts."""

    if not np.isfinite(z_L) or not np.isfinite(z_S):
        raise ValueError("Redshifts must be finite.")

    if not 0 <= z_L < z_S:
        raise ValueError("Require 0 <= z_L < z_S.")


def _validate_positive(value, name):
    """Check that a parameter is finite and strictly positive."""

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


# ============================================================
# POINT-MASS LENS PIPELINE
# ============================================================

def run_point_mass_pipeline(
    beta,
    M_L,
    z_L,
    z_S,
    M_true,
    theta_E=None,
):
    """
    Complete mass-bias calculation for a point-mass lens.

    Parameters
    ----------
    beta : float
        Angular source position in radians.

    M_L : float
        Lens mass in kg.

    z_L : float
        Lens redshift.

    z_S : float
        Source redshift.

    M_true : float
        True source-frame mass.

    theta_E : float, optional
        Angular Einstein radius in radians.
        If not provided, it is calculated automatically.

    Returns
    -------
    dict
        Einstein radius, impact parameter, magnification,
        and apparent mass-bias results.
    """

    # Input validation
    _validate_redshifts(z_L, z_S)
    _validate_positive(beta, "Source position beta")
    _validate_positive(M_L, "Lens mass")
    _validate_positive(M_true, "True source mass")

    # Step 1: Calculate angular Einstein radius
    if theta_E is None:
        theta_E = einstein_radius(M_L, z_L, z_S)

    _validate_positive(theta_E, "Einstein radius")

    # Step 2: Calculate dimensionless impact parameter
    y = beta / theta_E

    # Step 3: Calculate total magnification
    mu = float(point_mass_magnification(y))

    _validate_positive(mu, "Magnification")

    # Step 4: Calculate true luminosity distance
    d_L_true = luminosity_distance(z_S)

    # Step 5: Calculate apparent distance, redshift and mass
    bias_results = calculate_mass_bias(
        d_L_true=d_L_true,
        mu=mu,
        M_true=M_true,
        z_true=z_S,
    )

    # Step 6: Combine results
    return {
        "Lens model": "Point Mass",
        "Source position (rad)": beta,
        "Einstein radius (rad)": theta_E,
        "Impact parameter (y)": y,
        "Total magnification": mu,
        **bias_results,
    }


# ============================================================
# NFW LENS PIPELINE
# ============================================================

def run_nfw_pipeline(
    beta,
    rho_s,
    r_s,
    z_L,
    z_S,
    M_true,
    theta_max,
):
    """
    Complete mass-bias calculation for an NFW lens.

    Parameters
    ----------
    beta : float
        Angular source position in radians.

    rho_s : float
        Characteristic density in kg/m^3.

    r_s : float
        NFW scale radius in metres.

    z_L : float
        Lens redshift.

    z_S : float
        Source redshift.

    M_true : float
        True source-frame mass.

    theta_max : float
        Image-search boundary in radians.

    Returns
    -------
    dict
        Image positions, individual magnifications,
        total magnification, and mass-bias results.
    """

    # Input validation
    _validate_redshifts(z_L, z_S)
    _validate_positive(beta, "Source position beta")
    _validate_positive(rho_s, "Characteristic density")
    _validate_positive(r_s, "Scale radius")
    _validate_positive(M_true, "True source mass")
    _validate_positive(theta_max, "Image-search boundary")

    # Step 1: Calculate critical surface density
    sigma_c = sigma_crit(z_L, z_S)

    # Step 2: Calculate angular diameter distance to the lens
    D_lens = angular_diameter_distance(z_L) * Mpc_to_m

    # Step 3: Solve the NFW lens equation
    images = solve_images_nfw(
        beta,
        rho_s,
        r_s,
        z_L,
        z_S,
        theta_max,
    )

    if len(images) == 0:
        raise RuntimeError(
            "No images found. Check the NFW parameters "
            "and image-search interval."
        )

    # Step 4: Calculate magnification of each image
    magnifications = []

    for theta in images:

        # Physical impact radius in the lens plane
        R = D_lens * abs(theta)

        # Dimensionless radial coordinate
        x = R / r_s

        # Magnification of this image
        mu = mu_nfw(
            x,
            rho_s,
            r_s,
            sigma_c,
        )

        magnifications.append(abs(float(mu)))

    # Step 5: Calculate total magnification
    mu_total = sum(magnifications)

    _validate_positive(mu_total, "Total magnification")

    # Step 6: Calculate true luminosity distance
    d_L_true = luminosity_distance(z_S)

    # Step 7: Calculate mass bias
    bias_results = calculate_mass_bias(
        d_L_true=d_L_true,
        mu=mu_total,
        M_true=M_true,
        z_true=z_S,
    )

    # Step 8: Combine results
    return {
        "Lens model": "NFW",
        "Source position (rad)": beta,
        "Number of images": len(images),
        "Image positions (rad)": images,
        "Individual magnifications": magnifications,
        "Total magnification": mu_total,
        **bias_results,
    }


# ============================================================
# PARAMETER STUDY
# ============================================================

def run_parameter_study(
    model,
    beta_values,
    M_true,
    z_L,
    z_S,
    M_L=None,
    rho_s=None,
    r_s=None,
    theta_max=None,
):
    """
    Run a selected lens model over multiple source positions.

    Parameters
    ----------
    model : str
        'point_mass' or 'nfw'.

    beta_values : array-like
        Array of source positions in radians.

    M_true : float
        True source-frame mass.

    z_L : float
        Lens redshift.

    z_S : float
        Source redshift.

    M_L : float, optional
        Point-mass lens mass in kg.

    rho_s : float, optional
        NFW characteristic density in kg/m^3.

    r_s : float, optional
        NFW scale radius in metres.

    theta_max : float, optional
        NFW image-search boundary in radians.

    Returns
    -------
    list of dict
        Results for each source position.
    """

    # Validate model selection
    model = model.lower().strip()

    if model not in ("point_mass", "nfw"):
        raise ValueError(
            "Model must be 'point_mass' or 'nfw'."
        )

    # Validate common parameters
    _validate_redshifts(z_L, z_S)
    _validate_positive(M_true, "True source mass")

    # Validate source-position array
    beta_values = np.asarray(beta_values, dtype=float)

    if beta_values.ndim != 1:
        raise ValueError("beta_values must be a one-dimensional array.")

    if beta_values.size == 0:
        raise ValueError("beta_values cannot be empty.")

    if not np.all(np.isfinite(beta_values)):
        raise ValueError("All source positions must be finite.")

    if np.any(beta_values <= 0):
        raise ValueError("All source positions must be positive.")

    # Validate model-specific parameters
    if model == "point_mass":

        if M_L is None:
            raise ValueError(
                "Provide M_L for a point-mass lens."
            )

        _validate_positive(M_L, "Lens mass")

    elif model == "nfw":

        if any(v is None for v in (rho_s, r_s, theta_max)):
            raise ValueError(
                "Provide rho_s, r_s and theta_max for an NFW lens."
            )

        _validate_positive(rho_s, "Characteristic density")
        _validate_positive(r_s, "Scale radius")
        _validate_positive(theta_max, "Image-search boundary")

    # Run the selected model for each source position
    results = []

    for beta in beta_values:

        if model == "point_mass":

            result = run_point_mass_pipeline(
                beta=beta,
                M_L=M_L,
                z_L=z_L,
                z_S=z_S,
                M_true=M_true,
            )

        else:

            result = run_nfw_pipeline(
                beta=beta,
                rho_s=rho_s,
                r_s=r_s,
                z_L=z_L,
                z_S=z_S,
                M_true=M_true,
                theta_max=theta_max,
            )

        results.append(result)

    return results
