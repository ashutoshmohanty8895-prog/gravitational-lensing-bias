import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nfw import (
    solve_images_nfw,
    lens_equation_nfw,
    mu_nfw,
    sigma_crit,
)

from src.cosmology import angular_diameter_distance
from src.constants import Mpc_to_m


# ============================================================
# 1. LENS PARAMETERS
# ============================================================

rho_s = 1e-19       # Characteristic density (kg/m^3)
r_s = 1e20          # Scale radius (m)

z_L = 0.5           # Lens redshift
z_S = 1.0           # Source redshift

# Representative source positions (arcseconds)
beta_values_arcsec = np.array([
    0.1, 0.25, 0.5, 0.75, 1.0
])

# Convert source positions into radians
beta_values = np.deg2rad(beta_values_arcsec / 3600)

# Search interval: -20 to +20 arcseconds
theta_max = np.deg2rad(20 / 3600)


# ============================================================
# 2. COSMOLOGICAL DISTANCES
# ============================================================

sigma_c = sigma_crit(z_L, z_S)

D_L = angular_diameter_distance(z_L) * Mpc_to_m

print("=" * 65)
print("NFW LENS VALIDATION")
print("=" * 65)

print(f"Lens redshift: {z_L}")
print(f"Source redshift: {z_S}")
print(f"Characteristic density: {rho_s:.4e} kg/m^3")
print(f"Scale radius: {r_s:.4e} m")
print(f"Critical surface density: {sigma_c:.6e}")
print(f"Lens angular diameter distance: {D_L:.6e} m")


# ============================================================
# 3. INDEPENDENT MAGNIFICATION CALCULATION
# ============================================================

def independent_magnification(
    theta,
    beta,
    rho_s,
    r_s,
    z_L,
    z_S
):
    """
    Calculate signed magnification independently
    using numerical differentiation of the lens mapping.

    mu = 1 / (lambda_t * lambda_r)

    lambda_t = beta(theta) / theta

    lambda_r = d(beta(theta)) / d(theta)
    """

    if theta == 0:
        raise ValueError(
            "Image position theta cannot be zero."
        )

    # Numerical differentiation step
    h = 1e-5 * max(abs(theta), 1e-8)

    # Reconstruct the lens mapping:
    # beta(theta) = beta_source + lens_equation_residual

    beta_plus = beta + lens_equation_nfw(
        theta + h,
        beta,
        rho_s,
        r_s,
        z_L,
        z_S
    )

    beta_minus = beta + lens_equation_nfw(
        theta - h,
        beta,
        rho_s,
        r_s,
        z_L,
        z_S
    )

    beta_theta = beta + lens_equation_nfw(
        theta,
        beta,
        rho_s,
        r_s,
        z_L,
        z_S
    )

    # Radial eigenvalue
    lambda_r = (beta_plus - beta_minus) / (2 * h)

    # Tangential eigenvalue
    lambda_t = beta_theta / theta

    if lambda_t == 0 or lambda_r == 0:
        raise ZeroDivisionError(
            "Magnification is singular at this image position."
        )

    return 1.0 / (lambda_t * lambda_r)


# ============================================================
# 4. IMAGE POSITIONS AND MAGNIFICATION VALIDATION
# ============================================================

print("\n\n")
print("=" * 65)
print("IMAGE POSITION AND MAGNIFICATION VALIDATION")
print("=" * 65)

for beta, beta_arcsec in zip(
    beta_values,
    beta_values_arcsec
):

    print("\n" + "=" * 60)
    print(f"Source position: {beta_arcsec:.3f} arcsec")
    print("=" * 60)

    # Find image positions
    images = solve_images_nfw(
        beta,
        rho_s,
        r_s,
        z_L,
        z_S,
        theta_max
    )

    magnifications = []

    print("\nImage positions and magnification checks:")

    for theta in images:

        # Lens equation residual
        residual = lens_equation_nfw(
            theta,
            beta,
            rho_s,
            r_s,
            z_L,
            z_S
        )

        # Dimensionless projected radius
        R = D_L * abs(theta)
        x = R / r_s

        # Analytical magnification
        mu_analytical = float(
            mu_nfw(x, rho_s, r_s, sigma_c)
        )

        # Independent numerical magnification
        mu_independent = independent_magnification(
            theta,
            beta,
            rho_s,
            r_s,
            z_L,
            z_S
        )

        # Compare absolute magnifications
        mu_analytical_abs = abs(mu_analytical)
        mu_independent_abs = abs(mu_independent)

        relative_difference = (
            abs(mu_analytical_abs - mu_independent_abs)
            / max(mu_analytical_abs, 1e-30)
        )

        magnifications.append(mu_analytical_abs)

        theta_arcsec = np.rad2deg(theta) * 3600

        print(f"\nImage position: {theta_arcsec:.6f} arcsec")
        print(f"Lens equation residual: {residual:.3e}")
        print(f"Analytical |mu|: {mu_analytical_abs:.8f}")
        print(f"Independent |mu|: {mu_independent_abs:.8f}")
        print(f"Relative difference: {relative_difference:.3e}")

    # Total magnification
    total_mu = sum(magnifications)

    print("\nNumber of images:", len(images))
    print(f"Total magnification: {total_mu:.8f}")


# ============================================================
# 5. IMAGE COUNT AND MAGNIFICATION SCAN
# ============================================================

print("\n\n")
print("=" * 65)
print("SCANNING SOURCE POSITIONS")
print("=" * 65)

# Source positions from 0.01 to 1.2 arcseconds
beta_scan_arcsec = np.linspace(0.01, 1.2, 150)

image_count_scan = []
total_magnification_scan = []
image_positions_scan = []

for beta_arcsec in beta_scan_arcsec:

    beta = np.deg2rad(beta_arcsec / 3600)

    # Solve the lens equation
    images = solve_images_nfw(
        beta,
        rho_s,
        r_s,
        z_L,
        z_S,
        theta_max
    )

    # Store number of images
    image_count_scan.append(len(images))

    # Store image positions in arcseconds
    image_positions_arcsec = (
        np.rad2deg(np.asarray(images)) * 3600
    )

    image_positions_scan.append(image_positions_arcsec)

    # Calculate total magnification
    total_mu = 0.0

    for theta in images:

        R = D_L * abs(theta)
        x = R / r_s

        mu = float(
            mu_nfw(x, rho_s, r_s, sigma_c)
        )

        total_mu += abs(mu)

    total_magnification_scan.append(total_mu)


# Convert lists into NumPy arrays
image_count_scan = np.array(image_count_scan)

total_magnification_scan = np.array(
    total_magnification_scan
)

# ============================================================
# 6. FIGURE 1: IMAGE COUNT VS SOURCE POSITION
# ============================================================

fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.step(
    beta_scan_arcsec,
    image_count_scan,
    where="mid",
    linewidth=2
)

ax1.set_xlabel(
    "Source position (arcsec)",
    fontsize=12
)

ax1.set_ylabel(
    "Number of images",
    fontsize=12
)

ax1.set_title(
    "NFW Image Count vs Source Position",
    fontsize=14
)

ax1.set_yticks([1, 2, 3])

ax1.grid(True, alpha=0.4)

fig1.tight_layout()

fig1.savefig(
    PROJECT_ROOT / "figures" / "nfw_image_count_validation.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 7. FIGURE 2: TOTAL MAGNIFICATION VS SOURCE POSITION
# ============================================================

fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.plot(
    beta_scan_arcsec,
    total_magnification_scan,
    linewidth=2,
    label="NFW total magnification"
)

ax2.set_xlabel(
    "Source position (arcsec)",
    fontsize=12
)

ax2.set_ylabel(
    "Total magnification",
    fontsize=12
)

ax2.set_title(
    "NFW Total Magnification vs Source Position",
    fontsize=14
)

ax2.grid(True, alpha=0.4)

ax2.legend()

fig2.tight_layout()

fig2.savefig(
    PROJECT_ROOT / "figures" / "nfw_total_magnification.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 8. FIGURE 3: IMAGE POSITIONS VS SOURCE POSITION
# ============================================================

fig3, ax3 = plt.subplots(figsize=(9, 5))

for beta_arcsec, image_positions in zip(
    beta_scan_arcsec,
    image_positions_scan
):

    for theta_arcsec in image_positions:

        ax3.scatter(
            beta_arcsec,
            theta_arcsec,
            s=8
        )

ax3.set_xlabel(
    "Source position (arcsec)",
    fontsize=12
)

ax3.set_ylabel(
    "Image position (arcsec)",
    fontsize=12
)

ax3.set_title(
    "NFW Image Positions vs Source Position",
    fontsize=14
)

ax3.grid(True, alpha=0.4)

fig3.tight_layout()

fig3.savefig(
    PROJECT_ROOT / "figures" / "nfw_image_positions.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 9. IMAGE-COUNT TRANSITIONS
# ============================================================

print("\n\n")
print("=" * 65)
print("IMAGE-COUNT TRANSITIONS")
print("=" * 65)

for i in range(1, len(image_count_scan)):

    if image_count_scan[i] != image_count_scan[i - 1]:

        print(
            f"Transition near beta = "
            f"{beta_scan_arcsec[i]:.4f} arcsec: "
            f"{image_count_scan[i - 1]} -> "
            f"{image_count_scan[i]} images"
        )


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n\n")
print("=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)

print(
    "Image counts encountered:",
    np.unique(image_count_scan)
)

print(
    "Maximum total magnification:",
    np.max(total_magnification_scan)
)

print(
    "Minimum total magnification:",
    np.min(total_magnification_scan)
)

print("\nFigures saved in:")

print(
    PROJECT_ROOT / "figures" / "nfw_image_count_validation.png"
)

print(
    PROJECT_ROOT / "figures" / "nfw_total_magnification.png"
)

print(
    PROJECT_ROOT / "figures" / "nfw_image_positions.png"
)


# ============================================================
# 11. DISPLAY ALL FIGURES
# ============================================================

plt.show()