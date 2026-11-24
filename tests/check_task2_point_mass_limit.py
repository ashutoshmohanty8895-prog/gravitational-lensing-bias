"""
TASK 2: NFW -> POINT-MASS LIMIT

Keep M200 fixed and increase NFW concentration.
Compare NFW total magnification with the analytical
point-mass total magnification.

All internal angular calculations are in radians.
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import M_sun

from src.nfw import (
    solve_images_nfw,
    total_magnification_nfw
)

from src.point_mass import (
    einstein_radius,
    point_mass_magnification
)


# ============================================================
# 1. PHYSICAL SETUP
# ============================================================

M200_solar = 1.0e12
M200 = M200_solar * M_sun.value

z_L = 0.5
z_S = 1.0

concentrations = [5, 10, 20, 40, 80]

source_positions_arcsec = [0.10, 0.25, 0.50]

# Maximum angular position searched by the NFW image solver.
theta_max = np.deg2rad(10.0 / 3600.0)

# Grid resolution for locating image roots.
n_grid = 1001

cosmo = Planck18


# ============================================================
# 2. CALCULATE THE FIXED VIRIAL RADIUS
# ============================================================

rho_crit = cosmo.critical_density(z_L).to_value(
    u.kg / u.m**3
)

r200 = (
    3.0 * M200
    / (4.0 * np.pi * 200.0 * rho_crit)
) ** (1.0 / 3.0)


# ============================================================
# 3. POINT-MASS REFERENCE
# ============================================================

theta_E_point = einstein_radius(
    M200,
    z_L,
    z_S
)


# ============================================================
# 4. STORAGE
# ============================================================

results = {}

print("=" * 65)
print("TASK 2: NFW TO POINT-MASS LIMIT")
print("=" * 65)

print(f"Halo mass M200: {M200_solar:.3e} solar masses")
print(f"Lens redshift: {z_L}")
print(f"Source redshift: {z_S}")
print(f"Virial radius r200: {r200 / 3.086e19:.4f} kpc")
print(f"Point-mass Einstein radius: "
      f"{np.rad2deg(theta_E_point) * 3600:.6f} arcsec")


# ============================================================
# 5. CONCENTRATION SCAN
# ============================================================

for beta_arcsec in source_positions_arcsec:

    print("\n" + "=" * 65)
    print(f"SOURCE POSITION: {beta_arcsec:.3f} arcsec")
    print("=" * 65)

    beta = np.deg2rad(beta_arcsec / 3600.0)

    # Analytical point-mass magnification
    y = beta / theta_E_point

    mu_point = float(point_mass_magnification(y))

    results[beta_arcsec] = {
        "point_mass": mu_point,
        "nfw": []
    }

    print(f"Point-mass magnification: {mu_point:.8f}")

    for concentration in concentrations:

        # ----------------------------------------------------
        # NFW characteristic density
        # ----------------------------------------------------

        f_c = (
            np.log(1.0 + concentration)
            - concentration / (1.0 + concentration)
        )

        delta_c = (
            (200.0 / 3.0)
            * concentration**3
            / f_c
        )

        rho_s = delta_c * rho_crit

        # ----------------------------------------------------
        # NFW scale radius
        # ----------------------------------------------------

        r_s = r200 / concentration

        # ----------------------------------------------------
        # Solve the NFW lens equation
        # ----------------------------------------------------

        print(
            f"\nConcentration = {concentration}: "
            "solving lens equation...",
            flush=True
        )

        images = solve_images_nfw(
            beta=beta,
            rho_s=rho_s,
            r_s=r_s,
            z_L=z_L,
            z_S=z_S,
            theta_max=theta_max,
            n_grid=n_grid
        )

        if len(images) == 0:

            print("No images found. Check solver resolution.")

            mu_nfw = np.nan

        else:

            mu_nfw = float(
                total_magnification_nfw(
                    images,
                    rho_s,
                    r_s,
                    z_L,
                    z_S
                )
            )

        # ----------------------------------------------------
        # Relative difference from point-mass result
        # ----------------------------------------------------

        if np.isfinite(mu_nfw):

            relative_difference = (
                abs(mu_nfw - mu_point) / mu_point
            ) * 100.0

        else:

            relative_difference = np.nan

        results[beta_arcsec]["nfw"].append(mu_nfw)

        print(f"Scale radius: {r_s / 3.086e19:.6f} kpc")
        print(f"Characteristic density: {rho_s:.6e} kg/m^3")
        print(f"Number of images: {len(images)}")
        print(f"NFW total magnification: {mu_nfw:.8f}")
        print(f"Point-mass magnification: {mu_point:.8f}")
        print(f"Relative difference: {relative_difference:.6f}%")


# ============================================================
# 6. SUMMARY TABLE
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

for beta_arcsec in source_positions_arcsec:

    mu_point = results[beta_arcsec]["point_mass"]

    print(f"\nSource position: {beta_arcsec:.3f} arcsec")
    print(f"Point-mass magnification: {mu_point:.8f}")

    print("-" * 65)
    print(f"{'Concentration':<18}{'NFW magnification':<24}{'Difference (%)'}")
    print("-" * 65)

    for concentration, mu_nfw in zip(
        concentrations,
        results[beta_arcsec]["nfw"]
    ):

        if np.isfinite(mu_nfw):

            difference = (
                abs(mu_nfw - mu_point) / mu_point
            ) * 100.0

            print(
                f"{concentration:<18}"
                f"{mu_nfw:<24.8f}"
                f"{difference:.6f}"
            )

        else:

            print(
                f"{concentration:<18}"
                f"{'Not found':<24}"
                f"{'--'}"
            )


# ============================================================
# 7. PLOT 1: MAGNIFICATION VS CONCENTRATION
# ============================================================

fig1, ax1 = plt.subplots(figsize=(9, 6))

for beta_arcsec in source_positions_arcsec:

    mu_values = results[beta_arcsec]["nfw"]

    ax1.plot(
        concentrations,
        mu_values,
        marker="o",
        label=f"NFW: beta = {beta_arcsec} arcsec"
    )

    ax1.axhline(
        results[beta_arcsec]["point_mass"],
        linestyle="--",
        label=f"Point mass: beta = {beta_arcsec} arcsec"
    )

ax1.set_xscale("log")

ax1.set_xlabel("NFW concentration")
ax1.set_ylabel("Total magnification")

ax1.set_title(
    "NFW Magnification vs Concentration"
)

ax1.grid(True, alpha=0.3)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    "figures/task2_magnification_vs_concentration.png",
    dpi=300
)


# ============================================================
# 8. PLOT 2: RELATIVE DIFFERENCE
# ============================================================

fig2, ax2 = plt.subplots(figsize=(9, 6))

for beta_arcsec in source_positions_arcsec:

    mu_point = results[beta_arcsec]["point_mass"]

    mu_values = np.asarray(
        results[beta_arcsec]["nfw"],
        dtype=float
    )

    difference = (
        np.abs(mu_values - mu_point) / mu_point
    ) * 100.0

    ax2.plot(
        concentrations,
        difference,
        marker="o",
        label=f"beta = {beta_arcsec} arcsec"
    )

ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_xlabel("NFW concentration")
ax2.set_ylabel("Relative difference (%)")

ax2.set_title(
    "Deviation of NFW Magnification from Point-Mass Magnification"
)

ax2.grid(True, alpha=0.3)
ax2.legend()

fig2.tight_layout()

fig2.savefig(
    "figures/task2_relative_difference.png",
    dpi=300
)


plt.show()


print("\n" + "=" * 65)
print("TASK 2 COMPLETED")
print("=" * 65)

print("Figures saved:")
print("figures/task2_magnification_vs_concentration.png")
print("figures/task2_relative_difference.png")