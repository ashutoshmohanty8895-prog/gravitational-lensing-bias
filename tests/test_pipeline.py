import numpy as np

from src.pipeline import (
    run_point_mass_pipeline,
    run_parameter_study,
)


# ============================================================
# LENS AND SOURCE PARAMETERS
# ============================================================

M_L = 1e12 * 1.98847e30       # Lens mass in kg
M_true = 30.0                  # True source-frame mass in solar masses

z_L = 0.5                      # Lens redshift
z_S = 1.0                      # Source redshift

beta = np.deg2rad(0.5 / 3600)  # Source position: 0.5 arcsec


# ============================================================
# PART 1: SINGLE POINT-MASS CALCULATION
# ============================================================

results = run_point_mass_pipeline(
    M_L=M_L,
    z_L=z_L,
    z_S=z_S,
    beta=beta,
    M_true=M_true,
)


print("\n" + "=" * 60)
print("SINGLE POINT-MASS LENSING CALCULATION")
print("=" * 60)

for key, value in results.items():

    print(f"{key}: {value}")


# ============================================================
# BASIC PHYSICAL CHECKS
# ============================================================

assert results["Einstein radius (rad)"] > 0
assert results["Impact parameter (y)"] > 0
assert results["Total magnification"] >= 1
assert results["Apparent luminosity distance"] > 0
assert results["Apparent redshift"].value >= 0

print("\nAll basic physical checks passed.")


# ============================================================
# PART 2: PARAMETER STUDY
# ============================================================

beta_values = np.deg2rad(
    np.array([0.1, 0.25, 0.5, 0.75, 1.0]) / 3600
)

study_results = run_parameter_study(
    model="point_mass",
    beta_values=beta_values,
    M_L=M_L,
    M_true=M_true,
    z_L=z_L,
    z_S=z_S,
)


print("\n" + "=" * 60)
print("POINT-MASS PARAMETER STUDY")
print("=" * 60)

for result in study_results:

    print("\n" + "-" * 50)

    for key, value in result.items():

        print(f"{key}: {value}")


print("\nParameter study completed successfully.")
print("=" * 60)