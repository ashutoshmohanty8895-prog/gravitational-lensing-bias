import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import G, M_sun
from astropy.cosmology import Planck18

from src.nfw import solve_images_nfw, total_magnification_nfw
from src.point_mass import point_mass_magnification, einstein_radius
from src.bias import calculate_mass_bias


# --------------------------------------------------
# 1. Cosmological and lens parameters
# --------------------------------------------------

z_L = 0.5
z_S = 1.0

M_200 = 1e12 * M_sun.value       # Solar halo mass in kg
concentration = 10

# Critical density at lens redshift
rho_crit = Planck18.critical_density(z_L).to_value(
    u.kg / u.m**3
)

# Virial radius
r_200 = (
    3 * M_200 /
    (4 * np.pi * 200 * rho_crit)
)**(1/3)

# NFW scale radius
r_s = r_200 / concentration

# Characteristic density
delta_c = (
    (200 / 3) * concentration**3 /
    (
        np.log(1 + concentration)
        - concentration / (1 + concentration)
    )
)

rho_s = delta_c * rho_crit


# --------------------------------------------------
# 2. Point-mass Einstein radius
# --------------------------------------------------

theta_E = einstein_radius(M_200, z_L, z_S)


# --------------------------------------------------
# 3. Source positions
# --------------------------------------------------

y_values = np.linspace(0.1, 5, 20)

mu_point = []
mu_nfw = []

theta_max = np.deg2rad(30 / 3600)  # 30 arcseconds


# --------------------------------------------------
# 4. Calculate magnifications
# --------------------------------------------------
image_counts = []
for y in y_values:

    # Point-mass magnification
    mu_pm = point_mass_magnification(y)
    mu_point.append(float(mu_pm))

    # Convert dimensionless source position
    # into angular source position
    beta = y * theta_E

    # Solve NFW lens equation
    images = solve_images_nfw(
        beta=beta,
        rho_s=rho_s,
        r_s=r_s,
        z_L=z_L,
        z_S=z_S,
        theta_max=theta_max
    )
    image_counts.append(len(images))

    # Total NFW magnification
    if len(images) > 0:

        mu = total_magnification_nfw(
            images,
            rho_s,
            r_s,
            z_L,
            z_S
        )

        mu_nfw.append(float(mu))

    else:
        mu_nfw.append(np.nan)


mu_point = np.array(mu_point)
mu_nfw = np.array(mu_nfw)
print("Number of NFW images:", image_counts)

# --------------------------------------------------
# 5. Calculate mass bias
# --------------------------------------------------

d_L_true = Planck18.luminosity_distance(z_S).to_value(u.Mpc)

M_true = 30
z_true = z_S

mass_bias_point = []
mass_bias_nfw = []

for mp, mn in zip(mu_point, mu_nfw):

    result_pm = calculate_mass_bias(
        d_L_true=d_L_true,
        mu=mp,
        M_true=M_true,
        z_true=z_true
    )

    mass_bias_point.append(
        result_pm["Mass bias (%)"].value
    )

    if np.isfinite(mn):

        result_nfw = calculate_mass_bias(
            d_L_true=d_L_true,
            mu=mn,
            M_true=M_true,
            z_true=z_true
        )

        mass_bias_nfw.append(
            result_nfw["Mass bias (%)"].value
        )

    else:
        mass_bias_nfw.append(np.nan)


# --------------------------------------------------
# 6. Plot comparison
# --------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(9, 10))

axes[0].plot(
    y_values, mu_point,
    label="Point Mass",
    linewidth=2
)

axes[0].plot(
    y_values, mu_nfw,
    label="NFW",
    linewidth=2
)

axes[0].set_xlabel("Dimensionless source position (y)")
axes[0].set_ylabel("Total magnification")
axes[0].set_title("Point-Mass vs NFW Magnification")
axes[0].legend()
axes[0].grid()


axes[1].plot(
    y_values, mass_bias_point,
    label="Point Mass",
    linewidth=2
)

axes[1].plot(
    y_values, mass_bias_nfw,
    label="NFW",
    linewidth=2
)

axes[1].set_xlabel("Dimensionless source position (y)")
axes[1].set_ylabel("Mass bias (%)")
axes[1].set_title("Point-Mass vs NFW Mass Bias")
axes[1].legend()
axes[1].grid()

fig.subplots_adjust(hspace=0.45)

plt.savefig(
    "figures/point_mass_vs_nfw.png",
    dpi=300,
    bbox_inches="tight"
)
# --------------------------------------------------
# 7. NFW Image Count vs Source Position
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.step(
    y_values,
    image_counts,
    where="mid",
    linewidth=2,
    label="NFW image count"
)

ax.set_xlabel("Dimensionless source position (y)")
ax.set_ylabel("Number of images")
ax.set_title("NFW Image Count vs Source Position")

ax.set_yticks([1, 2, 3, 4])
ax.grid(True)
ax.legend()

plt.savefig(
    "figures/nfw_image_count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.show()

print("NFW halo parameters")
print("-------------------")
print(f"Virial radius: {r_200 / 3.085677581e19:.3f} kpc")
print(f"Scale radius: {r_s / 3.085677581e19:.3f} kpc")
print(f"Characteristic density: {rho_s:.4e} kg/m^3")

print("\nComparison completed.")