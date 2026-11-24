from src.point_mass import einstein_radius
from src.cosmology import angular_diameter_distance
from src.constants import Mpc_to_m

import numpy as np

# Lens parameters
M_L = 30 * 1.98847e30  # kg
z_L = 0.5
z_S = 1.0

# Calculate angular Einstein radius
theta_E = einstein_radius(M_L, z_L, z_S)

# Convert radians to arcseconds
theta_E_arcsec = theta_E * 206265

# Calculate angular diameter distance to lens
D_L = angular_diameter_distance(z_L) * Mpc_to_m

# Physical Einstein radius in the lens plane
R_E = D_L * theta_E

print("Angular Einstein radius:", theta_E, "radians")
print("Angular Einstein radius:", theta_E_arcsec, "arcseconds")
print("Physical Einstein radius:", R_E / 1000, "km")