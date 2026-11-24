"""
Physical constants and unit conversion factors
used throughout the gravitational lensing project.
"""

import astropy.constants as const
import astropy.units as u

# Fundamental physical constants (SI units)
G = const.G.value       # Gravitational constant (m^3 kg^-1 s^-2)
c = const.c.value       # Speed of light (m/s)

# Astronomical distance conversion
Mpc_to_m = (1 * u.Mpc).to(u.m).value