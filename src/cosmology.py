from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from functools import lru_cache
cosmo = Planck18
def comoving_distance(z): 
    cm = cosmo.comoving_distance(z)
    return cm.to(u.Mpc).value
def luminosity_distance(z):
    lm = cosmo.luminosity_distance(z)
    return lm.to(u.Mpc).value
@lru_cache(maxsize=None)
def angular_diameter_distance(z):
    adm = cosmo.angular_diameter_distance(z)
    return adm.to(u.Mpc).value
def angular_diameter_distance(z):
    adm = cosmo.angular_diameter_distance(z)
    return adm.to(u.Mpc).value
@lru_cache(maxsize=None)
def adm_diff(z1, z2):
    difference = cosmo.angular_diameter_distance(z1, z2)
    return difference.to(u.Mpc).value
def adm_diff(z1,z2):
    difference = cosmo.angular_diameter_distance(z1, z2) #angular diameter distance is not an ordinary geometric distance that can always be added or subtracted. It depends on how the universe expands between the observer, lens and source
    return difference.to(u.Mpc).value
#z_at_value(func, target value)
def redshift_from_luminosity_distance(dL):
    DL = dL * u.Mpc
    return z_at_value(cosmo.luminosity_distance,DL)