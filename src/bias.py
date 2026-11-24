from .cosmology import redshift_from_luminosity_distance
import numpy as np
def apparent_luminosity_distance(d_L_true, mu):
   if(mu<=0):
    raise ValueError
   else:
    d_L_app = d_L_true/ np.sqrt(mu)
    return d_L_app
def apparent_redshift(d_L_true, mu):

    if mu <= 0:
        raise ValueError("Magnification must be positive.")

    d_L_app = apparent_luminosity_distance(d_L_true, mu)

    z_app = redshift_from_luminosity_distance(d_L_app)

    return z_app
def apparent_source_mass(M_true, z_true, z_app):

    if M_true <= 0:
        raise ValueError("True mass must be positive.")

    if z_true < 0:
        raise ValueError("True redshift cannot be negative.")

    if z_app < 0:
        raise ValueError("Apparent redshift cannot be negative.")

    M_app = ((1 + z_true) * M_true) / (1 + z_app)

    return M_app
def mass_bias_percentage(M_app, M_true):

    if M_true <= 0:
        raise ValueError("True mass must be positive.")

    mass_bias = (M_app - M_true) / M_true * 100

    return mass_bias
def calculate_mass_bias(d_L_true, mu, M_true, z_true):

    d_L_app = apparent_luminosity_distance(d_L_true, mu)

    z_app = redshift_from_luminosity_distance(d_L_app)

    M_app = apparent_source_mass(M_true, z_true, z_app)

    bias = mass_bias_percentage(M_app, M_true)

    return {
        "Apparent luminosity distance": d_L_app,
        "Apparent redshift": z_app,
        "Apparent source mass": M_app,
        "Mass bias (%)": bias
    }