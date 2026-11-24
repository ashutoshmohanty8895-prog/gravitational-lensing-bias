from src.cosmology import luminosity_distance
import numpy as np

def test_luminosity_distance():
    assert 6700 < luminosity_distance(1.0) < 6900


from src.cosmology import (
    luminosity_distance,
    angular_diameter_distance,
    adm_diff,
    redshift_from_luminosity_distance
)
def test_luminosity_distance_monotonic():
    z = np.linspace(0.01, 5, 100)
    d = luminosity_distance(z)
    assert np.all(np.diff(d) > 0)

def test_roundtrip():
    for z in [0.1, 0.5, 1.0, 2.0]:
        recovered_z = redshift_from_luminosity_distance(
            luminosity_distance(z)
        )
        assert abs(recovered_z - z) < 1e-6
def test_distance_duality():
    for z in [0.1, 0.5, 1.0]:
        expected = luminosity_distance(z) / (1 + z)**2
        actual = angular_diameter_distance(z)
        assert abs(actual - expected) < 1e-6