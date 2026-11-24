import pytest

from src.bias import (
    apparent_luminosity_distance,
    apparent_source_mass,
    mass_bias_percentage,
    calculate_mass_bias
)


# Test 1: Apparent luminosity distance
def test_apparent_luminosity_distance():
    result = apparent_luminosity_distance(1000, 4)
    assert result == pytest.approx(500)


# Test 2: Invalid magnification
def test_invalid_magnification():
    with pytest.raises(ValueError):
        apparent_luminosity_distance(1000, 0)


# Test 3: Apparent source-frame mass
def test_apparent_source_mass():
    result = apparent_source_mass(30, 1.0, 0.5)
    assert result == pytest.approx(40)


# Test 4: Invalid true mass
def test_invalid_true_mass():
    with pytest.raises(ValueError):
        apparent_source_mass(-30, 1.0, 0.5)


# Test 5: Percentage mass bias
def test_mass_bias_percentage():
    result = mass_bias_percentage(40, 30)
    assert result == pytest.approx(33.3333333333)


# Test 6: Combined pipeline
def test_calculate_mass_bias():

    result = calculate_mass_bias(
       d_L_true=6791.26,
        mu=4,
        M_true=30,
        z_true=1.0
    )

    assert result["Apparent luminosity distance"]== pytest.approx(3395.63)
    assert result["Apparent redshift"].value == pytest.approx(0.5669867, abs=0.001)
    assert result["Apparent source mass"].value == pytest.approx(38.29005, abs=0.02)
    assert result["Mass bias (%)"].value == pytest.approx(27.6335, abs=0.1)