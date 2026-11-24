import pytest
import numpy as np

from src.point_mass import point_mass_magnification, einstein_radius


# --------------------------------------------------
# Tests for point_mass_magnification()
# --------------------------------------------------

def test_point_mass_magnification_at_unity():
    result = point_mass_magnification(1)
    assert result == pytest.approx(3 / np.sqrt(5))


def test_point_mass_magnification_array():
    result = point_mass_magnification([0.5, 1, 2])

    assert len(result) == 3
    assert result[0] > result[1] > result[2]


def test_point_mass_magnification_zero():
    with pytest.raises(ValueError):
        point_mass_magnification(0)


def test_point_mass_magnification_negative():
    with pytest.raises(ValueError):
        point_mass_magnification(-1)


def test_point_mass_magnification_positive():
    result = point_mass_magnification(2)
    assert result > 1


# --------------------------------------------------
# Tests for einstein_radius()
# --------------------------------------------------

def test_einstein_radius_positive():
    result = einstein_radius(
        M_L=30 * 1.98847e30,
        z_L=0.5,
        z_S=1.0
    )

    assert result > 0
    assert np.isfinite(result)


def test_einstein_radius_invalid_mass():
    with pytest.raises(ValueError):
        einstein_radius(
            M_L=0,
            z_L=0.5,
            z_S=1.0
        )


def test_einstein_radius_invalid_redshift():
    with pytest.raises(ValueError):
        einstein_radius(
            M_L=30 * 1.98847e30,
            z_L=1.0,
            z_S=0.5
        )
# --------------------------------------------------
# Additional physical validation tests
# --------------------------------------------------

def test_point_mass_magnification_approaches_one():
    result = point_mass_magnification(1000)
    assert result == pytest.approx(1.0, rel=1e-5)


def test_point_mass_magnification_small_y():
    result = point_mass_magnification(0.01)
    assert result > 100


def test_einstein_radius_increases_with_mass():
    r1 = einstein_radius(
        M_L=1e30,
        z_L=0.5,
        z_S=1.0
    )

    r2 = einstein_radius(
        M_L=4e30,
        z_L=0.5,
        z_S=1.0
    )

    assert r2 > r1
    assert r2 / r1 == pytest.approx(2.0)


def test_einstein_radius_invalid_source_redshift():
    with pytest.raises(ValueError):
        einstein_radius(
            M_L=30 * 1.98847e30,
            z_L=0.5,
            z_S=0.5
        )