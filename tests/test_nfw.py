import pytest
import numpy as np

from src.nfw import (
    rho_nfw,
    M_enclosed,
    sigma_nfw,
    sigma_bar_nfw,
)
from src.nfw import (
    lens_equation_nfw,
    solve_images_nfw,
    total_magnification_nfw,
)


def test_rho_nfw():
    result = rho_nfw(2.0, 4.0, 1.0)
    assert result == pytest.approx(2.0 / 9.0)


def test_rho_nfw_array():
    result = rho_nfw([1.0, 2.0, 3.0], 4.0, 1.0)

    assert len(result) == 3
    assert result[0] > result[1] > result[2]


def test_enclosed_mass():
    result = M_enclosed(1.0, 2.0, 1.0)

    expected = 8.0 * np.pi * (np.log(2.0) - 0.5)
    assert result == pytest.approx(expected)


def test_enclosed_mass_increases_with_radius():
    m1 = M_enclosed(1.0, 2.0, 1.0)
    m2 = M_enclosed(2.0, 2.0, 1.0)

    assert m2 > m1


def test_sigma_nfw_at_unity():
    result = sigma_nfw(1.0, 3.0, 2.0)

    assert result == pytest.approx(4.0)


def test_sigma_bar_nfw_at_unity():
    result = sigma_bar_nfw(1.0, 3.0, 2.0)

    expected = 24.0 * (1.0 - np.log(2.0))
    assert result == pytest.approx(expected)


def test_sigma_nfw_array():
    result = sigma_nfw([0.5, 1.0, 2.0], 3.0, 2.0)

    assert len(result) == 3
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)
from src.nfw import (
    sigma_crit,
    kappa_nfw,
    gamma_nfw,
    mu_nfw,
)


def test_sigma_crit_positive():
    result = sigma_crit(0.5, 1.0)

    assert result > 0
    assert np.isfinite(result)


def test_kappa_nfw_positive():
    sigma_c = sigma_crit(0.5, 1.0)

    result = kappa_nfw(1.0, 1e-19, 1e20, sigma_c)

    assert np.isfinite(result)
    assert result > 0


def test_gamma_nfw_finite():
    sigma_c = sigma_crit(0.5, 1.0)

    result = gamma_nfw(1.0, 1e-19, 1e20, sigma_c)

    assert np.isfinite(result)


def test_mu_nfw_finite():
    sigma_c = sigma_crit(0.5, 1.0)

    result = mu_nfw(1.0, 1e-19, 1e20, sigma_c)

    assert np.isfinite(result)



from src.nfw import (
    M_projected_nfw,
    alpha_hat_nfw,
)


def test_projected_mass_positive():

    result = M_projected_nfw(1.0, 1e-19, 1e20)

    assert result > 0
    assert np.isfinite(result)


def test_projected_mass_increases():

    m1 = M_projected_nfw(1.0, 1e-19, 1e20)
    m2 = M_projected_nfw(2.0, 1e-19, 1e20)

    assert m2 > m1


def test_nfw_deflection_positive():

    result = alpha_hat_nfw(1.0, 1e-19, 1e20)

    assert result > 0
    assert np.isfinite(result)
def test_lens_equation_nfw_finite():
    result = lens_equation_nfw(
        1e-5, 5e-6, 1e-19, 1e20, 0.5, 1.0
    )

    assert np.isfinite(result)


def test_solve_images_nfw():
    images = solve_images_nfw(
        beta=5e-6,
        rho_s=1e-19,
        r_s=1e20,
        z_L=0.5,
        z_S=1.0,
        theta_max=1e-3,
    )

    assert len(images) >= 1
    assert np.all(np.isfinite(images))


def test_total_magnification_nfw():
    images = solve_images_nfw(
        beta=5e-6,
        rho_s=1e-19,
        r_s=1e20,
        z_L=0.5,
        z_S=1.0,
        theta_max=1e-3,
    )

    result = total_magnification_nfw(
        images, 1e-19, 1e20, 0.5, 1.0
    )

    assert np.isfinite(result)
    assert result > 0
# --------------------------------------------------
# Additional NFW validation tests
# --------------------------------------------------

def test_rho_nfw_decreases_with_radius():
    r = np.array([1.0, 2.0, 4.0])
    result = rho_nfw(r, 4.0, 1.0)

    assert np.all(np.diff(result) < 0)


def test_enclosed_mass_is_zero_at_origin():
    result = M_enclosed(0.0, 2.0, 1.0)

    assert result == pytest.approx(0.0, abs=1e-12)


def test_projected_mass_is_zero_at_origin():
    result = M_projected_nfw(0.0, 1e-19, 1e20)

    assert result == pytest.approx(0.0, abs=1e-12)


def test_nfw_deflection_decreases_at_large_radius():
    alpha1 = alpha_hat_nfw(1.0, 1e-19, 1e20)
    alpha2 = alpha_hat_nfw(2.0, 1e-19, 1e20)

    assert alpha2 < alpha1