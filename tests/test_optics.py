"""Tests of the optics chain: Gladstone-Dale, Abel projection, image formation."""

import math

import numpy as np
import pytest
from scipy.special import erf

from optics.abel import chord_matrix, project
from optics.gas import K_AIR, density, refractive_index
from optics.imaging import OpticsConfig, bos_displacement, schlieren, shadowgraph


# ------------------------------------------------------------ Gladstone-Dale --
def test_gladstone_dale_standard_air():
    rho = density(101325.0, 293.15)            # 1.204 kg/m^3
    assert rho == pytest.approx(1.204, rel=1e-3)
    # Standard-air refractivity at 633 nm, 20 C, 1 atm is n - 1 ~ 2.71e-4 (Ciddor 1996).
    assert refractive_index(rho) - 1 == pytest.approx(2.71e-4, rel=0.02)


def test_gladstone_dale_linear_in_density():
    rho = np.array([0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(refractive_index(rho) - 1, K_AIR * rho)


# ---------------------------------------------------------------- Abel ------
def test_chord_matrix_uniform_cylinder():
    R, nr = 1.0, 200
    y = np.linspace(-0.99, 0.99, 41)
    L = chord_matrix(np.linspace(0, R, nr + 1), y).sum(1)
    np.testing.assert_allclose(L, 2 * np.sqrt(R ** 2 - y ** 2), rtol=1e-12)


def test_abel_projection_of_gaussian():
    # f(r) = exp(-r^2/s^2)  ->  P(y) = sqrt(pi) s exp(-y^2/s^2)
    s, dr, nr = 1.0, 0.01, 600
    r = (np.arange(nr) + 0.5) * dr
    y, P = project(np.exp(-(r / s) ** 2)[:, None], dr)
    exact = math.sqrt(math.pi) * s * np.exp(-(y / s) ** 2)
    assert np.max(np.abs(P[:, 0] - exact)) < 2e-3 * exact.max()


# --------------------------------------------- knife edge on a phase step ---
def _phase_step(eps_max, sigma=4e-3, n=512, dx=50e-6):
    """Smooth phase step OPD(x) = A (1 + erf(x/sigma)) / 2, uniform in y.
    Its deflection eps(x) = A exp(-x^2/sigma^2) / (sigma sqrt(pi)) peaks at eps_max."""
    x = (np.arange(n) - n / 2 + 0.5) * dx
    A = eps_max * sigma * math.sqrt(math.pi)
    opd = np.tile(A * 0.5 * (1 + erf(x / sigma)), (n // 4, 1))
    eps = eps_max * np.exp(-(x / sigma) ** 2)
    return opd, eps, dx


@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_knife_edge_linear_response(offset):
    """Extended source, knife edge normal along x: I/I_bg - 1 = eps/(w - a_knife)."""
    w = 1e-3
    opd, eps, dx = _phase_step(eps_max=0.2 * w)
    cfg = OpticsConfig(source_halfwidth=w, knife_offset=offset, knife_angle_deg=0.0, n_source=65)
    I, I_bg, _ = schlieren(opd, dx, cfg, device="cpu")
    assert I_bg == pytest.approx((1 - offset) / 2, abs=0.02)
    resp = I[I.shape[0] // 2] / I_bg - 1
    expected = eps / (w * (1 - offset))
    # Geometric-optics prediction; residual is diffraction from the finite source/knife.
    assert np.max(np.abs(resp - expected)) < 0.1 * expected.max()


def test_knife_orientation():
    w = 1e-3
    opd, eps, dx = _phase_step(eps_max=0.2 * w)
    mid = opd.shape[0] // 2
    resp = {}
    for ang in (0.0, 180.0, 90.0):
        I, I_bg, _ = schlieren(opd, dx, OpticsConfig(source_halfwidth=w, knife_angle_deg=ang), device="cpu")
        resp[ang] = I[mid] / I_bg - 1
    peak = np.argmax(eps)
    assert resp[0.0][peak] > 0.15            # bright where light is deflected towards +x
    assert resp[180.0][peak] < -0.15         # flipped knife: dark
    assert np.abs(resp[90.0]).max() < 0.02   # knife parallel to the gradient: blind to it


def test_coherent_knife_on_uniform_field():
    """Point source, knife through the undiffracted spot: amplitude 1/2, I = 1/4 everywhere."""
    opd = np.zeros((64, 64))
    I, I_bg, _ = schlieren(opd, 1e-4, OpticsConfig(source_halfwidth=0.0), device="cpu")
    np.testing.assert_allclose(I, 0.25, atol=1e-6)
    assert I_bg == pytest.approx(0.25)


# ------------------------------------------------------ shadowgraph / BOS ---
def test_shadowgraph_weak_lens_matches_laplacian():
    """Weak phase object: I - 1 ~ -L * lap(OPD)."""
    n, dx = 256, 50e-6
    x = (np.arange(n) - n / 2 + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    s, A = 1.5e-3, 2e-8
    opd = A * np.exp(-(X ** 2 + Y ** 2) / s ** 2)
    L = 0.05
    I, _ = shadowgraph(opd, dx, OpticsConfig(shadow_distance=L), device="cpu")
    lap = A * np.exp(-(X ** 2 + Y ** 2) / s ** 2) * (4 * (X ** 2 + Y ** 2) / s ** 4 - 4 / s ** 2)
    assert np.mean(I) == pytest.approx(1.0, abs=1e-4)   # energy conserved
    assert np.max(np.abs((I - 1) + L * lap)) < 0.05 * np.max(np.abs(L * lap))


def test_bos_displacement_scales_with_distance():
    opd, eps, dx = _phase_step(eps_max=1e-4)
    dxb, dyb = bos_displacement(opd, dx, OpticsConfig(bos_distance=0.5))
    np.testing.assert_allclose(dxb[0], 0.5 * eps, atol=0.01 * 0.5 * eps.max())
    assert np.abs(dyb).max() < 1e-12
