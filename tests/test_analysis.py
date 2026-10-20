"""Tests of the shock detector and the reference correlations."""

import numpy as np
import pytest

from analysis.shocks import CameraModel, DetectorConfig, detect
from analysis.theory import ashkenas_sherman, fully_expanded_diameter, fully_expanded_mach, pack_spacing
from optics.abel import project
from optics.gas import K_AIR, density
from optics.imaging import OpticsConfig, schlieren

D, CPD = 1e-3, 40


def _synthetic_jet_image(x_shock, seed=0):
    """Axisymmetric toy jet: density falls along the axis inside r < 1 D (expansion),
    then jumps up at a normal shock at x_shock. Rendered through the full optics chain."""
    h = 1.0 / CPD
    x = (np.arange(int(6 / h)) + 0.5) * h
    r = (np.arange(int(3 / h)) + 0.5) * h
    X, R = np.meshgrid(x, r)
    rho = np.ones_like(X)
    core = R < 1.0
    rho[core & (X < x_shock)] = (2.0 - 1.8 * X / x_shock)[core & (X < x_shock)]
    rho[core & (X >= x_shock)] = 1.5
    p_a = 5e3
    y, opd = project(K_AIR * (rho - 1.0) * density(p_a), D / CPD)
    img, bg, _ = schlieren(opd, D / CPD, OpticsConfig(source_halfwidth=0.5e-3), device="cpu")
    return CameraModel()(img / bg, np.random.default_rng(seed)), x, y / D


@pytest.mark.parametrize("x_shock", [2.0, 3.0125])
def test_mach_disk_detected_through_optics(x_shock):
    img, x, y = _synthetic_jet_image(x_shock)
    det = detect(img, x, y)
    # The step sits on a cell face at x_shock; detection within one pixel (D/40).
    assert det.x_md == pytest.approx(x_shock, abs=1.0 / CPD)


def test_shock_cells_on_synthetic_bands():
    x = (np.arange(400) + 0.5) * 0.025
    y = (np.arange(-120, 120) + 0.5) * 0.025
    band = lambda x0, a, w=0.05: a * np.exp(-((x - x0) / w) ** 2)
    prof = -0.4 * np.exp(-x / 0.5) + band(2.0, 0.4) + band(3.5, 0.15) + band(5.0, 0.12)
    img = 1.0 + np.where(np.abs(y)[:, None] < 1.5, prof[None, :], 0.0)
    img = CameraModel()(img, np.random.default_rng(3))
    det = detect(img, x, y)
    assert det.x_md == pytest.approx(2.0, abs=0.025)
    assert len(det.cells) == 2
    np.testing.assert_allclose(det.cells, [3.5, 5.0], atol=0.025)


def test_no_shock_in_pure_noise():
    rng = np.random.default_rng(1)
    x = (np.arange(300) + 0.5) * 0.025
    y = (np.arange(-100, 100) + 0.5) * 0.025
    det = detect(CameraModel()(np.ones((200, 300)), rng), x, y)
    assert det.x_md is None


def test_theory_values():
    assert ashkenas_sherman(100) == pytest.approx(6.7)
    # NPR at which a sonic jet is perfectly expanded: (1.2)^3.5 = 1.8929
    assert fully_expanded_mach(1.2 ** 3.5) == pytest.approx(1.0, abs=1e-12)
    assert fully_expanded_diameter(1.2 ** 3.5) == pytest.approx(1.0, abs=1e-12)
    # Isentropic table: M = 2 needs p0/p = 7.824 and A/A* = 1.6875
    assert fully_expanded_mach(7.824) == pytest.approx(2.0, abs=1e-3)
    assert fully_expanded_diameter(7.824) ** 2 == pytest.approx(1.6875, abs=1e-3)
    assert pack_spacing(7.824) == pytest.approx(1.306 * np.sqrt(3) * np.sqrt(1.6875), rel=1e-3)
