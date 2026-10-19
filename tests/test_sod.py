"""Validation of the Euler solver: Sod shock tube vs the exact Riemann solution,
plus sanity checks of the axisymmetric geometry terms."""

import numpy as np
import pytest
import torch

from flow.euler import EulerSolver2D
from flow.riemann_exact import sample, star_state
from flow.sod import SOD_LEFT, SOD_RIGHT, l1_error, run_sod


def test_exact_riemann_star_state():
    # Reference star state for Sod from Toro (2009), Table 4.2 (test 1).
    p_star, u_star = star_state(SOD_LEFT, SOD_RIGHT)
    assert p_star == pytest.approx(0.30313, abs=1e-5)
    assert u_star == pytest.approx(0.92745, abs=1e-5)


@pytest.mark.parametrize("direction", ["x", "y"])
def test_sod_l1_error(direction):
    n = 200
    _, num, ex, spread = run_sod(n, direction)
    # Measured at N=200 with minmod: L1(rho)=4.2e-3, L1(p)=3.1e-3 (results/sod_metrics.json).
    assert l1_error(num[0], ex[0], 1 / n) < 6e-3
    assert l1_error(num[2], ex[2], 1 / n) < 5e-3
    assert spread < 1e-12  # the 1D problem stays exactly 1D in the 2D code


def test_sod_converges():
    errs = [l1_error(num[0], ex[0], 1 / n) for n in (100, 200)
            for _, num, ex, _ in [run_sod(n)]]
    order = np.log2(errs[0] / errs[1])
    assert order > 0.7  # discontinuous solution: ~0.8 expected, not 2


def test_sod_x_equals_y():
    _, nx_, _, _ = run_sod(100, "x")
    _, ny_, _, _ = run_sod(100, "y")
    np.testing.assert_allclose(nx_[0], ny_[0], atol=1e-13)


def _axisym_solver(nx=24, ny=16, **kw):
    return EulerSolver2D(nx, ny, 0.1, 0.1, axisymmetric=True,
                         bc={"left": "reflect", "right": "reflect", "bottom": "reflect", "top": "reflect"}, **kw)


def test_axisymmetric_quiescent_gas_stays_at_rest():
    s = _axisym_solver()
    W = torch.zeros(4, s.ny, s.nx, dtype=torch.float64)
    W[0], W[3] = 1.0, 1.0
    s.set_initial(W)
    s.run(1.0)
    W1 = s.W
    assert W1[1].abs().max() < 1e-13 and W1[2].abs().max() < 1e-13
    assert (W1[3] - 1.0).abs().max() < 1e-13


def test_axisymmetric_mass_conserved_in_closed_box():
    s = _axisym_solver()
    W = torch.zeros(4, s.ny, s.nx, dtype=torch.float64)
    r2 = (s.x[None, :] - 1.2) ** 2 + s.y[:, None] ** 2
    W[0] = 1.0 + 2.0 * (r2 < 0.25)  # dense, high-pressure bubble on the axis
    W[3] = 1.0 + 9.0 * (r2 < 0.25)
    s.set_initial(W)
    vol = s.r_c * s.dx * s.dy  # cell volume per radian
    m0 = (s.U[0] * vol).sum().item()
    s.run(0.5)
    m1 = (s.U[0] * vol).sum().item()
    assert abs(m1 - m0) / m0 < 1e-12
    assert torch.isfinite(s.U).all()
