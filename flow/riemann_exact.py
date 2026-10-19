"""Exact solution of the 1D Riemann problem for the Euler equations.

Follows Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
(3rd ed., ch. 4): Newton iteration for the star-region pressure p*, then
self-similar sampling of the wave pattern at x/t. Used as ground truth for the
Sod shock-tube validation of the finite-volume solver.
"""

from __future__ import annotations

import numpy as np

from .euler import GAMMA


def _f_and_df(p, rho_k, p_k, c_k, g):
    """Pressure function f_K(p) and derivative for the left or right wave."""
    if p > p_k:  # shock
        A = 2.0 / ((g + 1.0) * rho_k)
        B = (g - 1.0) / (g + 1.0) * p_k
        sq = np.sqrt(A / (p + B))
        return (p - p_k) * sq, sq * (1.0 - 0.5 * (p - p_k) / (B + p))
    # rarefaction
    pr = p / p_k
    f = 2.0 * c_k / (g - 1.0) * (pr ** ((g - 1.0) / (2.0 * g)) - 1.0)
    df = 1.0 / (rho_k * c_k) * pr ** (-(g + 1.0) / (2.0 * g))
    return f, df


def star_state(left, right, g=GAMMA, tol=1e-12, max_iter=100):
    """Return (p*, u*) for primitive states left/right = (rho, u, p)."""
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    if 2.0 * (cL + cR) / (g - 1.0) <= uR - uL:
        raise ValueError("initial data generate vacuum")
    # Two-rarefaction initial guess (robust for Sod-type problems).
    z = (g - 1.0) / (2.0 * g)
    p = ((cL + cR - 0.5 * (g - 1.0) * (uR - uL)) / (cL / pL ** z + cR / pR ** z)) ** (1.0 / z)
    for _ in range(max_iter):
        fL, dL = _f_and_df(p, rL, pL, cL, g)
        fR, dR = _f_and_df(p, rR, pR, cR, g)
        p_new = max(p - (fL + fR + uR - uL) / (dL + dR), tol)
        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break
        p = p_new
    fL, _ = _f_and_df(p, rL, pL, cL, g)
    fR, _ = _f_and_df(p, rR, pR, cR, g)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def sample(xi, left, right, g=GAMMA):
    """Exact (rho, u, p) at similarity coordinates xi = (x - x0) / t (array)."""
    xi = np.asarray(xi, dtype=float)
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    ps, us = star_state(left, right, g)
    gm, gp = (g - 1.0) / (g + 1.0), (g - 1.0) / (2.0 * g)
    rho, u, p = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)

    for i, s in enumerate(xi):
        if s <= us:  # left of contact
            if ps > pL:  # left shock
                SL = uL - cL * np.sqrt((g + 1) / (2 * g) * ps / pL + gp)
                if s <= SL:
                    rho[i], u[i], p[i] = rL, uL, pL
                else:
                    rho[i] = rL * (ps / pL + gm) / (gm * ps / pL + 1.0)
                    u[i], p[i] = us, ps
            else:  # left rarefaction
                cs = cL * (ps / pL) ** gp
                head, tail = uL - cL, us - cs
                if s <= head:
                    rho[i], u[i], p[i] = rL, uL, pL
                elif s >= tail:
                    rho[i], u[i], p[i] = rL * (ps / pL) ** (1 / g), us, ps
                else:  # inside the fan
                    u[i] = 2 / (g + 1) * (cL + 0.5 * (g - 1) * uL + s)
                    c = 2 / (g + 1) * (cL + 0.5 * (g - 1) * (uL - s))
                    rho[i] = rL * (c / cL) ** (2 / (g - 1))
                    p[i] = pL * (c / cL) ** (2 * g / (g - 1))
        else:  # right of contact
            if ps > pR:  # right shock
                SR = uR + cR * np.sqrt((g + 1) / (2 * g) * ps / pR + gp)
                if s >= SR:
                    rho[i], u[i], p[i] = rR, uR, pR
                else:
                    rho[i] = rR * (ps / pR + gm) / (gm * ps / pR + 1.0)
                    u[i], p[i] = us, ps
            else:  # right rarefaction
                cs = cR * (ps / pR) ** gp
                head, tail = uR + cR, us + cs
                if s >= head:
                    rho[i], u[i], p[i] = rR, uR, pR
                elif s <= tail:
                    rho[i], u[i], p[i] = rR * (ps / pR) ** (1 / g), us, ps
                else:
                    u[i] = 2 / (g + 1) * (-cR + 0.5 * (g - 1) * uR + s)
                    c = 2 / (g + 1) * (cR - 0.5 * (g - 1) * (uR - s))
                    rho[i] = rR * (c / cR) ** (2 / (g - 1))
                    p[i] = pR * (c / cR) ** (2 * g / (g - 1))
    return rho, u, p
