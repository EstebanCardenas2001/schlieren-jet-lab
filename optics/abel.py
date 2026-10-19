"""Forward Abel projection: axisymmetric field f(x, r) -> line-of-sight integral.

A camera looking perpendicular to the jet axis sees, at image height y,

    P(x, y) = integral f dz = 2 * integral_{|y|}^{R} f(x, r) r dr / sqrt(r^2 - y^2).

Discretisation: f is treated as constant on each annular shell
[r_{j-1/2}, r_{j+1/2}] (exactly how the finite-volume solver stores it), so the
projection is P_i = sum_j L_ij f_j with exact chord lengths

    L_ij = 2 (sqrt(r_{j+1/2}^2 - y_i^2) - sqrt(r_{j-1/2}^2 - y_i^2))   (clipped at 0).

No singular integrand and no interpolation are needed.
"""

from __future__ import annotations

import numpy as np


def chord_matrix(r_edges: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L[i, j]: path length of the ray at height y[i] through shell j."""
    y2 = (np.asarray(y, dtype=np.float64) ** 2)[:, None]
    re2 = np.asarray(r_edges, dtype=np.float64) ** 2
    outer = np.sqrt(np.clip(re2[None, 1:] - y2, 0.0, None))
    inner = np.sqrt(np.clip(re2[None, :-1] - y2, 0.0, None))
    return 2.0 * (outer - inner)


def image_heights(nr: int, dr: float) -> np.ndarray:
    """Image rows at y = +-r_j (cell centres), mirrored about the axis: length 2*nr."""
    r = (np.arange(nr) + 0.5) * dr
    return np.concatenate([-r[::-1], r])


def project(f_rx: np.ndarray, dr: float) -> tuple[np.ndarray, np.ndarray]:
    """Project f[r, x] (r from the axis outwards, uniform spacing dr).

    Returns (y, P) with P of shape [2*nr, nx]; the field is zero beyond the
    outer radius nr*dr, so pass a field relative to the ambient value.
    """
    nr = f_rx.shape[0]
    r_edges = np.arange(nr + 1) * dr
    y = image_heights(nr, dr)
    L = chord_matrix(r_edges, y)
    return y, L @ f_rx
