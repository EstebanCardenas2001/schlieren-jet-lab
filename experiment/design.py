"""BOS experiment design: how many pixels will a hot plume shift the background?

Model: an axisymmetric plume whose temperature excess is Gaussian across it,
T(r) = T_a + dT exp(-r^2/s^2), at ambient pressure. Ideal gas and
Gladstone-Dale give the refractive-index change

    dn(r) = K rho_a (T_a / T(r) - 1).

For a Gaussian dn(r) = dn0 exp(-r^2/s^2), the Abel projection is
OPD(y) = sqrt(pi) s dn0 exp(-y^2/s^2), so the peak ray deflection is

    eps_max = max |dOPD/dy| = sqrt(2 pi) exp(-1/2) |dn0| ~ 1.52 |dn0|,

*independent of the plume width s*: a wider plume has a longer path but a
gentler gradient. (The exact profile from T(r) is not Gaussian in dn; the
code integrates it numerically with optics.abel, and the Gaussian formula is
checked in the tests.) The background shift is Z_D * eps in the background
plane; divided by the camera's pixel footprint on the background it gives
pixels.

    python -m experiment.design [--zd 0.5] [--footprint-mm 0.2]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from optics.abel import project
from optics.gas import K_AIR, T_ROOM, density

ROOT = Path(__file__).resolve().parents[1]
GAUSS_PEAK = math.sqrt(2 * math.pi) * math.exp(-0.5)  # 1.520


def peak_deflection(dT: float, width_m: float = 0.02, T_a: float = T_ROOM, p_a: float = 101325.0,
                    K: float = K_AIR, n: int = 400) -> float:
    """Peak |eps| [rad] of a Gaussian-temperature plume (1/e half-width `width_m`), numerically."""
    dr = 4 * width_m / n
    r = (np.arange(n) + 0.5) * dr
    T = T_a + dT * np.exp(-(r / width_m) ** 2)
    dn = K * density(p_a, T_a) * (T_a / T - 1.0)
    y, opd = project(dn[:, None], dr)
    return float(np.max(np.abs(np.gradient(opd[:, 0], y))))


def expected_pixels(dT, z_d, footprint_m, **kw) -> float:
    return z_d * peak_deflection(dT, **kw) / footprint_m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zd", type=float, default=0.5, help="flow-to-background distance [m]")
    ap.add_argument("--footprint-mm", type=float, default=0.2,
                    help="background width covered by one pixel [mm] (e.g. 400 mm / 1920 px ~ 0.21)")
    ap.add_argument("--width-mm", type=float, default=20.0, help="plume 1/e half-width [mm]")
    a = ap.parse_args()
    rows = []
    print(f"Z_D = {a.zd} m, pixel footprint = {a.footprint_mm} mm, plume half-width {a.width_mm} mm")
    print(f"{'dT [K]':>7} {'eps_max [urad]':>15} {'shift [um]':>11} {'shift [px]':>11}")
    for dT in (10, 20, 50, 100, 200):
        eps = peak_deflection(dT, a.width_mm / 1e3)
        px = a.zd * eps / (a.footprint_mm / 1e3)
        rows.append({"dT_K": dT, "eps_max_rad": eps, "shift_m": a.zd * eps, "shift_px": px})
        print(f"{dT:7.0f} {eps * 1e6:15.1f} {a.zd * eps * 1e6:11.1f} {px:11.2f}")
    out = ROOT / "results" / "experiment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps({"z_d_m": a.zd, "footprint_mm": a.footprint_mm,
                                                 "plume_halfwidth_mm": a.width_mm, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
