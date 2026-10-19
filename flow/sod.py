"""Sod shock-tube validation of the 2D solver against the exact Riemann solution.

The 1D problem is run through the full 2D code (a thin strip with slip walls
on the transverse sides), along x and along y, so both flux directions and
the boundary machinery are exercised.

    python -m flow.sod [--fast] [--device cuda]

writes results/figures/sod.png and results/sod_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .euler import EulerSolver2D
from .riemann_exact import sample

# Sod (1978): rho, u, p left / right of a diaphragm at x = 0.5, output at t = 0.2.
SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
T_END = 0.2
NT = 4  # transverse cells of the strip

ROOT = Path(__file__).resolve().parents[1]


def run_sod(n: int, direction: str = "x", limiter: str = "minmod",
            device: str = "cpu", dtype=torch.float64, cfl: float = 0.4):
    """Return cell centres, numerical (rho, u, p) and exact (rho, u, p) at T_END."""
    along_x = direction == "x"
    nx, ny = (n, NT) if along_x else (NT, n)
    h = 1.0 / n
    bc = ({"left": "outflow", "right": "outflow", "bottom": "reflect", "top": "reflect"} if along_x else
          {"left": "reflect", "right": "reflect", "bottom": "outflow", "top": "outflow"})
    s = EulerSolver2D(nx, ny, h, h, bc=bc, limiter=limiter, device=device, dtype=dtype)
    coord = s.x if along_x else s.y
    left = coord < 0.5
    shape = (ny, nx)
    left2d = left[None, :].expand(shape) if along_x else left[:, None].expand(shape)
    W = torch.empty((4, *shape), device=s.device, dtype=dtype)
    W[0] = torch.where(left2d, SOD_LEFT[0], SOD_RIGHT[0])
    W[1] = 0.0
    W[2] = 0.0
    W[3] = torch.where(left2d, SOD_LEFT[2], SOD_RIGHT[2])
    s.set_initial(W)
    s.run(T_END, cfl=cfl)

    Wn = s.W.cpu().numpy()
    if along_x:
        rho, u, p = Wn[0].mean(0), Wn[1].mean(0), Wn[3].mean(0)
        transverse_spread = np.abs(Wn[0] - Wn[0].mean(0)).max()
    else:
        rho, u, p = Wn[0].mean(1), Wn[2].mean(1), Wn[3].mean(1)
        transverse_spread = np.abs(Wn[0] - Wn[0].mean(1, keepdims=True)).max()
    x = coord.cpu().numpy()
    exact = sample((x - 0.5) / T_END, SOD_LEFT, SOD_RIGHT)
    return x, (rho, u, p), exact, transverse_spread


def l1_error(num, exact, h):
    return float(np.sum(np.abs(num - exact)) * h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="coarse resolutions only (CI)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ns = [50, 100, 200] if args.fast else [100, 200, 400, 800]
    metrics = {"t_end": T_END, "limiter": "minmod", "cfl": 0.4, "runs": []}
    for lim in ("minmod", "vanleer"):
        for n in ns:
            x, num, ex, spread = run_sod(n, "x", lim, args.device)
            metrics["runs"].append({
                "limiter": lim, "n": n,
                "L1_rho": l1_error(num[0], ex[0], 1.0 / n),
                "L1_u": l1_error(num[1], ex[1], 1.0 / n),
                "L1_p": l1_error(num[2], ex[2], 1.0 / n),
                "transverse_spread": float(spread),
            })
    # Observed convergence order of L1(rho) between successive resolutions.
    for lim in ("minmod", "vanleer"):
        runs = [r for r in metrics["runs"] if r["limiter"] == lim]
        metrics[f"order_L1_rho_{lim}"] = [
            float(np.log2(a["L1_rho"] / b["L1_rho"])) for a, b in zip(runs, runs[1:])]

    # x vs y symmetry check at the middle resolution.
    n_mid = ns[len(ns) // 2]
    _, nx_, _, _ = run_sod(n_mid, "x", "minmod", args.device)
    _, ny_, _, _ = run_sod(n_mid, "y", "minmod", args.device)
    metrics["xy_max_abs_diff_rho"] = float(np.abs(nx_[0] - ny_[0]).max())

    for r in metrics["runs"]:
        print(f"{r['limiter']:8s} N={r['n']:4d}  L1(rho)={r['L1_rho']:.3e}  L1(u)={r['L1_u']:.3e}  "
              f"L1(p)={r['L1_p']:.3e}")
    for lim in ("minmod", "vanleer"):
        print(f"observed L1(rho) order, {lim}: {np.round(metrics[f'order_L1_rho_{lim}'], 3)}")
    print(f"max |rho_x - rho_y| at N={n_mid}: {metrics['xy_max_abs_diff_rho']:.2e}")

    out = ROOT / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "sod_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Figure: density / velocity / pressure at N = 200 (both limiters) vs exact.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_fig = 200
    xf = np.linspace(0, 1, 2001)
    exf = sample((xf - 0.5) / T_END, SOD_LEFT, SOD_RIGHT)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    sols = {lim: run_sod(n_fig, "x", lim, args.device) for lim in ("minmod", "vanleer")}
    for k, (ax, name) in enumerate(zip(axes, ["density", "velocity", "pressure"])):
        ax.plot(xf, exf[k], "k-", lw=1.2, label="exact")
        for lim, mk in (("minmod", "o"), ("vanleer", "s")):
            x, num, _, _ = sols[lim]
            ax.plot(x, num[k], mk, ms=2.5, mfc="none", label=f"HLLC + {lim}")
        ax.set_title(name)
        ax.set_xlabel("x")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Sod shock tube, t = {T_END}, N = {n_fig}")
    fig.savefig(out / "figures" / "sod.png", dpi=150)
    print(f"wrote {out / 'figures' / 'sod.png'} and {out / 'sod_metrics.json'}")


if __name__ == "__main__":
    main()
