"""Underexpanded axisymmetric jet from a sonic nozzle into quiescent gas.

Physics / setup (non-dimensional)
---------------------------------
* Ambient: rho_a = 1, p_a = 1, so T_a = p_a/rho_a = 1 (gas constant R = 1)
  and c_a = sqrt(gamma).
* Nozzle: exit diameter D = 1, sonic exit (M_e = 1). The reservoir is at
  stagnation pressure p0 = NPR * p_a and stagnation temperature T0 = T_a
  (room-temperature gas feeding a room-temperature chamber). Isentropic sonic
  conditions give
      p_e = p0 (2/(g+1))^(g/(g-1)) ~ 0.528 p0,   T_e = 2 T0/(g+1),
      u_e = c_e = sqrt(g T_e).
  For NPR > 1.89 the exit pressure exceeds p_a, so the jet is underexpanded:
  it expands through a Prandtl-Meyer fan at the lip, overshoots, and the
  reflected compression waves coalesce into a barrel shock that ends, at high
  NPR, in a normal shock (the Mach disk).
* Geometry: axisymmetric (x, r), exit plane at x = 0. The nozzle is treated
  as thin-lipped: outside r = D/2 the left boundary is open (Riemann-invariant
  far field), like the top and right boundaries, so ambient gas can be
  entrained from upstream. Bottom = axis.
* Sponge layer: in the outer 40 % of the radius the solution is relaxed
  towards the quiescent ambient state. Without it (and with a closed back
  wall, as in a first version) slow vortices and entropy blobs from the
  start-up stay trapped in the ambient gas. Their sharp edges project into
  vertical lines in synthetic Schlieren images that look like normal shocks.
  A real chamber removes them by turbulent mixing, which Euler lacks.
* Inviscid Euler: no viscosity, so there is no physical mixing layer and the
  jet shear layer rolls up through numerical (grid) instabilities only.
  The *near field* (expansion, barrel shock, Mach disk, first cells) is
  inviscid-dominated and well captured by Euler; the far field is not.
* Scaling: with no viscosity the solution depends only on NPR and gamma, not
  on the absolute pressure level. Real densities follow by multiplying rho by
  the dimensional ambient density, which is how the optics stage models
  "lower ambient pressure at the same NPR".

The domain is sized from a rough a-priori Mach-disk estimate so that every
NPR resolves the same physics. The *measurement* of the Mach disk is done
later on synthetic Schlieren images, never on these fields.

    python -m flow.jet [--fast] [--npr 5 10 ...] [--cells-per-d 40] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .euler import GAMMA, EulerSolver2D, cons_to_prim, mach_number

ROOT = Path(__file__).resolve().parents[1]
FULL_NPRS = [5, 10, 15, 20, 30, 45, 60, 80, 100]
FAST_NPRS = [10, 40]
SPONGE_STRENGTH = 1.0  # peak relaxation rate, in units of sqrt(p_a/rho_a)/D
WINDOW_TRANSITS = 3.0  # averaging window, in jet transit times Lx/u_max


def sonic_exit_state(npr: float, g: float = GAMMA) -> dict:
    p0, T0 = npr, 1.0
    p_e = p0 * (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    T_e = 2.0 * T0 / (g + 1.0)
    u_e = math.sqrt(g * T_e)
    return {"rho": p_e / T_e, "u": u_e, "p": p_e, "T": T_e}


def domain_for(npr: float) -> tuple[float, float]:
    """Domain (Lx, Lr) in nozzle diameters, scaled with the expected cell size."""
    x_md_guess = 0.67 * math.sqrt(npr)  # sizing only, not a result
    Lx = max(6.0, 2.6 * x_md_guess)
    Lr = max(3.0, 0.5 * Lx)
    return Lx, Lr


def simulate(npr: float, cells_per_d: int, device: str, dtype=torch.float32,
             cfl: float = 0.4, limiter: str = "minmod", max_windows: int = 6,
             tol: float = 1e-3, n_snapshots: int = 8, compile: bool | None = None,
             verbose: bool = True) -> dict:
    g = GAMMA
    Lx, Lr = domain_for(npr)
    h = 1.0 / cells_per_d
    nx, ny = round(Lx / h), round(Lr / h)
    ex = sonic_exit_state(npr)
    W_amb = torch.tensor([1.0, 0.0, 0.0, 1.0])
    W_in = torch.tensor([ex["rho"], ex["u"], 0.0, ex["p"]])

    if compile is None:
        compile = torch.device(device).type == "cuda"
    r_cells = (torch.arange(ny) + 0.5) * h
    r_s = 0.6 * Lr
    ramp = ((r_cells - r_s) / (Lr - r_s)).clamp(min=0.0)
    sponge = SPONGE_STRENGTH * (ramp ** 2)[:, None].expand(ny, nx)
    s = EulerSolver2D(nx, ny, h, h, axisymmetric=True,
                      bc={"left": "farfield", "right": "farfield", "bottom": "reflect", "top": "farfield"},
                      W_amb=W_amb, inflow=("left", r_cells < 0.5, W_in),  # nozzle exit r < D/2
                      sponge=sponge, limiter=limiter, device=device, dtype=dtype, compile=compile)
    W0 = W_amb.to(s.device, dtype)[:, None, None].expand(4, ny, nx).clone()
    s.set_initial(W0)

    # Time scales from the fully expanded jet speed u_max = sqrt(2 g T0/(g-1)).
    # 1) start-up: let the starting vortex (moving at roughly 0.4 u_max) leave the domain;
    # 2) average the fields over windows of WINDOW_TRANSITS jet transit times Lx/u_max.
    # Only the supersonic core upstream of the Mach disk (expansion fan, barrel
    # shock) becomes truly steady; the Mach disk itself oscillates axially by a
    # few percent and the subsonic flow behind it and the inviscid shear layer
    # stay unsteady. Steadiness is therefore judged on the *time-averaged*
    # density in the upstream supersonic core: the median relative difference
    # between two consecutive window averages must be below `tol`. The median
    # (rather than the mean) ignores the thin supersonic layer between barrel
    # shock and jet boundary, which the unsteady shear layer keeps buffeting;
    # its 90th percentile is recorded too. The averaged field is what a Schlieren camera with
    # an exposure much longer than D/u_jet records; the instantaneous snapshots
    # kept from the last window act as short-exposure frames, from which the
    # analysis stage measures the Mach-disk jitter.
    u_max = math.sqrt(2 * g / (g - 1))
    t_start = Lx / (0.4 * u_max)
    window = WINDOW_TRANSITS * Lx / u_max
    x_md_guess = 0.67 * math.sqrt(npr)  # region selection only
    box = (s.x[None, :] < 0.8 * x_md_guess) & (s.y[:, None] < 0.45 * x_md_guess + 0.5)
    t0 = time.time()
    s.run(t_start, cfl=cfl)
    if verbose:
        print(f"  NPR={npr:5.1f} start-up done: t={s.t.item():.2f}, {s.nsteps} steps, {time.time()-t0:.1f} s")

    acc = {"sum": torch.zeros_like(s.U), "n": 0, "frames": []}

    def accumulate(solver):
        W = cons_to_prim(solver.U, g)
        acc["sum"] += W
        acc["n"] += 1
        acc["frames"].append(W[0].clone())

    history, prev = [], None
    for k in range(1, max_windows + 1):
        acc["sum"].zero_(); acc["n"] = 0; acc["frames"] = []
        s.run(t_start + k * window, cfl=cfl, check_every=10, callback=accumulate)
        avg = acc["sum"] / acc["n"]
        if prev is not None:
            core = box & (mach_number(avg, g) > 1.2) & (mach_number(prev, g) > 1.2)  # supersonic core
            a, b = avg[0][core], prev[0][core]
            rel = (a - b).abs() / a
            change = rel.median().item()
            history.append({"t": s.t.item(), "steps": s.nsteps, "samples": acc["n"],
                            "core_avg_rel_change": change,
                            "core_avg_rel_change_p90": torch.quantile(rel.float(), 0.9).item()})
            if verbose:
                print(f"  NPR={npr:5.1f} t={s.t.item():7.2f} steps={s.nsteps:6d} "
                      f"window-avg core drho median={change:.1e} p90={history[-1]['core_avg_rel_change_p90']:.1e}")
            if change < tol:
                break
        prev = avg
    runtime = time.time() - t0

    W_avg = avg.float().cpu().numpy()
    M_avg = mach_number(avg).float().cpu().numpy()
    idx = np.linspace(0, len(acc["frames"]) - 1, n_snapshots).round().astype(int)
    frames = torch.stack([acc["frames"][i] for i in idx]).float().cpu().numpy()
    final_change = history[-1]["core_avg_rel_change"]
    return {
        "x": s.x.cpu().numpy(), "r": s.y.cpu().numpy(),
        # time-averaged fields (primary output; Mach is computed from the averaged primitives)
        "rho": W_avg[0], "u": W_avg[1], "v": W_avg[2], "p": W_avg[3], "mach": M_avg,
        # instantaneous density snapshots, equally spaced over the last window
        "rho_frames": frames,
        "meta": {
            "npr": npr, "gamma": g, "D": 1.0, "cells_per_d": cells_per_d, "nx": nx, "ny": ny,
            "Lx": Lx, "Lr": Lr, "geometry": "axisymmetric (x, r); r=0 is the axis; thin-lipped nozzle, open back boundary",
            "sponge": {"r_start": r_s, "strength": SPONGE_STRENGTH},
            "units": "non-dimensional: rho/rho_a, p/p_a, velocities / sqrt(p_a/rho_a), lengths / D",
            "fields": "rho/u/v/p/mach: time average over the last window; rho_frames: snapshots in it",
            "exit_state": ex, "t_start": t_start, "window": window, "t_final": s.t.item(),
            "steps": s.nsteps, "cfl": cfl, "limiter": limiter,
            "dtype": str(dtype), "device": str(s.device), "runtime_s": runtime,
            "steady_tol": tol, "converged": final_change < tol, "compiled": compile, "history": history,
        },
    }


def save(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npr = result["meta"]["npr"]
    path = out_dir / f"jet_npr{npr:g}.npz"
    np.savez_compressed(path, **{k: v for k, v in result.items() if k != "meta"},
                        meta=json.dumps(result["meta"]))
    return path


def load(path: Path) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def overview_figure(paths: list[Path], fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(paths)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.0 * n + 0.6), constrained_layout=True, squeeze=False)
    for (axd, axm), p in zip(axes, paths):
        d = load(p)
        x, r = d["x"], d["r"]
        ext = [x[0], x[-1], -r[-1], r[-1]]  # mirror about the axis for display
        mirror = lambda f: np.concatenate([f[::-1], f], axis=0)
        im = axd.imshow(np.log10(mirror(d["rho"])), origin="lower", extent=ext, cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=axd, label=r"$\log_{10}\rho/\rho_a$", shrink=0.9)
        im = axm.imshow(mirror(d["mach"]), origin="lower", extent=ext, cmap="magma", aspect="equal")
        fig.colorbar(im, ax=axm, label="Mach", shrink=0.9)
        axd.set_ylabel(f"NPR {d['meta']['npr']:g}\nr/D")
    axes[-1, 0].set_xlabel("x/D")
    axes[-1, 1].set_xlabel("x/D")
    fig.savefig(fig_path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="2 NPRs on a coarse grid (CI)")
    ap.add_argument("--npr", type=float, nargs="*")
    ap.add_argument("--cells-per-d", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "flow")
    args = ap.parse_args()

    nprs = args.npr or (FAST_NPRS if args.fast else FULL_NPRS)
    cpd = args.cells_per_d or (12 if args.fast else 40)
    paths, summary = [], []
    for npr in nprs:
        res = simulate(npr, cpd, args.device)
        paths.append(save(res, args.out))
        m = res["meta"]
        summary.append({k: m[k] for k in ("npr", "nx", "ny", "cells_per_d", "t_final", "steps",
                                          "runtime_s", "converged")}
                       | {"final_core_avg_rel_change": m["history"][-1]["core_avg_rel_change"]})
        print(f"NPR {npr:g}: {m['nx']}x{m['ny']} cells, {m['steps']} steps, {m['runtime_s']:.1f} s, "
              f"converged={m['converged']} -> {paths[-1]}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    fig_path = ROOT / "results" / "figures" / "jet_overview.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    overview_figure(paths, fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
