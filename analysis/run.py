"""Measure Mach-disk position and shock-cell spacing on the synthetic Schlieren
images, with uncertainties, and compare with Ashkenas-Sherman / Prandtl-Pack.

Mach-disk position = mean over the 8 short-exposure frames (each with its own
camera noise), i.e. the time-mean position, as from a high-speed sequence.
The simulated Mach disk oscillates axially (by up to ~0.8 D peak-to-peak at
NPR 80-100); in a long-exposure (time-averaged) image the brightest point of
such a smeared, sometimes bimodal disk can sit at one end of its travel, so
the long-exposure detection is reported only as a secondary value.
Uncertainty (added in quadrature):
  noise   std over N_MC camera-noise realisations of the time-averaged image
  jitter  std over the frames: the oscillation itself (conservative: the full
          spread, not the standard error of the mean)
  grid    one CFD cell (D/40): a captured shock is ~2 cells wide, so the image
          peak can sit anywhere within about a cell of the true discontinuity

Shock-cell spacing = distance from the Mach disk to the first downstream
compression band (searched in a band as wide as the jet, see shocks.py),
measured per frame: median over frames, jitter = 1.4826 * MAD. Reported only
if found in >= 75 % of frames with relative jitter < 0.3; else "not detected".

A CFD-field cross-check of the Mach-disk position (steepest axial density rise
on the axis, averaged over the same frames: the field quantity a vertical-knife
Schlieren responds to) is computed to validate the optical pipeline. It is labelled
as such and is not used for any result.

    python -m analysis.run [--fast]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from flow.jet import FAST_NPRS, FULL_NPRS, load
from optics.render import load_images
from .shocks import CameraModel, DetectorConfig, detect
from .theory import AS_COEFF, ashkenas_sherman, fully_expanded_mach, pack_spacing

ROOT = Path(__file__).resolve().parents[1]
N_MC = 20
DET = DetectorConfig()
CELL_DET = DET
CAM = CameraModel()
AS_RANGE_MIN = 15  # Ashkenas-Sherman data cover p0/pa >= ~15


def cfd_mach_disk(flow: dict) -> float:
    """Cross-check only: frame-averaged position of the steepest density rise on the axis,
    searched between x = 0.3 D and 1.5 D past the point where the mean axis Mach number drops below 1."""
    x = flow["x"]
    x_sub = x[int(np.argmax((x > 0.3) & (flow["mach"][0] < 1.0)))]
    window = (x > 0.3) & (x < x_sub + 1.5)
    pos = [x[np.argmax(np.where(window, np.gradient(r[0], x), -np.inf))] for r in flow["rho_frames"]]
    return float(np.mean(pos))


def first_spacing(det) -> float | None:
    s = det.spacings
    return float(s[0]) if len(s) else None


def analyse(npr: float, seed: int = 0) -> dict:
    o = load_images(ROOT / "results" / "optics" / f"images_npr{npr:g}.npz")
    x, y = o["x_over_D"], o["y_over_D"]
    dx = float(x[1] - x[0])
    rng = np.random.default_rng(seed)

    mc = [detect(CAM(o["schlieren_x"], rng), x, y, CELL_DET) for _ in range(N_MC)]
    fr = [detect(CAM(f, rng), x, y, CELL_DET) for f in o["schlieren_x_frames"]]
    md_mc = np.array([d.x_md for d in mc if d.x_md is not None])
    md_fr = np.array([d.x_md for d in fr if d.x_md is not None])
    s_mc = np.array([s for d in mc if (s := first_spacing(d)) is not None])
    s_fr = np.array([s for d in fr if (s := first_spacing(d)) is not None])

    res = {"npr": npr, "p_a_Pa": o["meta"]["p_a"], "pixel_D": dx,
           "md_detect_rate_mc": len(md_mc) / N_MC, "md_detect_rate_frames": len(md_fr) / len(fr)}
    res["cells_long_exposure"] = mc[0].cells
    if len(md_mc):
        s_noise = float(md_mc.std(ddof=1)) if len(md_mc) > 1 else 0.0
        s_jit = float(md_fr.std(ddof=1)) if len(md_fr) > 1 else 0.0
        res |= {"x_md": float(md_fr.mean()), "x_md_long_exposure": float(np.median(md_mc)),
                "x_md_frames": md_fr.tolist(),
                "sigma_noise": s_noise, "sigma_jitter": s_jit, "sigma_grid": dx,
                "sigma_total": float(np.sqrt(s_noise ** 2 + s_jit ** 2 + dx ** 2)),
                "snr_md": float(np.median([d.prominence_md / d.noise for d in mc if d.x_md is not None]))}
    # Shock cells move from frame to frame, and single frames occasionally lock on a
    # neighbouring band, so robust statistics are used: median and 1.4826*MAD.
    rate_fr = len(s_fr) / len(fr)
    res |= {"cell_detect_rate_frames": rate_fr, "cell_spacing_frames": s_fr.tolist(),
            "cell_spacing_long_exposure": float(np.median(s_mc)) if len(s_mc) else None}
    ok = False
    if rate_fr >= 0.75:
        med = float(np.median(s_fr))
        jit = float(1.4826 * np.median(np.abs(s_fr - med)))
        sn = float(s_mc.std(ddof=1)) if len(s_mc) > 1 else 0.0
        ok = jit / med < 0.3
        res |= {"cell_jitter_rel": jit / med}
    if ok:
        res |= {"cell_spacing": med, "cell_spacing_sigma": float(np.sqrt(jit ** 2 + sn ** 2 + 2 * dx ** 2))}
    else:
        res |= {"cell_spacing": None, "cell_spacing_sigma": None}

    flow = load(ROOT / "results" / "flow" / f"jet_npr{npr:g}.npz")
    res |= {"x_md_cfd_crosscheck": cfd_mach_disk(flow),
            "ashkenas_sherman": float(ashkenas_sherman(npr)),
            "pack_spacing": float(pack_spacing(npr)), "M_j": float(fully_expanded_mach(npr))}
    return res


def fit_coefficient(rows: list[dict], npr_min: float) -> dict:
    """Weighted least squares for x_md = C sqrt(NPR)."""
    r = [q for q in rows if q.get("x_md") is not None and q["npr"] >= npr_min]
    s = np.sqrt([q["npr"] for q in r])
    xm = np.array([q["x_md"] for q in r])
    w = 1.0 / np.array([q["sigma_total"] for q in r]) ** 2
    C = float(np.sum(w * s * xm) / np.sum(w * s * s))
    sC = float(1.0 / np.sqrt(np.sum(w * s * s)))
    chi2 = float(np.sum(w * (xm - C * s) ** 2))
    dof = len(r) - 1
    # Inflate the error if the scatter exceeds the stated uncertainties.
    sC_scaled = sC * max(1.0, np.sqrt(chi2 / dof)) if dof > 0 else sC
    return {"npr_min": npr_min, "n": len(r), "C": C, "sigma_C": sC_scaled, "chi2_red": chi2 / dof if dof else None}


# ------------------------------------------------------------------ figures --
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def mach_disk_figure(rows, fit, path):
    plt = _plt()
    r = [q for q in rows if q.get("x_md") is not None]
    npr = np.array([q["npr"] for q in r], float)
    xm = np.array([q["x_md"] for q in r])
    sx = np.array([q["sigma_total"] for q in r])
    cfd = np.array([q["x_md_cfd_crosscheck"] for q in r])
    nn = np.linspace(min(npr.min(), 4), npr.max() * 1.05, 200)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    a.plot(nn, ashkenas_sherman(nn), "k-", lw=1.2, label=f"Ashkenas–Sherman  {AS_COEFF}$\\sqrt{{NPR}}$")
    a.plot(nn, fit["C"] * np.sqrt(nn), "C0--", lw=1,
           label=f"fit (NPR ≥ {fit['npr_min']}): {fit['C']:.3f} ± {fit['sigma_C']:.3f} $\\sqrt{{NPR}}$")
    a.errorbar(npr, xm, yerr=sx, fmt="o", color="C3", ms=5, capsize=3, label="Schlieren measurement (±1σ)")
    a.plot(npr, cfd, "s", mfc="none", color="0.4", ms=6, label="CFD-field cross-check (not a result)")
    a.axvspan(nn.min(), AS_RANGE_MIN, color="0.92", zorder=0)
    a.text(nn.min() + 0.5, a.get_ylim()[1] * 0.9 if a.get_ylim()[1] > 0 else 1, "below A–S data range", fontsize=8)
    a.set_ylabel("Mach disk position $x_{md}/D$")
    a.legend(fontsize=8, loc="lower right")
    b.axhline(1.0, color="k", lw=1)
    b.errorbar(npr, xm / ashkenas_sherman(npr), yerr=sx / ashkenas_sherman(npr), fmt="o", color="C3", capsize=3)
    b.plot(npr, cfd / ashkenas_sherman(npr), "s", mfc="none", color="0.4")
    b.axvspan(nn.min(), AS_RANGE_MIN, color="0.92", zorder=0)
    b.set_xlabel("nozzle pressure ratio  NPR = $p_0/p_a$")
    b.set_ylabel("measured / A–S")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cell_figure(rows, path):
    plt = _plt()
    npr_all = np.array([q["npr"] for q in rows], float)
    nn = np.linspace(npr_all.min(), npr_all.max(), 200)
    fig, a = plt.subplots(figsize=(7, 4), constrained_layout=True)
    a.plot(nn, pack_spacing(nn), "k-", lw=1, label="Prandtl–Pack 1.306$\\sqrt{M_j^2-1}\\,D_j$ (weakly underexpanded)")
    det = [q for q in rows if q["cell_spacing"] is not None]
    if det:
        a.errorbar([q["npr"] for q in det], [q["cell_spacing"] for q in det],
                   yerr=[q["cell_spacing_sigma"] for q in det], fmt="o", color="C3", capsize=3,
                   label="Schlieren: Mach disk → first downstream compression")
    nd = [q["npr"] for q in rows if q["cell_spacing"] is None]
    for n in nd:
        a.axvline(n, color="0.8", lw=0.8, zorder=0)
    if nd:
        a.plot([], [], color="0.8", label="not reliably detected")
    a.set_xlabel("NPR")
    a.set_ylabel("shock-cell spacing / D")
    a.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def headline_figure(npr, row, path, seed=1):
    """density | Schlieren (camera model) | detected shocks."""
    plt = _plt()
    o = load_images(ROOT / "results" / "optics" / f"images_npr{npr:g}.npz")
    flow = load(ROOT / "results" / "flow" / f"jet_npr{npr:g}.npz")
    x, y = o["x_over_D"], o["y_over_D"]
    img = CAM(o["schlieren_x"], np.random.default_rng(seed))
    det = detect(img, x, y, CELL_DET)
    keep = flow["x"] <= x[-1] + 1e-9
    rho = flow["rho"][:, keep]
    rho_m = np.concatenate([rho[::-1], rho], 0)
    dx = x[1] - x[0]
    ext = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dx / 2, y[-1] + dx / 2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    from matplotlib.colors import LogNorm
    from optics.gas import density
    im = ax[0].imshow(rho_m * density(o["meta"]["p_a"]), origin="lower", extent=ext, cmap="viridis", norm=LogNorm())
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="density [kg/m³]")
    ax[0].set_title("CFD density (time-averaged)")
    ax[1].imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=2)
    ax[1].set_title("synthetic Schlieren (vertical knife, 8-bit camera, 1 % noise)")
    ax[2].imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=2)
    h = 0.6 * y.max()
    ax[2].axhspan(-DET.band_D, DET.band_D, color="C0", alpha=0.25, lw=0)
    ax[2].vlines(row["x_md"], -h, h, color="C3", lw=2, label=f"Mach disk {row['x_md']:.2f} ± {row['sigma_total']:.2f} D")
    ax[2].vlines(row["ashkenas_sherman"], -h, h, color="lime", lw=1.5, ls="--",
                 label=f"Ashkenas–Sherman {row['ashkenas_sherman']:.2f} D")
    for c in det.cells:
        ax[2].vlines(c, -h * 0.6, h * 0.6, color="orange", lw=1.2)
    if det.cells:
        ax[2].plot([], [], color="orange", label="downstream compressions")
    ax[2].set_title(f"detected shocks, NPR {npr:g}, p$_a$ = {o['meta']['p_a']/100:.0f} mbar")
    ax[2].legend(fontsize=8, loc="lower right")
    for a in ax:
        a.set_xlabel("x/D")
    ax[0].set_ylabel("y/D")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    nprs = FAST_NPRS if args.fast else FULL_NPRS
    rows = [analyse(n) for n in nprs]
    fits = {"as_range": fit_coefficient(rows, AS_RANGE_MIN if not args.fast else 0),
            "all": fit_coefficient(rows, 0)}
    rel = [q["x_md"] / q["ashkenas_sherman"] - 1 for q in rows if q.get("x_md") and q["npr"] >= AS_RANGE_MIN]
    cc = [q["x_md"] / q["x_md_cfd_crosscheck"] - 1 for q in rows if q.get("x_md")]
    out = {"detector": asdict(CELL_DET), "camera": asdict(CAM), "n_mc": N_MC, "rows": rows, "fits": fits,
           "mean_rel_dev_from_AS_in_range": float(np.mean(rel)) if rel else None,
           "max_abs_rel_dev_from_AS_in_range": float(np.max(np.abs(rel))) if rel else None,
           "max_abs_rel_dev_image_vs_cfd": float(np.max(np.abs(cc))) if cc else None}
    d = ROOT / "results" / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    (d / "shock_metrics.json").write_text(json.dumps(out, indent=2))
    figs = ROOT / "results" / "figures"
    mach_disk_figure(rows, fits["as_range"], figs / "mach_disk_vs_theory.png")
    cell_figure(rows, figs / "shock_cells.png")
    h = 30 if 30 in nprs else nprs[-1]
    headline_figure(h, next(q for q in rows if q["npr"] == h), figs / "headline.png")

    print(f"{'NPR':>5} {'x_md':>7} {'±σ':>6} {'(noise/jitter/grid)':>22} {'long-exp':>8} {'A-S':>6} {'CFD xchk':>8} {'cell':>6} {'±σ':>5} {'Pack':>5}")
    for q in rows:
        cs = f"{q['cell_spacing']:6.2f} {q['cell_spacing_sigma']:5.2f}" if q["cell_spacing"] else f"{'n/d':>6} {'':>5}"
        print(f"{q['npr']:5g} {q['x_md']:7.3f} {q['sigma_total']:6.3f} "
              f"({q['sigma_noise']:.3f}/{q['sigma_jitter']:.3f}/{q['sigma_grid']:.3f}) {q['x_md_long_exposure']:8.3f} {q['ashkenas_sherman']:6.2f} "
              f"{q['x_md_cfd_crosscheck']:8.3f} {cs} {q['pack_spacing']:5.2f}")
    for k, f in fits.items():
        print(f"fit {k}: C = {f['C']:.4f} ± {f['sigma_C']:.4f} (n={f['n']}, chi2_red={f['chi2_red']})")
    print(f"mean rel. deviation from A-S (NPR>={AS_RANGE_MIN}): {out['mean_rel_dev_from_AS_in_range']}")
    print(f"max |image - CFD| / CFD: {out['max_abs_rel_dev_image_vs_cfd']}")


if __name__ == "__main__":
    main()
