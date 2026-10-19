"""Render synthetic shadowgraph / knife-edge Schlieren / BOS images of the CFD jets.

Physical setup (all choices in PHYS / OpticsConfig):
* nozzle exit diameter D = 1 mm, air at T = 293 K, Gladstone-Dale K = 2.26e-4 m^3/kg;
* NPR sweep at a fixed reservoir pressure p0 = 2 bar, so the ambient pressure
  is p_a = p0 / NPR (400 mbar ... 20 mbar), like pumping a chamber down
  with a fixed feed pressure;
* the inviscid CFD fields are non-dimensional and self-similar in pressure, so
  the dimensional density is rho = rho_nd * rho_a(p_a). This makes the "lower
  ambient pressure" study exact within the Euler model: the optical signal
  simply scales with p_a at fixed NPR (until the Schlieren response saturates).

Outputs
  results/optics/images_npr*.npz  images for stage 3 (time-averaged + 8 snapshots)
  results/optics/contrast_vs_pressure.json
  results/figures/optics_sweep.png, optics_contrast.png

    python -m optics.render [--fast] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from flow.jet import FAST_NPRS, FULL_NPRS, load
from .abel import project
from .gas import K_AIR, T_ROOM, density, mean_free_path
from .imaging import OpticsConfig, bos_displacement, schlieren, shadowgraph

ROOT = Path(__file__).resolve().parents[1]

PHYS = {
    "D": 1e-3,                 # nozzle exit diameter [m]
    "T": T_ROOM,               # ambient = stagnation temperature [K]
    "K": K_AIR,                # Gladstone-Dale constant [m^3/kg]
    "p0_sweep": 2e5,           # reservoir pressure of the NPR sweep [Pa]
    "contrast_npr": 30,        # NPR of the ambient-pressure study
    "bos_pixel": 10e-6,        # BOS camera pixel projected on the background [m]
    "intensity_noise": 0.01,   # 1 % rms: shot noise of a ~1e4 e- pixel
    "bos_noise_px": 0.1,       # typical cross-correlation displacement precision [px]
    "fov_margin_D": 1.0,       # crop this much at the CFD outflow boundary (edge artefacts)
}
CFG = OpticsConfig(source_halfwidth=0.5e-3, max_upsample=16)


def opd_from_rho(rho_nd: np.ndarray, p_a: float, dr: float) -> tuple[np.ndarray, np.ndarray]:
    """Abel-projected OPD [m] of the non-dimensional density field at ambient pressure p_a."""
    drho = (rho_nd - 1.0) * density(p_a, PHYS["T"])  # relative to ambient: no truncation edge
    return project(PHYS["K"] * drho, dr)


def render_case(d: dict, p_a: float, cfg: OpticsConfig, device, frames: bool = True) -> dict:
    dr = PHYS["D"] / d["meta"]["cells_per_d"]
    y, opd = opd_from_rho(d["rho"], p_a, dr)
    out = {"x_over_D": d["x"], "y_over_D": y / PHYS["D"], "opd": opd.astype(np.float32)}
    sx, bg_x, info_x = schlieren(opd, dr, cfg, device)
    sy, bg_y, _ = schlieren(opd, dr, replace(cfg, knife_angle_deg=90.0), device)
    sh, info_sh = shadowgraph(opd, dr, cfg, device)
    bx, by = bos_displacement(opd, dr, cfg)
    out |= {"schlieren_x": sx / bg_x, "schlieren_y": sy / bg_y, "shadow": sh,
            "bos_dx_px": (bx / PHYS["bos_pixel"]).astype(np.float32),
            "bos_dy_px": (by / PHYS["bos_pixel"]).astype(np.float32)}
    if frames and "rho_frames" in d:
        fr = []
        for rho_f in d["rho_frames"]:
            _, opd_f = opd_from_rho(rho_f, p_a, dr)
            s, bg, _ = schlieren(opd_f, dr, cfg, device)
            fr.append(s / bg)
        out["schlieren_x_frames"] = np.stack(fr).astype(np.float32)
    # Field of view: drop the last fov_margin_D next to the outflow boundary,
    # where the boundary condition and the FFT padding leave an edge artefact.
    keep = d["x"] <= d["x"][-1] - PHYS["fov_margin_D"]
    for k, v in list(out.items()):
        if k != "y_over_D" and isinstance(v, np.ndarray) and v.shape[-1] == keep.size:
            out[k] = v[..., keep]
    out["meta"] = {"p_a": p_a, "npr": d["meta"]["npr"], "p0": p_a * d["meta"]["npr"],
                   "pixel_m": dr, "phys": PHYS, "optics": asdict(cfg),
                   "sampling_schlieren": info_x, "sampling_shadow": info_sh,
                   "images": "schlieren_* normalised by the no-flow background I_bg; shadow is I/I0; "
                             "BOS displacement in background-plane pixels"}
    return out


def save_images(out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in out.items() if k != "meta"},
                        meta=json.dumps(out["meta"], default=float))


def load_images(path: Path) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


# ----------------------------------------------------- contrast vs pressure --
def signal_metrics(out: dict) -> dict:
    """Peak signal (99th percentile) of each diagnostic over the image."""
    p99 = lambda a: float(np.percentile(np.abs(a), 99))
    return {"schlieren": p99(out["schlieren_x"] - 1.0),
            "shadow": p99(out["shadow"] - 1.0),
            "bos_px": p99(np.hypot(out["bos_dx_px"], out["bos_dy_px"]))}


def threshold_pressure(p: np.ndarray, s: np.ndarray, thr: float) -> float | None:
    """Lowest ambient pressure at which signal s(p) still exceeds thr (log interpolation)."""
    order = np.argsort(p)
    p, s = p[order], s[order]
    above = s >= thr
    if not above.any():
        return None
    i = np.argmax(above)  # first pressure (from low) with signal above threshold
    if i == 0:
        return float(p[0])
    lp = np.interp(np.log(thr), np.log([s[i - 1], s[i]]), np.log([p[i - 1], p[i]]))
    return float(np.exp(lp))


def contrast_study(device, fast: bool) -> dict:
    npr = PHYS["contrast_npr"] if not fast else FAST_NPRS[-1]
    d = load(ROOT / "results" / "flow" / f"jet_npr{npr:g}.npz")
    pressures = np.logspace(5, 1, 9 if fast else 17)  # 1 bar ... 0.1 mbar
    systems = {"standard (w = 0.5 mrad)": CFG,
               "high-sensitivity (w = 0.05 mrad)": replace(CFG, source_halfwidth=0.05e-3)}
    res = {"npr": npr, "p_a": pressures.tolist(), "systems": {}, "examples": {}}
    for name, cfg in systems.items():
        rows = []
        for p_a in pressures:
            out = render_case(d, p_a, cfg, device, frames=False)
            rows.append(signal_metrics(out) | {"undersampled": out["meta"]["sampling_schlieren"]["undersampled"],
                                               "upsample": out["meta"]["sampling_schlieren"]["upsample"]})
            if name.startswith("standard") and np.isclose(np.log10(p_a), [5, 3, 2]).any():
                res["examples"][f"{p_a:.0f}"] = out["schlieren_x"]
        res["systems"][name] = rows
    # Detection limits.
    lim = {}
    for name, rows in res["systems"].items():
        lim[name] = {
            "schlieren_min_p_a_Pa": threshold_pressure(pressures, np.array([r["schlieren"] for r in rows]),
                                                       PHYS["intensity_noise"]),
            "shadow_min_p_a_Pa": threshold_pressure(pressures, np.array([r["shadow"] for r in rows]),
                                                    PHYS["intensity_noise"]),
            "bos_min_p_a_Pa": threshold_pressure(pressures, np.array([r["bos_px"] for r in rows]),
                                                 PHYS["bos_noise_px"]),
        }
    res["detection_limits"] = lim
    # Pressure below which the *ambient* gas is no longer a continuum on the scale of D.
    res["p_a_at_Kn_0p01_Pa"] = mean_free_path(1e5, PHYS["T"]) * 1e5 / (0.01 * PHYS["D"])
    return res


# ------------------------------------------------------------------ figures --
def _mirror_extent(o):
    x, y = o["x_over_D"], o["y_over_D"]
    dx = x[1] - x[0]
    return [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dx / 2, y[-1] + dx / 2]


def sweep_figure(outs: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("shadow", "shadowgraph  I/I$_0$", "gray", (0, 2)),
            ("schlieren_x", "Schlieren, vertical knife  I/I$_{bg}$", "gray", (0, 2)),
            ("schlieren_y", "Schlieren, horizontal knife  I/I$_{bg}$", "gray", (0, 2)),
            ("bos", "BOS |displacement| [px]", "magma", None)]
    fig, axes = plt.subplots(len(outs), 4, figsize=(15, 2.7 * len(outs)), constrained_layout=True, squeeze=False)
    for row, o in zip(axes, outs):
        ext = _mirror_extent(o)
        for ax, (key, title, cmap, lim) in zip(row, cols):
            img = np.hypot(o["bos_dx_px"], o["bos_dy_px"]) if key == "bos" else o[key]
            vmin, vmax = lim if lim else (0, np.percentile(img, 99.5))
            im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
            if key == "bos":
                fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x/D")
        m = o["meta"]
        row[0].set_ylabel(f"NPR {m['npr']:g}, p$_a$ = {m['p_a']/100:.0f} mbar\ny/D")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def contrast_figure(res: dict, path: Path, ext) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.array(res["p_a"]) / 100.0  # mbar
    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :2])
    styles = {"schlieren": ("Schlieren |ΔI/I$_{bg}$|", "C0"), "shadow": ("shadowgraph |ΔI/I$_0$|", "C1"),
              "bos_px": ("BOS |Δ| [px]", "C2")}
    for (name, rows), ls in zip(res["systems"].items(), ["-", "--"]):
        for key, (lab, c) in styles.items():
            if key != "schlieren" and ls == "--":
                continue  # shadowgraph/BOS do not depend on the Schlieren source width
            ax.loglog(p, [r[key] for r in rows], ls, color=c, marker="o", ms=3,
                      label=f"{lab}" + (f", {name}" if key == "schlieren" else ""))
    ax.axhline(PHYS["intensity_noise"], color="k", lw=0.8, ls=":")
    ax.text(p.max() * 0.9, PHYS["intensity_noise"] * 1.15, "1 % intensity noise", fontsize=8)
    ax.axhline(PHYS["bos_noise_px"], color="C2", lw=0.8, ls=":")
    ax.text(p.max() * 0.9, PHYS["bos_noise_px"] * 1.15, "0.1 px BOS precision", fontsize=8, color="C2")
    pk = res["p_a_at_Kn_0p01_Pa"] / 100
    ax.axvspan(p.min(), pk, color="0.85", zorder=0)
    ax.text(pk * 0.85, ax.get_ylim()[1] * 0.2, "ambient Kn > 0.01:\nEuler model questionable", fontsize=8)
    ax.set_xlabel("ambient pressure p$_a$ [mbar]")
    ax.set_ylabel("peak signal (99th percentile)")
    ax.set_title(f"Signal vs ambient pressure, NPR {res['npr']:g} jet, D = 1 mm")
    ax.legend(fontsize=8, loc="lower left")
    ax.invert_xaxis()
    axs = fig.add_subplot(gs[0, 2])
    axs.axis("off")
    lines = ["Lowest detectable p_a (signal > noise):"]
    for name, lim in res["detection_limits"].items():
        lines.append(f"\n{name}")
        for k, v in lim.items():
            lines.append(f"  {k.split('_min')[0]:10s} {('%.2f mbar' % (v / 100)) if v else 'n/a'}")
    axs.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    for i, (k, img) in enumerate(sorted(res["examples"].items(), key=lambda kv: -float(kv[0]))):
        a = fig.add_subplot(gs[1, i])
        a.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=2)
        a.set_title(f"Schlieren (standard), p$_a$ = {float(k)/100:g} mbar", fontsize=9)
        a.set_xlabel("x/D")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    nprs = FAST_NPRS if args.fast else FULL_NPRS
    out_dir = ROOT / "results" / "optics"
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outs, summary = [], []
    for npr in nprs:
        d = load(ROOT / "results" / "flow" / f"jet_npr{npr:g}.npz")
        p_a = PHYS["p0_sweep"] / npr
        o = render_case(d, p_a, CFG, args.device)
        save_images(o, out_dir / f"images_npr{npr:g}.npz")
        outs.append(o)
        summary.append({"npr": npr, "p_a_Pa": p_a} | signal_metrics(o)
                       | {"upsample": o["meta"]["sampling_schlieren"]["upsample"]})
        print(f"NPR {npr:g}: p_a = {p_a/100:.1f} mbar, signals {signal_metrics(o)}")
    show = outs if args.fast else [o for o in outs if o["meta"]["npr"] in (5, 15, 30, 100)]
    sweep_figure(show, fig_dir / "optics_sweep.png")

    res = contrast_study(args.device, args.fast)
    ext = _mirror_extent(next(o for o in outs if o["meta"]["npr"] == res["npr"]))
    contrast_figure(res, fig_dir / "optics_contrast.png", ext)
    res.pop("examples")
    res["sweep"] = summary
    res["runtime_s"] = time.time() - t0
    (out_dir / "contrast_vs_pressure.json").write_text(json.dumps(res, indent=2))
    for name, lim in res["detection_limits"].items():
        print(name, {k: (None if v is None else round(v / 100, 3)) for k, v in lim.items()}, "[mbar]")
    print(f"ambient Kn = 0.01 at p_a = {res['p_a_at_Kn_0p01_Pa']/100:.2f} mbar; runtime {res['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
