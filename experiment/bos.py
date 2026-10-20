"""One command: record (webcam) or load (phone video) a BOS run -> displacement map.

This is a *subsonic* demonstration: the refractive-index field of a heat-gun
plume or a gas-duster jet, not a supersonic measurement.

    # webcam: 2 s reference (flow OFF), countdown, 5 s with flow ON
    python -m experiment.bos --camera 0 --ref-seconds 2 --flow-seconds 5 --name heatgun

    # phone / existing video: frames 0-29 reference, 60-end flow
    python -m experiment.bos --video run.mp4 --ref 0:30 --flow 60: --name heatgun

    # synthetic recording with a known plume (pipeline check, no hardware)
    python -m experiment.bos --synthetic

Processing: reference = mean of the reference frames (averaging lowers sensor
noise); every flow frame is compared with it (cross-correlation and/or
Farneback optical flow), camera shake is removed as the field median, and
the time-mean displacement and its rms fluctuation are reported. The noise
floor is measured on the reference frames themselves: single reference frames
vs the mean of the other reference frames (no flow in either). This is the
*random* error only; systematic errors (spatial averaging over the
interrogation window, out-of-focus blur of the flow) are not included. Its value divided by sqrt(N_flow)
estimates the noise of the time-mean field (assumes independent frame noise).

Outputs in results/experiment/<name>/: bos.npz, bos.png, bos.json
(+ recording.mkv when recording from a camera).

Tips for real footage: tripod; lock focus on the background, exposure and
white balance (phone "pro" mode / AE-AF lock); background 0.5-1 m behind the
flow; small aperture so the flow is not too blurred; highest bitrate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .processing import CCConfig, displacement, remove_global_motion

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- frame I/O --
def _parse_slice(text: str) -> slice:
    a, _, b = text.partition(":")
    return slice(int(a) if a else 0, int(b) if b else None)


def _gray(frame) -> np.ndarray:
    import cv2
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32) / 255.0


def iter_frames(path: Path, sl: slice, every: int = 1, roi=None):
    """Yield grayscale frames [0..1] with indices in slice `sl` (every n-th)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = n if sl.stop is None else min(sl.stop, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sl.start)
    for k in range(sl.start, stop):
        ok, fr = cap.read()
        if not ok:
            break
        if (k - sl.start) % every == 0:
            g = _gray(fr)
            if roi is not None:
                x0, y0, x1, y1 = roi
                g = g[y0:y1, x0:x1]
            yield g
    cap.release()


def record(camera: int, ref_seconds: float, flow_seconds: float, out: Path,
           width: int | None = None, height: int | None = None) -> tuple[Path, int, float]:
    """Record reference then flow frames from a webcam into a lossless video."""
    import cv2
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):  # let auto-exposure settle before the reference
        cap.read()
    ok, fr = cap.read()
    h, w = fr.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), False)
    print(f"camera {w}x{h} @ {fps:.0f} fps -> {out}")

    def grab(seconds):
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            ok, fr = cap.read()
            if not ok:
                break
            vw.write(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
            n += 1
        return n

    print("REFERENCE: keep the flow OFF and the camera still ...")
    n_ref = grab(ref_seconds)
    for k in (3, 2, 1):
        print(f"  turn the flow ON in {k} ...")
        grab(1.0)  # recorded (transition) but excluded from processing
    print("FLOW: recording ...")
    grab(flow_seconds)
    vw.release()
    cap.release()
    return out, n_ref, fps


# -------------------------------------------------------------- processing --
def process_video(path: Path, ref: slice, flow: slice, method: str = "cc", every: int = 1,
                  roi=None, cc: CCConfig = CCConfig(), n_noise: int = 5) -> dict:
    ref_frames = list(iter_frames(path, ref, 1, roi))
    if len(ref_frames) < 2:
        raise ValueError("need at least 2 reference frames")
    ref_mean = np.mean(ref_frames, axis=0)

    # Noise floor: single reference frames (no flow) against the reference mean.
    # Leave-one-out: the tested frame is excluded from the mean it is compared with.
    noise, n_r = [], len(ref_frames)
    for fr in ref_frames[:: max(1, n_r // n_noise)][:n_noise]:
        loo = (ref_mean * n_r - fr) / (n_r - 1)
        d = remove_global_motion(displacement(loo, fr, method, cc))
        noise.append(float(np.sqrt(np.mean(d["dx"] ** 2 + d["dy"] ** 2))))

    s_dx = s_dy = s2 = None
    n, shifts, x = 0, [], None
    for fr in iter_frames(path, flow, every, roi):
        d = remove_global_motion(displacement(ref_mean, fr, method, cc))
        if s_dx is None:
            s_dx, s_dy, s2 = np.zeros_like(d["dx"]), np.zeros_like(d["dy"]), np.zeros_like(d["dx"])
            x, y = d["x"], d["y"]
        s_dx += d["dx"]
        s_dy += d["dy"]
        s2 += d["dx"] ** 2 + d["dy"] ** 2
        shifts.append(d["global_shift"])
        n += 1
    if n == 0:
        raise ValueError("no flow frames in the requested range")
    mdx, mdy = s_dx / n, s_dy / n
    rms = np.sqrt(np.clip(s2 / n - mdx ** 2 - mdy ** 2, 0, None))  # temporal rms fluctuation
    mag = np.hypot(mdx, mdy)
    noise_frame = float(np.median(noise))
    noise_mean = noise_frame / np.sqrt(n)
    return {
        "x": x, "y": y, "dx": mdx, "dy": mdy, "mag": mag, "rms": rms, "ref": ref_mean,
        "stats": {"method": method, "n_ref": len(ref_frames), "n_flow": n,
                  "p99_mean_displacement_px": float(np.percentile(mag, 99)),
                  "max_mean_displacement_px": float(mag.max()),
                  "noise_floor_frame_px": noise_frame, "noise_floor_mean_px": noise_mean,
                  "snr_p99": float(np.percentile(mag, 99) / max(noise_mean, 1e-9)),
                  "camera_shake_rms_px": float(np.sqrt(np.mean(np.sum(np.square(shifts), 1))))},
    }


# ----------------------------------------------------------------- figures --
def figure(results: dict[str, dict], path: Path, truth=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(results)
    ncol = 2 + (1 if truth is not None else 0)
    fig, axes = plt.subplots(len(methods), ncol + 1, figsize=(4.2 * (ncol + 1), 3.6 * len(methods)),
                             constrained_layout=True, squeeze=False)
    vmax = max(np.percentile(r["mag"], 99.5) for r in results.values())
    if truth is not None:
        vmax = max(vmax, np.hypot(*truth).max())
    for row, m in zip(axes, methods):
        r = results[m]
        h, w = r["ref"].shape
        ext = [0, w, h, 0]
        row[0].imshow(r["ref"], cmap="gray", extent=ext)
        row[0].set_title("reference (mean)")
        ext_d = [r["x"][0], r["x"][-1], r["y"][-1], r["y"][0]]
        im = row[1].imshow(r["mag"], cmap="magma", extent=ext_d, vmin=0, vmax=vmax)
        sk = max(1, len(r["x"]) // 25)
        X, Y = np.meshgrid(r["x"][::sk], r["y"][::sk])
        row[1].quiver(X, Y, r["dx"][::sk, ::sk], -r["dy"][::sk, ::sk], color="c", scale=vmax * 30, width=0.003)
        row[1].set_title(f"{m}: time-mean |Δ| [px]\nnoise floor {r['stats']['noise_floor_mean_px']:.1e} px")
        fig.colorbar(im, ax=row[1], shrink=0.8)
        im = row[2].imshow(r["rms"], cmap="viridis", extent=ext_d)
        row[2].set_title(f"{m}: rms fluctuation [px]")
        fig.colorbar(im, ax=row[2], shrink=0.8)
        if truth is not None:
            im = row[3].imshow(np.hypot(*truth), cmap="magma", extent=ext, vmin=0, vmax=vmax)
            row[3].set_title("truth: time-mean |Δ| [px]")
            fig.colorbar(im, ax=row[3], shrink=0.8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_truth(r: dict, truth) -> dict:
    """RMS error of the time-mean field against the known truth (sampled at the vector positions)."""
    from scipy.ndimage import map_coordinates
    tdx, tdy = truth
    Y, X = np.meshgrid(r["y"], r["x"], indexing="ij")
    ex = r["dx"] - map_coordinates(tdx, [Y, X], order=1)
    ey = r["dy"] - map_coordinates(tdy, [Y, X], order=1)
    m = 8 if r["dx"].shape[0] > 100 else 0  # dense flow: ignore an 8 px border
    sl = (slice(m, -m or None), slice(m, -m or None))
    err = np.hypot(ex, ey)[sl]
    tmag = np.hypot(map_coordinates(tdx, [Y, X], order=1), map_coordinates(tdy, [Y, X], order=1))[sl]
    return {"rms_error_px": float(np.sqrt(np.mean(err ** 2))), "p99_error_px": float(np.percentile(err, 99)),
            "true_peak_px": float(tmag.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--camera", type=int)
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ref", default="0:30", help="reference frame range a:b (video)")
    ap.add_argument("--flow", default="60:", help="flow frame range a:b (video)")
    ap.add_argument("--ref-seconds", type=float, default=2.0)
    ap.add_argument("--flow-seconds", type=float, default=5.0)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--method", default="both", choices=["cc", "flow", "both"])
    ap.add_argument("--every", type=int, default=1, help="use every n-th flow frame")
    ap.add_argument("--roi", help="x0,y0,x1,y1 crop in pixels")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    truth = None
    if a.synthetic:
        from .synthetic import write_video
        name = a.name or "synthetic"
        out = ROOT / "results" / "experiment" / name
        video = out / "recording.mkv"
        truth = write_video(video)
        ref, flow = slice(0, 20), slice(20, None)
    elif a.camera is not None:
        name = a.name or time.strftime("run_%Y%m%d_%H%M%S")
        out = ROOT / "results" / "experiment" / name
        video, n_ref, fps = record(a.camera, a.ref_seconds, a.flow_seconds, out / "recording.mkv", a.width, a.height)
        ref, flow = slice(0, n_ref), slice(n_ref + int(round(3 * fps)), None)
    else:
        name = a.name or a.video.stem
        out = ROOT / "results" / "experiment" / name
        video, ref, flow = a.video, _parse_slice(a.ref), _parse_slice(a.flow)
    out.mkdir(parents=True, exist_ok=True)

    cc = CCConfig(window=a.window, step=a.window // 2)
    methods = ["cc", "flow"] if a.method == "both" else [a.method]
    results, summary = {}, {"video": str(video), "ref": [ref.start, ref.stop], "flow": [flow.start, flow.stop],
                            "roi": roi, "cc_window": a.window, "methods": {}}
    for m in methods:
        t0 = time.time()
        r = process_video(video, ref, flow, m, a.every, roi, cc)
        r["stats"]["runtime_s"] = time.time() - t0
        if truth is not None:
            r["stats"]["vs_truth"] = compare_truth(r, truth)
        results[m] = r
        summary["methods"][m] = r["stats"]
        print(m, json.dumps(r["stats"], indent=1))
    np.savez_compressed(out / "bos.npz", **{f"{m}_{k}": v for m, r in results.items()
                                            for k, v in r.items() if k != "stats"})
    (out / "bos.json").write_text(json.dumps(summary, indent=2))
    figure(results, out / "bos.png", truth)
    print(f"wrote {out / 'bos.png'} and {out / 'bos.json'}")


if __name__ == "__main__":
    main()
