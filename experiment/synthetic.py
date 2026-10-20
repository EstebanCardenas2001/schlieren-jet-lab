"""Synthetic BOS data with a known displacement field, for testing the pipeline.

* camera_background: a random-dot pattern as a camera sees it (dots ~2.5 px,
  slight optical blur).
* warp: the flow image is the background seen through the flow: the point that
  sits at x - d(x) on the background appears at x, i.e. I_flow(x) = I_ref(x - d(x)),
  so the measured displacement is +d (bilinear/bicubic resampling by OpenCV).
* plume_field: a heat-gun-like plume rising from the bottom of the frame. The
  projected index deficit is a Gaussian across the plume, widening with
  height and swaying sideways in time, and the displacement is proportional
  to its gradient, as in BOS (Delta = Z_D grad OPD). Its peak magnitude is
  given in pixels.
* write_video: reference frames, then flow frames, with sensor noise and small
  random camera jitter, saved losslessly (FFV1) so codec artefacts do not
  enter the test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .pattern import random_dots


def camera_background(h: int, w: int, dot_px: float = 2.5, blur: float = 0.5, seed: int = 0) -> np.ndarray:
    img = random_dots(w, h, dot_px, fill=0.4, seed=seed)
    return gaussian_filter(img, blur) if blur > 0 else img


def warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    import cv2
    h, w = img.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), X - dx.astype(np.float32), Y - dy.astype(np.float32),
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def plume_field(h: int, w: int, amp_px: float, t: float = 0.0, sway_px: float = 0.0, seed_phase: float = 0.0):
    """Displacement (dx, dy) of a plume rising from the bottom centre; max |d| = amp_px."""
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
    height = (h - Y) / h                         # 0 at the bottom (nozzle) ... 1 at the top
    s = w * (0.04 + 0.08 * height)               # plume half-width grows with height
    xc = w / 2 + sway_px * height * np.sin(2 * np.pi * (t + 0.7 * height) + seed_phase)
    G = -np.exp(-((X - xc) / s) ** 2) * np.clip(1.2 - height, 0, 1) / np.sqrt(s / s.min())
    gy, gx = np.gradient(G)
    m = np.hypot(gx, gy).max()
    return amp_px * gx / m, amp_px * gy / m


def write_video(path: Path, h=360, w=480, n_ref=20, n_flow=40, amp_px=1.0, sway_px=8.0,
                noise=0.01, jitter_px=0.05, fps=20, seed=0):
    """Write a lossless synthetic BOS recording; return the true time-mean displacement of the flow frames."""
    import cv2
    rng = np.random.default_rng(seed)
    bg = camera_background(h, w, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), False)
    assert vw.isOpened(), "FFV1 codec unavailable"
    true_dx, true_dy = np.zeros((h, w)), np.zeros((h, w))
    for k in range(n_ref + n_flow):
        jx, jy = rng.normal(0, jitter_px, 2)     # camera shake moves everything
        if k < n_ref:
            dx, dy = np.zeros((h, w)), np.zeros((h, w))
        else:
            dx, dy = plume_field(h, w, amp_px, t=(k - n_ref) / fps, sway_px=sway_px)
            true_dx += dx / n_flow
            true_dy += dy / n_flow
        frame = warp(bg, dx + jx, dy + jy) + rng.normal(0, noise, (h, w))
        vw.write(np.clip(np.round(frame * 255), 0, 255).astype(np.uint8))
    vw.release()
    return true_dx, true_dy
