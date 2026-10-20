"""BOS displacement between a reference and a flow image.

Two methods:

* Cross-correlation (PIV-style): the images are cut into interrogation
  windows (default 24 px, 50 % overlap). For each window the FFT
  cross-correlation R(d) = sum a(x) b(x + d) of the mean-subtracted
  reference (a) and flow (b) windows peaks at the local shift d. Sub-pixel
  position from a three-point Gaussian fit in x and in y. Spurious vectors
  are flagged by the normalised median test (Westerweel & Scarano 2005) and
  replaced by the local median. A single pass underestimates shifts (pairs
  are lost at window edges), so a few passes of iterative image deformation
  follow: the flow image is warped back by the current (interpolated)
  estimate and the residual shift is added (Scarano 2002).
* Dense optical flow: OpenCV's Farneback polynomial-expansion method,
  one vector per pixel.

Convention: displacement = position in the flow image minus position in the
reference, in pixels, x to the right, y downwards (image rows).

A BOS displacement is the apparent shift of the background seen through the
flow: Delta = Z_D * eps on the background, with eps the ray deflection (see
optics/imaging.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter


@dataclass
class CCConfig:
    window: int = 24           # [px]; must stay small against the flow features (plume width)
    step: int = 12
    median_eps: float = 0.1     # [px] noise level in the normalised median test
    median_thresh: float = 2.0
    min_peak: float = 0.3       # normalised correlation peak below this -> invalid
    passes: int = 3             # 1 = plain correlation, >1 adds image-deformation passes


def _gauss3(cm, c0, cp):
    """Three-point Gaussian peak offset (falls back to parabolic if values are non-positive)."""
    if cm > 0 and c0 > 0 and cp > 0:
        lm, l0, lp = np.log(cm), np.log(c0), np.log(cp)
        den = lm - 2 * l0 + lp
        return 0.5 * (lm - lp) / den if den < 0 else 0.0
    den = cm - 2 * c0 + cp
    return 0.5 * (cm - cp) / den if den < 0 else 0.0


def _remap(img, map_x, map_y):
    import cv2
    return cv2.remap(img.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _to_pixels(xc, yc, f, shape):
    """Bilinear interpolation of a window-grid field to every pixel (edge values held)."""
    from scipy.interpolate import RegularGridInterpolator
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    itp = RegularGridInterpolator((yc, xc), f, bounds_error=False, fill_value=None)
    pts = np.stack([np.clip(Y, yc[0], yc[-1]), np.clip(X, xc[0], xc[-1])], -1)
    return itp(pts)


def cross_correlation(ref: np.ndarray, cur: np.ndarray, cfg: CCConfig = CCConfig()):
    """Window-based displacement with iterative image deformation.
    Returns (xc, yc, dx, dy, peak, valid) on the window grid."""
    xc, yc, dx, dy, peak, valid = _cc_pass(ref, cur, cfg)
    if cfg.passes > 1:
        Y, X = np.mgrid[0:ref.shape[0], 0:ref.shape[1]].astype(np.float64)
        for _ in range(cfg.passes - 1):
            # cur_def(x) = cur(x + d(x)) undoes the current estimate; correlate the remainder.
            cur_def = _remap(cur, X + _to_pixels(xc, yc, dx, ref.shape), Y + _to_pixels(xc, yc, dy, ref.shape))
            _, _, rx, ry, peak, valid = _cc_pass(ref, cur_def, cfg)
            dx, dy = dx + rx, dy + ry
    return xc, yc, dx, dy, peak, valid


def _cc_pass(ref: np.ndarray, cur: np.ndarray, cfg: CCConfig):
    w, s = cfg.window, cfg.step
    A = sliding_window_view(ref.astype(np.float32), (w, w))[::s, ::s]
    B = sliding_window_view(cur.astype(np.float32), (w, w))[::s, ::s]
    ny, nx = A.shape[:2]
    A = A - A.mean(axis=(2, 3), keepdims=True)
    B = B - B.mean(axis=(2, 3), keepdims=True)
    norm = np.sqrt((A ** 2).sum(axis=(2, 3)) * (B ** 2).sum(axis=(2, 3))) + 1e-12
    R = np.fft.irfft2(np.conj(np.fft.rfft2(A)) * np.fft.rfft2(B), s=(w, w))
    R = np.fft.fftshift(R, axes=(2, 3)) / norm[..., None, None]
    flat = R.reshape(ny, nx, -1)
    k = flat.argmax(-1)
    py, px = np.unravel_index(k, (w, w))
    peak = flat.max(-1)
    dx = np.zeros((ny, nx), np.float64)
    dy = np.zeros((ny, nx), np.float64)
    for j in range(ny):
        for i in range(nx):
            r, yy, xx = R[j, i], py[j, i], px[j, i]
            if 0 < xx < w - 1 and 0 < yy < w - 1:
                dx[j, i] = xx - w // 2 + _gauss3(r[yy, xx - 1], r[yy, xx], r[yy, xx + 1])
                dy[j, i] = yy - w // 2 + _gauss3(r[yy - 1, xx], r[yy, xx], r[yy + 1, xx])
            else:
                dx[j, i] = dy[j, i] = np.nan
    valid = (peak >= cfg.min_peak) & np.isfinite(dx)
    dx, dy, valid = replace_outliers(dx, dy, valid, cfg)
    yc = np.arange(ny) * s + (w - 1) / 2
    xc = np.arange(nx) * s + (w - 1) / 2
    return xc, yc, dx, dy, peak, valid


def replace_outliers(dx, dy, valid, cfg: CCConfig):
    """Normalised median test on the 3x3 neighbourhood; replace flagged vectors by the median."""
    dx, dy = dx.copy(), dy.copy()
    for comp in (dx, dy):
        comp[~valid] = np.nan
    flagged = ~valid
    for comp in (dx, dy):
        filled = np.where(np.isnan(comp), np.nanmedian(comp), comp)
        med = median_filter(filled, size=3, mode="nearest")
        res = np.abs(filled - med)
        rmed = median_filter(res, size=3, mode="nearest")
        flagged |= res / (rmed + cfg.median_eps) > cfg.median_thresh
    for comp in (dx, dy):
        filled = np.where(np.isnan(comp), np.nanmedian(comp), comp)
        med = median_filter(np.where(flagged, np.nan, filled), size=3, mode="nearest")
        med = np.where(np.isnan(med), median_filter(filled, size=3, mode="nearest"), med)
        comp[:] = np.where(flagged, med, filled)
    return dx, dy, ~flagged


def optical_flow(ref: np.ndarray, cur: np.ndarray, winsize: int = 21):
    """Farneback dense flow (dx, dy) per pixel."""
    import cv2
    # Float images are taken to be on a 0..1 scale (noise may push them slightly past 1).
    to8 = lambda im: np.clip(np.round(im * 255 if im.dtype.kind == "f" and im.max() < 2 else im),
                             0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(to8(ref), to8(cur), None, pyr_scale=0.5, levels=3,
                                        winsize=winsize, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    return flow[..., 0].astype(np.float64), flow[..., 1].astype(np.float64)


def displacement(ref, cur, method: str = "cc", cc: CCConfig = CCConfig()) -> dict:
    """Unified interface: returns dict with grid coordinates and dx, dy [px]."""
    if method == "cc":
        xc, yc, dx, dy, peak, valid = cross_correlation(ref, cur, cc)
        return {"x": xc, "y": yc, "dx": dx, "dy": dy, "valid_fraction": float(valid.mean())}
    if method == "flow":
        dx, dy = optical_flow(ref, cur)
        return {"x": np.arange(ref.shape[1], dtype=float), "y": np.arange(ref.shape[0], dtype=float),
                "dx": dx, "dy": dy, "valid_fraction": 1.0}
    raise ValueError(method)


def remove_global_motion(d: dict) -> dict:
    """Subtract the field median: camera shake / tripod drift moves the whole background,
    a plume only a part of it (assumes the flow covers well under half the field)."""
    d = dict(d)
    d["global_shift"] = (float(np.median(d["dx"])), float(np.median(d["dy"])))
    d["dx"] = d["dx"] - d["global_shift"][0]
    d["dy"] = d["dy"] - d["global_shift"][1]
    return d
