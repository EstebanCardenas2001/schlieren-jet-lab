"""Shock detection on knife-edge Schlieren images (images only, no CFD fields).

With the knife-edge normal along +x (vertical knife), the image brightness is
I/I_bg - 1 ~ eps_x / w, with eps_x = d(OPD)/dx. A *compression* (density
rising along x, e.g. a normal shock) bends light towards +x and appears
bright; the expansion core of the jet appears dark. Along the jet axis an
underexpanded jet therefore shows a dark expansion region ended by a sharp
bright line, the Mach disk, followed by weaker bright/dark bands of the
downstream shock cells.

Algorithm
1. Camera model (optional): Gaussian read/shot noise plus 8-bit quantisation.
2. Centreline profile: mean over a thin band |y| <= band around the axis.
3. Noise level from the median absolute deviation of the image background
   (outer rows), scaled to the band average.
4. Bright peaks with prominence > k_sigma * noise (and > a fraction of the
   strongest peak), searched for x > x_min.
5. Mach disk = first such peak, sub-pixel position from a parabola through
   the peak and its neighbours.
6. Shock cells: behind the Mach disk the axis carries a subsonic mixing core;
   the downstream shock cells live in the supersonic annulus around it. In
   the projected image their compressions appear as bright bands spanning
   the jet width, so they are searched for in a *wide* band |y| <= 0.4 x_md
   (the barrel width scales with x_md), as peaks downstream of the Mach disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class DetectorConfig:
    band_D: float = 0.1          # half-height of the centreline band [D]
    smooth_px: float = 1.0       # Gaussian smoothing of the profile [px]
    k_sigma: float = 5.0         # peak prominence threshold in noise sigmas
    md_rel_prominence: float = 0.2    # Mach disk: prominence >= this * strongest peak
    cell_rel_prominence: float = 0.15  # shock cells: prominence >= this * Mach-disk prominence (wide band)
    cell_band_frac: float = 0.4  # wide band half-height / x_md
    cell_min_gap_D: float = 0.3  # first cell must lie this far downstream of the Mach disk
    x_min_D: float = 0.3         # ignore the nozzle-exit region
    background_frac: float = 0.15  # outer fraction of rows used for the noise estimate


@dataclass
class CameraModel:
    bg_counts: float = 128.0     # background level [counts] (8-bit camera, half full scale)
    noise_rel: float = 0.01      # rms noise relative to the background
    bits: int = 8

    def __call__(self, img_rel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """I/I_bg -> noisy, quantised, saturating I/I_bg."""
        counts = img_rel * self.bg_counts + rng.normal(0.0, self.noise_rel * self.bg_counts, img_rel.shape)
        counts = np.clip(np.round(counts), 0, 2 ** self.bits - 1)
        return counts / self.bg_counts


@dataclass
class Detection:
    x_md: float | None                      # Mach disk position [D]
    cells: list[float] = field(default_factory=list)  # bright peaks after the Mach disk [D]
    prominence_md: float | None = None
    noise: float | None = None              # band-averaged noise sigma

    @property
    def spacings(self) -> np.ndarray:
        xs = ([self.x_md] if self.x_md is not None else []) + self.cells
        return np.diff(xs)


def centreline_profile(img: np.ndarray, y: np.ndarray, band: float) -> tuple[np.ndarray, int]:
    rows = np.abs(y) <= band
    return img[rows].mean(0), int(rows.sum())


def robust_noise(img: np.ndarray, frac: float) -> float:
    n = max(2, int(frac * img.shape[0]))
    bg = np.concatenate([img[:n], img[-n:]]).ravel()
    return float(1.4826 * np.median(np.abs(bg - np.median(bg))))


def _subpixel(prof: np.ndarray, i: int) -> float:
    if 0 < i < len(prof) - 1:
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def detect(img_rel: np.ndarray, x: np.ndarray, y: np.ndarray, cfg: DetectorConfig = DetectorConfig()) -> Detection:
    """Detect the Mach disk and downstream shock cells in a vertical-knife image I/I_bg."""
    prof, nrows = centreline_profile(img_rel - 1.0, y, cfg.band_D)
    prof = gaussian_filter1d(prof, cfg.smooth_px)
    # Noise of the band average; smoothing reduces it further, ignored (conservative).
    sigma = robust_noise(img_rel, cfg.background_frac) / np.sqrt(nrows)
    sigma = max(sigma, 1e-6)
    dx = x[1] - x[0]
    i0 = int(np.searchsorted(x, cfg.x_min_D))
    peaks, props = find_peaks(prof[i0:], prominence=cfg.k_sigma * sigma)
    if len(peaks) == 0:
        return Detection(None, noise=sigma)
    peaks = peaks + i0
    prom = props["prominences"]
    strong = prom >= cfg.md_rel_prominence * prom.max()
    k_md = int(np.argmax(strong))  # first strong peak
    to_x = lambda i: float(x[0] + _subpixel(prof, i) * dx)
    x_md = to_x(peaks[k_md])
    return Detection(x_md, detect_cells(img_rel, x, y, x_md, sigma * np.sqrt(nrows), cfg),
                     float(prom[k_md]), sigma)


def detect_cells(img_rel, x, y, x_md, pixel_noise, cfg: DetectorConfig) -> list[float]:
    """Bright compression bands downstream of the Mach disk, in a band |y| <= cell_band_frac * x_md."""
    prof, nrows = centreline_profile(img_rel - 1.0, y, cfg.cell_band_frac * x_md)
    prof = gaussian_filter1d(prof, cfg.smooth_px)
    sigma = max(pixel_noise / np.sqrt(nrows), 1e-6)
    dx = x[1] - x[0]
    peaks, props = find_peaks(prof, prominence=cfg.k_sigma * sigma)
    if len(peaks) == 0:
        return []
    xp = x[peaks]
    i_md = int(np.argmin(np.abs(xp - x_md)))  # the Mach disk's own peak in the wide profile
    ref = props["prominences"][i_md]
    return [float(x[0] + _subpixel(prof, p) * dx) for p, xx, pr in zip(peaks, xp, props["prominences"])
            if xx > x_md + cfg.cell_min_gap_D and pr >= cfg.cell_rel_prominence * ref]
