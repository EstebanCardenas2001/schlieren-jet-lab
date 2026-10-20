"""Printable random-dot background for background-oriented Schlieren (BOS).

Cross-correlation finds sub-pixel shifts best when each dot images onto about
2-3 camera pixels (smaller dots cause "peak locking" to integer pixels,
larger ones blunt the correlation peak), with dots covering ~25-50 % of the
area. Choose the dot size from the camera's footprint on the background:

    dot_mm ~ 2.5 * (field of view width [mm]) / (image width [px])

e.g. a 1920 px wide frame covering 400 mm of background -> dot ~ 0.5 mm.

    python -m experiment.pattern --dot-mm 0.5 --paper A4 --out results/experiment/pattern

writes <out>.png (at --dpi) and <out>.pdf (exact physical size: print at
100 % / "actual size", no "fit to page").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PAPER_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "letter": (279.4, 215.9)}  # landscape (w, h)


def random_dots(width_px: int, height_px: int, dot_px: float, fill: float = 0.4,
                seed: int = 0) -> np.ndarray:
    """Black dots of diameter dot_px on white, placed uniformly at random.

    Dots may overlap; the number is chosen so that the *covered* fraction is
    `fill` (Poisson coverage: 1 - exp(-n * a / A) = fill).
    """
    rng = np.random.default_rng(seed)
    area = width_px * height_px
    a_dot = np.pi * dot_px ** 2 / 4
    n = int(-np.log(1 - fill) * area / a_dot)
    img = np.ones((height_px, width_px), dtype=np.float32)
    cx = rng.uniform(0, width_px, n)
    cy = rng.uniform(0, height_px, n)
    r = dot_px / 2
    rr = int(np.ceil(r)) + 1
    # Stamp a supersampled disc kernel at sub-pixel offsets (anti-aliased edges).
    for ox in range(-rr, rr + 1):
        for oy in range(-rr, rr + 1):
            xi = np.floor(cx).astype(int) + ox
            yi = np.floor(cy).astype(int) + oy
            d = np.hypot(xi + 0.5 - cx, yi + 0.5 - cy)
            cov = np.clip(r + 0.5 - d, 0.0, 1.0)  # ~ area fraction of the pixel inside the disc
            ok = (xi >= 0) & (xi < width_px) & (yi >= 0) & (yi < height_px) & (cov > 0)
            np.minimum.at(img, (yi[ok], xi[ok]), 1.0 - cov[ok])
    return img


def make_pattern(dot_mm: float, paper: str = "A4", dpi: int = 300, fill: float = 0.4, seed: int = 0):
    w_mm, h_mm = PAPER_MM[paper]
    px_per_mm = dpi / 25.4
    img = random_dots(round(w_mm * px_per_mm), round(h_mm * px_per_mm), dot_mm * px_per_mm, fill, seed)
    return img, (w_mm, h_mm)


def save_pattern(img: np.ndarray, size_mm: tuple[float, float], out: Path, dpi: int) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_suffix(".png")), np.round(img * 255).astype(np.uint8))
    fig = plt.figure(figsize=(size_mm[0] / 25.4, size_mm[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dot-mm", type=float, default=0.5)
    ap.add_argument("--paper", default="A4", choices=PAPER_MM)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fill", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/experiment/pattern"))
    a = ap.parse_args()
    img, size = make_pattern(a.dot_mm, a.paper, a.dpi, a.fill, a.seed)
    save_pattern(img, size, a.out, a.dpi)
    print(f"{a.paper} {size[0]:.0f}x{size[1]:.0f} mm, dots {a.dot_mm} mm, fill {1 - img.mean():.2f} "
          f"-> {a.out.with_suffix('.png')}, {a.out.with_suffix('.pdf')} (print at 100 %)")


if __name__ == "__main__":
    main()
