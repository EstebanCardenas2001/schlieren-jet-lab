"""Image formation from a phase screen: shadowgraph, knife-edge Schlieren, BOS.

The test section is treated as a thin phase object: a collimated beam leaves it
with field E(x, y) = exp(i k OPD(x, y)), where OPD = integral (n - 1) dz is the
optical path difference from the Abel projection. In the paraxial limit the
local ray deflection is eps = grad OPD, i.e. the local spatial frequency of E
is eps / lambda. All three diagnostics follow from E:

* Knife-edge Schlieren: a 4f system images the test section onto the camera;
  its Fourier plane holds the angular spectrum of E (angle a = lambda * f).
  The knife is a half-plane mask in that plane, blocking a_n < a_knife along
  the knife normal n = (cos t, sin t). A real Schlieren source has finite size,
  so the image of the source in the Fourier plane is a slit of angular
  half-width w. Points of an incoherent source add in intensity: each source
  point is a tilted plane wave, which shifts the spectrum across the knife.
  For deflections |eps| < w the response is linear,
      I / I_bg - 1 = eps_n / (w - a_knife),
  and it saturates beyond that. w -> 0 gives the coherent (Hilbert-filter) limit.
* Shadowgraph: free-space (angular-spectrum, paraxial) propagation of E over a
  distance L to a screen. Weak signal: I/I0 - 1 ~ -L lap(OPD). Strong
  gradients form caustics, which wave propagation handles naturally.
* BOS: a background at distance Z_D behind the flow appears shifted by
  Z_D * eps (paraxial, flow thin compared with Z_D), measured on the background.

Sampling: the phase k*OPD must be resolved (Nyquist) before any FFT, so OPD
is upsampled (bicubic) until the largest phase step per pixel is below
`max_phase_step`. Intensities are then averaged back to the original pixels,
the way a camera pixel integrates light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class OpticsConfig:
    wavelength: float = 632.8e-9   # He-Ne [m]
    # knife-edge Schlieren
    knife_angle_deg: float = 0.0   # knife-edge normal: 0 -> sensitive to d/dx (axial), 90 -> d/dy
    source_halfwidth: float = 1e-3  # angular half-width w of the source image [rad]
    knife_offset: float = 0.0      # knife position / w; 0 blocks half of the source image
    n_source: int = 33             # incoherent source points across the slit
    # shadowgraph / BOS
    shadow_distance: float = 0.05  # screen distance L [m]
    bos_distance: float = 0.5      # background distance Z_D [m]
    # sampling
    max_phase_step: float = 0.5    # [rad / pixel] after upsampling
    max_upsample: int = 8
    pad_fraction: float = 0.25


def _device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def upsample_factor(opd: np.ndarray, dx: float, cfg: OpticsConfig) -> tuple[int, float]:
    """Smallest integer factor that keeps the phase step per pixel below max_phase_step."""
    k = 2 * math.pi / cfg.wavelength
    gy, gx = np.gradient(opd, dx)
    step = k * float(np.max(np.hypot(gx, gy))) * dx  # rad per original pixel
    f = max(1, math.ceil(step / cfg.max_phase_step))
    return min(f, cfg.max_upsample), step


def _prepare_field(opd: np.ndarray, dx: float, cfg: OpticsConfig, device):
    """Upsample + reflect-pad OPD, return complex field and bookkeeping."""
    dev = _device(device)
    f, step = upsample_factor(opd, dx, cfg)
    o = torch.as_tensor(opd, dtype=torch.float64, device=dev)[None, None]
    if f > 1:
        o = F.interpolate(o, scale_factor=f, mode="bicubic", align_corners=False)
    H, W = o.shape[-2:]
    py, px = int(cfg.pad_fraction * H), int(cfg.pad_fraction * W)
    # Reflect padding keeps the field continuous at the borders, so the FFT's
    # periodic wrap-around does not create spurious edge diffraction.
    o = F.pad(o, (px, px, py, py), mode="reflect")[0, 0]
    k = 2 * math.pi / cfg.wavelength
    E = torch.exp(1j * (k * o)).to(torch.complex64)
    info = {"upsample": f, "phase_step_native": step,
            "phase_step_after": step / f, "undersampled": step / f > cfg.max_phase_step}
    return E, (py, px, H, W), dx / f, f, info


def _crop_downsample(I: torch.Tensor, crop, f: int) -> np.ndarray:
    py, px, H, W = crop
    I = I[py:py + H, px:px + W]
    if f > 1:
        I = F.avg_pool2d(I[None, None], f)[0, 0]
    return I.float().cpu().numpy()


def _angle_grids(shape, h: float, wavelength: float, device):
    Hn, Wn = shape
    fy = torch.fft.fftfreq(Hn, d=h, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(Wn, d=h, device=device, dtype=torch.float64)
    return wavelength * fx[None, :], wavelength * fy[:, None]


def schlieren(opd: np.ndarray, dx: float, cfg: OpticsConfig = OpticsConfig(), device=None):
    """Knife-edge Schlieren image I / I_incident and its no-flow background I_bg."""
    E, crop, h, f, info = _prepare_field(opd, dx, cfg, device)
    S = torch.fft.fft2(E)
    ax, ay = _angle_grids(E.shape, h, cfg.wavelength, E.device)
    t = math.radians(cfg.knife_angle_deg)
    an = ax * math.cos(t) + ay * math.sin(t)                 # angle along the knife normal
    da = cfg.wavelength / h * (abs(math.cos(t)) / E.shape[1] + abs(math.sin(t)) / E.shape[0])
    w = cfg.source_halfwidth
    a_knife = cfg.knife_offset * w
    offsets = np.linspace(-w, w, cfg.n_source) if (w > 0 and cfg.n_source > 1) else np.array([0.0])

    I = torch.zeros(E.shape, dtype=torch.float32, device=E.device)
    I_bg = 0.0
    for s in offsets:
        # Source point s puts the undiffracted light at a_n = s. The edge is
        # softened over one frequency bin so light sitting exactly on the
        # knife is half transmitted.
        Hs = ((an + s - a_knife) / da + 0.5).clamp(0.0, 1.0)
        I += torch.fft.ifft2(S * Hs.to(S.dtype)).abs().square()
        I_bg += float(np.clip((s - a_knife) / da + 0.5, 0.0, 1.0)) ** 2
    I /= len(offsets)
    I_bg /= len(offsets)
    info.update({"I_bg": I_bg, "n_source": len(offsets)})
    return _crop_downsample(I, crop, f), I_bg, info


def shadowgraph(opd: np.ndarray, dx: float, cfg: OpticsConfig = OpticsConfig(), device=None):
    """Shadowgraph intensity I / I0 on a screen at distance L (paraxial Fresnel)."""
    E, crop, h, f, info = _prepare_field(opd, dx, cfg, device)
    ax, ay = _angle_grids(E.shape, h, cfg.wavelength, E.device)
    # Paraxial transfer function exp(-i pi lambda L (fx^2 + fy^2)), written with a = lambda f.
    phase = -math.pi * cfg.shadow_distance / cfg.wavelength * (ax ** 2 + ay ** 2)
    Hz = torch.polar(torch.ones_like(phase), phase).to(E.dtype)
    I = torch.fft.ifft2(torch.fft.fft2(E) * Hz).abs().square()
    return _crop_downsample(I, crop, f), info


def deflection(opd: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Ray deflection angles (eps_x, eps_y) = grad OPD [rad]."""
    gy, gx = np.gradient(opd, dx)
    return gx, gy


def bos_displacement(opd: np.ndarray, dx: float, cfg: OpticsConfig = OpticsConfig()):
    """Apparent background displacement (dx, dy) in the background plane [m]."""
    ex, ey = deflection(opd, dx)
    return cfg.bos_distance * ex, cfg.bos_distance * ey
