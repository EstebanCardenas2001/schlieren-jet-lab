"""2D compressible Euler solver (finite volume) in PyTorch.

Scheme
------
* Conserved variables U = (rho, rho*u, rho*v, E) on a uniform Cartesian grid,
  stored as a tensor of shape [4, ny, nx]  (index order: variable, y/r, x).
* MUSCL reconstruction of the *primitive* variables (rho, u, v, p) with a slope
  limiter (minmod or van Leer). Limiting primitives rather than conserved
  variables keeps pressure positive more reliably in the strong expansions of
  an underexpanded jet.
* HLLC approximate Riemann solver (Toro 2009, ch. 10) with Einfeldt
  (Roe-averaged) wave-speed bounds. HLLC restores the contact wave that plain
  HLL smears, which matters here because the jet boundary is a contact /
  shear layer.
* Unsplit dimension-by-dimension fluxes, SSP-RK2 (Heun) time stepping,
  CFL-limited time step.
* Planar (x, y) or axisymmetric (x, r) geometry. In axisymmetric mode the
  y-axis is the radius r, the bottom boundary is the symmetry axis, and the
  finite-volume update uses the true cell volume r_j*dr and face areas
  r_{j+-1/2}. The only remaining source term is the radial pressure force
  p/r. This form conserves mass exactly and keeps a uniform gas at rest
  exactly at rest ("well balanced"), unlike the cruder -(1/r)*H source form.

Everything is vectorised tensor slicing, so the same code runs on CPU or GPU.
The time loop avoids host-device syncs: dt stays a tensor and the stop
condition is checked only every few steps.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

GAMMA = 1.4  # ideal diatomic gas (air, N2)
NG = 2       # ghost-cell layers needed by the MUSCL stencil

RHO_FLOOR = 1e-10
P_FLOOR = 1e-10


# ---------------------------------------------------------------- variables --
def cons_to_prim(U: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v))
    return torch.stack([rho, u, v, p])


def prim_to_cons(W: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    rho, u, v, p = W
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return torch.stack([rho, rho * u, rho * v, E])


def sound_speed(W: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    return torch.sqrt(gamma * W[3] / W[0])


def mach_number(W: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    return torch.sqrt(W[1] ** 2 + W[2] ** 2) / sound_speed(W, gamma)


# ----------------------------------------------------------- reconstruction --
def _limit(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    """Limited slope from left/right differences a, b (TVD limiters)."""
    zero = torch.zeros_like(a)
    if kind == "minmod":
        return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), zero)
    if kind == "vanleer":
        ab = a * b
        return torch.where(ab > 0, 2.0 * ab / (a + b), zero)
    raise ValueError(f"unknown limiter {kind!r}")


def muscl(W: torch.Tensor, dim: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right face states along `dim` from a field padded with NG ghosts.

    W has n = N + 4 cells along `dim` (N interior). Returns WL, WR with N + 1
    entries: the two sides of every face bounding the interior cells.
    """
    n = W.shape[dim]
    N = n - 2 * NG
    d = W.narrow(dim, 1, n - 1) - W.narrow(dim, 0, n - 1)           # d[k] = W[k+1] - W[k]
    s = _limit(d.narrow(dim, 0, n - 2), d.narrow(dim, 1, n - 2), kind)  # slope of cell k+1
    WL = W.narrow(dim, 1, N + 1) + 0.5 * s.narrow(dim, 0, N + 1)
    WR = W.narrow(dim, 2, N + 1) - 0.5 * s.narrow(dim, 1, N + 1)
    # Guard against non-physical reconstructed states in extreme expansions.
    for Wf in (WL, WR):
        Wf[0].clamp_(min=RHO_FLOOR)
        Wf[3].clamp_(min=P_FLOOR)
    return WL, WR


# ------------------------------------------------------------- HLLC flux ----
def hllc_flux(WL: torch.Tensor, WR: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """HLLC flux normal to a face. Component 1 of W is the normal velocity."""
    rL, uL, vL, pL = WL
    rR, uR, vR, pR = WR
    g1 = gamma - 1.0
    EL = pL / g1 + 0.5 * rL * (uL * uL + vL * vL)
    ER = pR / g1 + 0.5 * rR * (uR * uR + vR * vR)
    cL = torch.sqrt(gamma * pL / rL)
    cR = torch.sqrt(gamma * pR / rR)

    # Einfeldt wave-speed bounds built on Roe averages (positively conservative).
    sL, sR = torch.sqrt(rL), torch.sqrt(rR)
    wsum = sL + sR
    u_roe = (sL * uL + sR * uR) / wsum
    v_roe = (sL * vL + sR * vR) / wsum
    H_roe = (sL * (EL + pL) / rL + sR * (ER + pR) / rR) / wsum
    c_roe = torch.sqrt((g1 * (H_roe - 0.5 * (u_roe ** 2 + v_roe ** 2))).clamp(min=1e-14))
    SL = torch.minimum(uL - cL, u_roe - c_roe)
    SR = torch.maximum(uR + cR, u_roe + c_roe)

    # Contact (middle) wave speed.
    mL = rL * (SL - uL)
    mR = rR * (SR - uR)
    Ss = (pR - pL + uL * mL - uR * mR) / (mL - mR)

    def flux(r, u, v, p, E):
        ru = r * u
        return torch.stack([ru, ru * u + p, ru * v, u * (E + p)])

    def star_minus_U(r, u, v, p, E, S):
        f = r * (S - u) / (S - Ss)
        Us = torch.stack([f, f * Ss, f * v, f * (E / r + (Ss - u) * (Ss + p / (r * (S - u))))])
        U = torch.stack([r, r * u, r * v, E])
        return Us - U

    FL = flux(rL, uL, vL, pL, EL)
    FR = flux(rR, uR, vR, pR, ER)
    FsL = FL + SL * star_minus_U(rL, uL, vL, pL, EL, SL)
    FsR = FR + SR * star_minus_U(rR, uR, vR, pR, ER, SR)

    return torch.where(SL >= 0, FL,
           torch.where(Ss >= 0, FsL,
           torch.where(SR > 0, FsR, FR)))


_SWAP_UV = [0, 2, 1, 3]  # reorder (rho, u, v, *) <-> (rho, v, u, *)


# ------------------------------------------------------------------ solver --
class EulerSolver2D:
    """Finite-volume Euler solver on x in [x0, x0+nx*dx], y/r in [y0, y0+ny*dy].

    Boundary conditions are given per side ("left", "right", "bottom", "top"):
      "outflow"  zero-gradient extrapolation
      "reflect"  slip wall / symmetry plane / axis
      "farfield" Riemann-invariant boundary to the quiescent state W_amb
    `inflow = (side, mask, W_in)` overrides the ghost cells of that side where
    `mask` (length of that side) is True with the fixed primitive state W_in
    (a supersonic / sonic inflow, e.g. a nozzle exit).
    `sponge` (shape [ny, nx], units 1/time) adds a relaxation term
    -sponge * (U - U_amb) that absorbs structures near open boundaries.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, *,
                 x0: float = 0.0, y0: float = 0.0,
                 axisymmetric: bool = False,
                 bc: dict[str, str] | None = None,
                 W_amb: torch.Tensor | None = None,
                 inflow: tuple[str, torch.Tensor, torch.Tensor] | None = None,
                 limiter: str = "minmod",
                 gamma: float = GAMMA,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64,
                 sponge: torch.Tensor | None = None,
                 compile: bool = False):
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        self.gamma, self.limiter = gamma, limiter
        self.axisymmetric = axisymmetric
        self.device, self.dtype = torch.device(device), dtype
        self.bc = {"left": "outflow", "right": "outflow", "bottom": "outflow", "top": "outflow"}
        if bc:
            self.bc.update(bc)
        if axisymmetric:
            assert y0 == 0.0 and self.bc["bottom"] == "reflect", "axisymmetric: bottom must be the axis (r=0, reflect)"
        kw = dict(device=self.device, dtype=dtype)
        self.x = x0 + (torch.arange(nx, **kw) + 0.5) * dx
        self.y = y0 + (torch.arange(ny, **kw) + 0.5) * dy
        if axisymmetric:
            self.r_c = self.y[:, None]                                   # cell-centre radius
            self.r_f = (torch.arange(ny + 1, **kw) * dy)[:, None]        # face radius
        self.W_amb = None if W_amb is None else W_amb.to(**kw)
        if inflow is not None:
            side, mask, W_in = inflow
            inflow = (side, mask.to(self.device), W_in.to(**kw))
        self.inflow = inflow
        self.sponge = None
        if sponge is not None:
            assert self.W_amb is not None, "sponge needs W_amb"
            self.sponge = sponge.to(**kw)
            self.U_amb = prim_to_cons(self.W_amb[:, None, None], gamma)
        self.U: torch.Tensor | None = None
        # Time is accumulated in float64 even for float32 fields; otherwise
        # t + (t_end - t) can round below t_end and the run never terminates.
        self.t = torch.zeros((), device=self.device, dtype=torch.float64)
        self.nsteps = 0
        # torch.compile fuses the few hundred small elementwise kernels of one RK
        # step; without it the GPU is launch-bound (~10 ms/step at any grid size).
        self._advance = torch.compile(self._rk2, dynamic=False) if compile else self._rk2
        self._max_dt = torch.compile(self._max_dt_impl, dynamic=False) if compile else self._max_dt_impl

    # -------------------------------------------------------------- setup --
    def set_initial(self, W: torch.Tensor) -> None:
        self.U = prim_to_cons(W.to(self.device, self.dtype), self.gamma)

    @property
    def W(self) -> torch.Tensor:
        return cons_to_prim(self.U, self.gamma)

    # --------------------------------------------------------- boundaries --
    def _fill_side(self, Wp: torch.Tensor, side: str) -> None:
        g = NG
        kind = self.bc[side]
        dim = 2 if side in ("left", "right") else 1
        normal = 1 if dim == 2 else 2                     # index of normal velocity
        lo = side in ("left", "bottom")
        n = Wp.shape[dim]

        def sl(i):  # slice along dim
            idx = [slice(None)] * 3
            idx[dim] = i
            return tuple(idx)

        first = g if lo else n - g - 1                    # first interior cell next to boundary
        step = 1 if lo else -1                            # direction into the domain
        for k in range(g):                                # k = 0 is the ghost adjacent to the boundary
            ghost = first - step * (k + 1)
            if kind == "reflect":
                Wp[sl(ghost)] = Wp[sl(first + step * k)]
                Wp[(normal,) + sl(ghost)[1:]] *= -1.0
            else:
                Wp[sl(ghost)] = Wp[sl(first)]
        if kind == "farfield":
            Wb = self._farfield_state(Wp[sl(first)], normal, -step)
            for k in range(g):
                Wp[sl(first - step * (k + 1))] = Wb
        if self.inflow is not None and self.inflow[0] == side:
            _, mask, W_in = self.inflow
            for k in range(g):
                ghost = first - step * (k + 1)
                Wg = Wp[sl(ghost)]
                off = 0 if dim == 2 else g                # x sides get a view of interior rows only
                inner = slice(off, off + mask.numel())    # interior span along the side
                seg = Wg[:, inner]
                seg[:] = torch.where(mask[None, :], W_in[:, None], seg)

    def _farfield_state(self, Wi: torch.Tensor, normal: int, out_sign: int) -> torch.Tensor:
        """Riemann-invariant far-field state (e.g. Blazek, CFD Principles, sec. 8.2).

        Subsonic: the outgoing invariant R+ = u_n + 2c/(g-1) comes from the
        interior, the incoming R- = u_n - 2c/(g-1) from the ambient gas at rest.
        Entropy and tangential velocity are taken from upwind: interior on
        outflow, ambient on inflow. Supersonic outflow is extrapolated.
        This lets waves leave with little reflection and pins the far field to
        the ambient state without imposing the velocity.
        """
        assert self.W_amb is not None, "farfield boundary needs W_amb"
        g = self.gamma
        rho_a, p_a = self.W_amb[0], self.W_amb[3]
        c_a = torch.sqrt(g * p_a / rho_a)
        tang = 3 - normal
        un_i = out_sign * Wi[normal]
        c_i = torch.sqrt(g * Wi[3] / Wi[0])
        R_out = un_i + 2.0 * c_i / (g - 1.0)
        R_in = -2.0 * c_a / (g - 1.0)
        un_b = 0.5 * (R_out + R_in)
        c_b = 0.25 * (g - 1.0) * (R_out - R_in)
        outflow = un_b > 0
        s_b = torch.where(outflow, Wi[3] / Wi[0] ** g, p_a / rho_a ** g)  # entropy p/rho^g
        rho_b = (c_b * c_b / (g * s_b)) ** (1.0 / (g - 1.0))
        Wb = torch.empty_like(Wi)
        Wb[0] = rho_b
        Wb[normal] = out_sign * un_b
        Wb[tang] = torch.where(outflow, Wi[tang], torch.zeros_like(un_b))
        Wb[3] = rho_b * c_b * c_b / g
        supersonic_out = un_i >= c_i
        return torch.where(supersonic_out[None], Wi, Wb)

    def pad(self, W: torch.Tensor) -> torch.Tensor:
        g = NG
        Wp = W.new_empty((4, self.ny + 2 * g, self.nx + 2 * g))
        Wp[:, g:-g, g:-g] = W
        # x sides first (interior rows), then y sides over full width.
        for side in ("left", "right"):
            self._fill_side_rows(Wp, side)
        for side in ("bottom", "top"):
            self._fill_side(Wp, side)
        return Wp

    def _fill_side_rows(self, Wp: torch.Tensor, side: str) -> None:
        # Operate on interior rows only so the inflow mask aligns with y cells.
        g = NG
        view = Wp[:, g:-g, :]
        self._fill_side(view, side)

    # ----------------------------------------------------------------- RHS --
    def rhs(self, U: torch.Tensor) -> torch.Tensor:
        g = NG
        W = cons_to_prim(U, self.gamma)
        Wp = self.pad(W)

        WL, WR = muscl(Wp[:, g:-g, :], 2, self.limiter)
        F = hllc_flux(WL, WR, self.gamma)                               # [4, ny, nx+1]

        WL, WR = muscl(Wp[:, :, g:-g], 1, self.limiter)
        G = hllc_flux(WL[_SWAP_UV], WR[_SWAP_UV], self.gamma)[_SWAP_UV]  # [4, ny+1, nx]

        dU = -(F[:, :, 1:] - F[:, :, :-1]) / self.dx
        if self.axisymmetric:
            rG = self.r_f * G
            dU = dU - (rG[:, 1:] - rG[:, :-1]) / (self.r_c * self.dy)
            dU[2] = dU[2] + W[3] / self.r_c                             # radial pressure force p/r
        else:
            dU = dU - (G[:, 1:] - G[:, :-1]) / self.dy
        if self.sponge is not None:
            dU = dU - self.sponge * (U - self.U_amb)
        return dU

    def _max_dt_impl(self, U: torch.Tensor, cfl: float) -> torch.Tensor:
        W = cons_to_prim(U, self.gamma)
        c = sound_speed(W, self.gamma)
        rate = ((W[1].abs() + c) / self.dx + (W[2].abs() + c) / self.dy).max()
        return cfl / rate

    def max_dt(self, cfl: float) -> torch.Tensor:
        return self._max_dt(self.U, cfl)

    # ------------------------------------------------------------ stepping --
    def _rk2(self, U0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """One SSP-RK2 (Heun) step."""
        U1 = U0 + dt * self.rhs(U0)
        return 0.5 * U0 + 0.5 * (U1 + dt * self.rhs(U1))

    def step(self, dt: torch.Tensor) -> None:
        self.U = self._advance(self.U, dt.to(self.dtype))
        self.t = self.t + dt.to(torch.float64)
        self.nsteps += 1

    def run(self, t_end: float, cfl: float = 0.4, check_every: int = 25,
            callback: Callable[["EulerSolver2D"], None] | None = None,
            max_steps: int | None = None) -> None:
        """Advance to exactly t_end. Only syncs with the host every `check_every` steps."""
        t_end_t = torch.tensor(t_end, device=self.device, dtype=torch.float64)
        while True:
            for _ in range(check_every):
                dt = torch.minimum(self.max_dt(cfl).double(), t_end_t - self.t).clamp(min=0.0)
                self.step(dt)
            if callback is not None:
                callback(self)
            if self.t.item() >= t_end * (1 - 1e-12):
                break
            if not math.isfinite(self.U[3].sum().item()):
                raise FloatingPointError(f"solution blew up at t={self.t.item():.4g}")
            if max_steps is not None and self.nsteps >= max_steps:
                break
