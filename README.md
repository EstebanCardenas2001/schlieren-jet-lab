# schlieren-jet-lab

Simulated underexpanded jet → synthetic Schlieren images → automated shock measurement → comparison with theory, plus a small home background-oriented Schlieren (BOS) experiment.

## Progress

| Stage | Status |
|---|---|
| 1. Flow solver (`flow/`) | ✅ done |
| 2. Synthetic Schlieren (`optics/`) | ✅ done |
| 3. Shock analysis (`analysis/`) | ⏳ next |
| 4. BOS experiment (`experiment/`) | ☐ |
| 5. Packaging, CI, final README | ☐ |

Every number below is copied from a file in `results/` that a script writes; the file is named next to each number.

```bash
uv venv && uv pip install -r requirements.txt
python -m pytest tests     # 17 tests, ~10 s on CPU
python -m flow.sod         # Sod validation         -> results/sod_metrics.json, figures/sod.png
python -m flow.jet         # NPR sweep, GPU ~7 min  -> results/flow/*.npz, figures/jet_overview.png
python -m optics.render    # synthetic images ~1 min -> results/optics/*, figures/optics_*.png
```
`--fast` on `flow.jet` / `optics.render` runs a small CPU-sized version (used by CI later).

---

## Stage 1: flow solver (`flow/`)

**Scheme** ([flow/euler.py](flow/euler.py)): a 2D finite-volume Euler solver in PyTorch that runs on CPU or GPU.
- MUSCL reconstruction of the primitive variables, with a minmod or van Leer limiter.
- HLLC Riemann solver with Einfeldt wave speeds.
- SSP-RK2 time stepping with a CFL-limited step; ideal gas, γ = 1.4.
- `torch.compile` fuses each RK step (3.5× faster on GPU).

**Geometry: axisymmetric (x, r).**
- It uses the true finite-volume form: cell volume r·dr, face areas r±dr/2, and the radial pressure force p/r as the only source term.
- So mass is conserved to machine precision and gas at rest stays exactly at rest. Both are checked in [tests/test_sod.py](tests/test_sod.py).

**Validation: Sod shock tube vs the exact Riemann solution** ([flow/riemann_exact.py](flow/riemann_exact.py), [flow/sod.py](flow/sod.py)).
- L1(ρ) error at N = 200 is 4.2e-3 with minmod and 2.8e-3 with van Leer.
- The observed order is 0.80–0.91. Below 1 is expected for a solution with a contact discontinuity.
- Runs along x and along y give bit-identical results.
- Source: `results/sod_metrics.json`.

![Sod](results/figures/sod.png)

**Jet case** ([flow/jet.py](flow/jet.py)):
- **Nozzle and reservoir:** sonic nozzle (M = 1 at the exit) into quiescent gas, with NPR = p0/p_a ∈ {5, 10, 15, 20, 30, 45, 60, 80, 100}. The reservoir temperature equals the ambient temperature.
- **Units:** non-dimensional (ambient ρ, p; nozzle diameter D). There's no viscosity, so the flow depends only on NPR and γ.
- **Grid:** 40 cells per D. The domain grows with √NPR, up to 697 × 348 cells at NPR 100.
- **Boundaries:** thin-lipped nozzle with an open back boundary, Riemann-invariant far field on the open sides, and a sponge layer in the outer 40% of the radius.
  - A first version had a closed back wall and no sponge. It trapped start-up vortices in the ambient gas, and their edges projected into fake "shocks" in the Schlieren images. Removing that is what the open boundary and sponge are for.
- **Time-averaged output:** the inviscid shear layer and the subsonic flow behind the Mach disk never become steady. The saved fields are averages over 3 jet transit times, which is what a continuous-light Schlieren exposure records. Each run also saves 8 instantaneous snapshots ("short-exposure frames") for the uncertainty analysis in stage 3.
- **Convergence:** the supersonic core before the Mach disk is steady. The median change between consecutive averaging windows is ≤ 5.0e-4 for all 9 NPRs, so all are marked `converged` (`results/flow/summary.json`).
- **Runtime:** 433 s for the whole sweep on a Tesla T4 (`results/flow/summary.json`).

![jet overview](results/figures/jet_overview.png)

## Stage 2: synthetic Schlieren (`optics/`)

**Chain:** CFD density → refractive index → optical path difference (OPD) through the jet → three synthetic diagnostics.

1. **Gladstone–Dale** ([optics/gas.py](optics/gas.py)): n − 1 = K·ρ with K = 2.26e-4 m³/kg for air. That gives n − 1 = 2.72e-4 for standard air, matching refractometry (tested).
   - **Dimensional setup:** D = 1 mm, T = 293 K, reservoir p0 = 2 bar, so p_a = p0/NPR runs from 400 to 20 mbar. This is like pumping a chamber down at a fixed feed pressure.
   - Because the inviscid flow is self-similar in pressure, ρ = ρ_nd · ρ_a(p_a) is exact within the model.
2. **Abel projection** ([optics/abel.py](optics/abel.py)): integrate Δn along the line of sight.
   - OPD(x, y) = ∫ Δn dz, computed with exact chord lengths through the solver's annular shells.
   - Ambient density is subtracted, so the finite domain adds no edge.
   - Tested on a uniform cylinder and on a Gaussian, whose projection is known analytically.
3. **Image formation** ([optics/imaging.py](optics/imaging.py)): the test section is a thin phase screen E = exp(ik·OPD), and ray deflection is ε = ∇OPD.
   - **Knife-edge Schlieren:** a 4f system with a half-plane filter in the Fourier plane.
     - The knife orientation and its offset into the source image are parameters.
     - A finite incoherent source of angular half-width w is modelled as a sum of tilted plane waves, each shifting the spectrum across the knife.
     - The test reproduces the geometric-optics response I/I_bg − 1 = ε/(w − a_knife) on a known smooth phase step to within 3%. It also checks that flipping the knife inverts the contrast and that a knife parallel to the gradient is blind to it.
   - **Shadowgraph:** paraxial Fresnel propagation over a distance L = 5 cm. The test checks the weak-object limit I − 1 = −L·∇²OPD and energy conservation.
   - **BOS:** background displacement Z_D·ε with Z_D = 0.5 m, converted to pixels of 10 µm projected onto the background.
   - **Sampling:** the phase is upsampled until it changes by at most 0.5 rad per pixel (up to 16×). Intensity is then box-averaged back to camera pixels. The last 1 D at the outflow boundary is cropped from the field of view.

Standard Schlieren system: w = 0.5 mrad, vertical knife cutting half the source.

![optics sweep](results/figures/optics_sweep.png)

Peak signal (99th percentile over the image) across the sweep (`results/optics/contrast_vs_pressure.json`):

| NPR | p_a [mbar] | Schlieren \|ΔI/I_bg\| | shadowgraph \|ΔI/I₀\| | BOS \|Δ\| [px] |
|---:|---:|---:|---:|---:|
| 5 | 400 | 0.44 | 0.16 | 21.1 |
| 10 | 200 | 0.46 | 0.14 | 20.1 |
| 15 | 133 | 0.41 | 0.13 | 18.0 |
| 20 | 100 | 0.32 | 0.11 | 14.8 |
| 30 | 66.7 | 0.24 | 0.088 | 11.1 |
| 45 | 44.4 | 0.16 | 0.062 | 8.0 |
| 60 | 33.3 | 0.12 | 0.048 | 6.2 |
| 80 | 25.0 | 0.072 | 0.036 | 4.9 |
| 100 | 20.0 | 0.064 | 0.030 | 4.2 |

**Contrast vs ambient pressure** (NPR 30 jet, p_a from 1 bar to 0.1 mbar):
- Every signal scales linearly with p_a. The Schlieren signal also saturates at high p_a, once deflections exceed the source half-width.
- Detectability thresholds: 1% intensity noise for Schlieren and shadowgraph, 0.1 px for BOS.

The lowest detectable ambient pressures (`results/optics/contrast_vs_pressure.json`) are:

| system | Schlieren | shadowgraph | BOS |
|---|---:|---:|---:|
| standard (w = 0.5 mrad) | 2.99 mbar | 7.67 mbar | 0.60 mbar |
| high-sensitivity (w = 0.05 mrad) | 0.71 mbar | 7.67 mbar | 0.60 mbar |

- **Diffraction caps the gain from a smaller source.** Shrinking the source 10× improves the Schlieren threshold only about 4×. At w = 0.05 mrad the knife's diffraction scale λ/w ≈ 13 mm is larger than the jet itself, so the knife no longer acts as a sharp angle filter.
- **The continuum model breaks down first.** At D = 1 mm the ambient gas has a Knudsen number above 0.01 below p_a = 6.65 mbar (same file). Most of the "detectable" range of the sensitive systems therefore lies where the Euler model itself becomes questionable. A real low-pressure experiment needs a rarefied-gas model (e.g. DSMC) to go with the better optics.

![contrast vs pressure](results/figures/optics_contrast.png)

## Known limitations so far
- **Euler only:** no viscosity, turbulence or condensation. The shear layer is resolved only by the grid, and the far field isn't physical.
- **Continuum assumption:** questionable below about 7 mbar ambient at D = 1 mm (see above).
- **Grid convergence of the jet:** not yet quantified, only one resolution (40 cells per D).
- **Optics:** paraxial, thin-phase-object approximation. There's no camera noise or blur in the images yet; stage 3 adds noise for the detection test. The knife is ideal (no lens aberrations).
- **Gas:** air only; real lithography-tool jets may use other gases (K and γ differ).
