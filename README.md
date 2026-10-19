# schlieren-jet-lab

Simulated underexpanded jet → synthetic Schlieren images → automated shock measurement → comparison with theory, plus a small home background-oriented Schlieren (BOS) experiment.

**Status:** work in progress. Stage 1 (flow solver) is done. Optics, analysis, experiment, and packaging are still to come.

## Stage 1: flow solver (`flow/`)

- `flow/euler.py`: 2D finite-volume Euler solver in PyTorch (CPU/GPU). MUSCL reconstruction (minmod / van Leer), HLLC flux, SSP-RK2, γ = 1.4. Geometry is **axisymmetric (x, r)**, written in finite-volume form (cell volume r·dr plus a p/r source term), so mass is conserved exactly and gas at rest stays at rest.
- `flow/sod.py`, `flow/riemann_exact.py`: Sod shock-tube validation against the exact Riemann solution → `results/sod_metrics.json`, `results/figures/sod.png`.
- `flow/jet.py`: sonic nozzle into quiescent gas over an NPR sweep (5 to 100) → `results/flow/jet_npr*.npz` (time-averaged fields plus 8 instantaneous snapshots).

```bash
uv venv && uv pip install -r requirements.txt
python -m pytest tests          # Sod + axisymmetric checks
python -m flow.sod              # validation figure + metrics
python -m flow.jet              # NPR sweep (GPU: ~5 min on a T4)
python -m flow.jet --fast --device cpu   # small CI-sized run
```
