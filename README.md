# Quantum Mpemba Effect in Disordered Lattices

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the simulation code, data pipelines, and figure generation scripts for the paper:

> **"Quantum Mpemba Effect in Disordered Lattices"**  
> P. Chuklanov, M. Less, P. V. Zakharenko, D. E. Zaitsev, A. P. Alodjants.

The study investigates dissipative single-photon dynamics and anomalous relaxation (quantum Mpemba effect) on topologically disordered (rewired) 2D lattices described by Markovian Lindblad master equations.

---

## Repository Structure

```
.
├── src/qdyn_research/      # Core scientific package
│   ├── liouvillian.py      # Lindblad generator and Liouvillian superoperator assembly
│   ├── spectral.py         # Dense & sparse eigensolvers (rightmost-mode Arnoldi)
│   ├── topology.py         # 2D grid generation, rewiring, and graph metrics
│   ├── metrics.py          # Modal projections, IPR, von Neumann entropy distance
│   ├── simulation.py       # ODE time-evolution integration
│   ├── mpemba_validation.py# Hot/cold state engineering and crossing validation
│   └── plot_style.py       # Publication-quality figure styling (Springer sn-jnl)
├── scripts/                # Executable reproduction and analysis scripts
│   ├── fig1_grouped_modes.py
│   ├── fig2_projection_bars_and_modes.py
│   ├── ...
│   ├── fig13_random_benchmark.py
│   └── precalc/            # Canonical graph and precomputed data caches
├── paper/                  # LaTeX manuscript source (Springer sn-jnl) and vector figures
├── tests/                  # Verification suite (exact identities, solver accuracy)
└── docs/
    └── REPRODUCING.md      # Detailed figure-by-figure reproduction guide
```

---

## Installation

### Prerequisites
- Python $\ge$ 3.10

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/alpha7en/mpemba_new.git
cd mpemba_new
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Verification & Tests

The test suite validates physical identities (generator trace preservation, Lindbladian representation, Dirichlet state validity, phase-invariant group metrics) and solver accuracy against exact ground truth:

```bash
# Run fast smoke & artifact tests (~1 min)
pytest

# Run full heavy verification suite (includes dense 1296x1296 eigensolves)
pytest -m slow
```

---

## Reproducing Paper Figures

All manuscript figures can be generated directly from `scripts/` using cached reductions or recomputed from scratch. See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for full details on data sources and seeds.

```bash
# Generate specific figure panels
python scripts/fig1_grouped_modes.py
python scripts/fig2_projection_bars_and_modes.py
python scripts/fig3_relaxation_dynamics.py
python scripts/fig4_average_path_length.py
python scripts/fig5_topology_and_spectrum.py
python scripts/fig6_lambda_visualization.py
python scripts/fig7_ipr_visualization.py
python scripts/fig8_overlap.py
python scripts/fig9_task_b_heatmap.py
python scripts/fig10_random_benchmark.py
python scripts/fig11_algorithm_visual.py
```

Generated figure outputs are automatically styled for publication and saved to `paper/figures/`.

---

## Data Availability

- Lightweight canonical datasets and the exact graph realization used in the paper are committed in `scripts/precalc/`.
- Heavy Monte Carlo sweeps can be regenerated using `scripts/run_sweep_rightmost.py` and `scripts/run_dirichlet_benchmark.py`.

---

## Citation

```bibtex
@article{chuklanov2026mpemba,
  title={Quantum Mpemba Effect in Disordered Lattices},
  author={Chuklanov, P. and Less, M. and Zakharenko, P. V. and Zaitsev, D. E. and Alodjants, A. P.},
  year={2026}
}
```
