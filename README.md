# Transport-Inspired Rare-Event Sampling in Spin Systems

**Learned Importance Improves Deep-Target Reliability**

M.Y. Alzaatreh — Natural Sciences Unit, Fahad Bin Sultan University

Status: manuscript submitted to *Physical Review E*.

---

## Abstract

Rare configurations determine important properties of magnetic systems, such as the far
tail of the magnetization distribution or the lowest-energy arrangements of a frustrated
magnet. They are difficult to simulate because most computational effort is spent on
typical states, while the configurations that determine a rare-event probability are
visited only infrequently. Radiation-transport simulations face a similar problem and
improve sampling by directing computational effort toward states that contribute most
strongly to the quantity of interest. This work adapts that idea to lattice spin systems
using weighted ensemble sampling, which redistributes computational effort among
trajectories while preserving their statistical weights. Tests against problems with known
answers first confirm that the implementation reproduces the correct probabilities. For the
two-dimensional Ising model, weighted ensemble and direct Monte Carlo reach the same depth
in the magnetization tail at matched update cost. Direct sampling is more precise over most
of the tail, while weighted ensemble becomes about twice as precise at the rarest resolved
point. The main test considers the Edwards-Anderson spin glass, where a neural network
learns a progress coordinate from the full spin configuration. Across six independent
disorder realizations, simulations using the learned coordinate reach the deep low-energy
target in 67.2% of runs, compared with 52.8% when energy alone is used, an improvement
confirmed by a stratified analysis across realizations. Neural-network evaluation increases
computational time by a factor of four to five, so the gain in reliability does not yet
translate into a wall-clock speedup. These results show that information learned from the
full configuration can improve access to rare states while leaving the underlying physical
dynamics unchanged.

## Manuscript

The submitted manuscript (source and PDF) is at
[`project/07_paper/paper_V10.tex`](project/07_paper/paper_V10.tex) /
[`project/07_paper/paper_V10.pdf`](project/07_paper/paper_V10.pdf).

## Repository contents

This repository contains the full simulation and analysis code used to produce every
result and figure in the manuscript, organized as a sequence of validated stages:

| Stage | Contents |
|---|---|
| [`01_van_ising/`](project/01_van_ising) | Variational autoregressive network learns the 2D Ising Boltzmann distribution, validated against the exact Onsager solution and exact enumeration |
| [`02_magnetic_phases_doc/`](project/02_magnetic_phases_doc) | Background document on magnetic phases and ML methods for them |
| [`03_phase1_framing/`](project/03_phase1_framing) | Theoretical framing connecting the adjoint importance function of transport theory to the rare-event committor |
| [`04_phase2_core/`](project/04_phase2_core) | Weight windows and Weighted Ensemble sampling for 2D Ising rare-magnetization tails, gated against exact enumeration |
| [`05_neural_importance/`](project/05_neural_importance) | The learned importance function, evaluated head-to-head against baseline coordinates on the Ising model and the Edwards-Anderson spin glass |
| [`06_fw_cadis/`](project/06_fw_cadis) | Forward-weighted CADIS variance-reduction comparison |
| [`07_paper/`](project/07_paper) | Manuscript source, figures, and revision archive |
| [`08_paper_methods/`](project/08_paper_methods) | Extended methods write-up |

See [`project/README.md`](project/README.md) for full setup, reproduction instructions
(`run_all.py`), and a walkthrough of how each stage builds on and validates the last.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r project/requirements.txt
cd project
python3 run_all.py --preset smoke   # ~2 min: verifies every stage and validation gate runs
```

Everything runs on a single CPU; no GPU is required. See
[`project/README.md`](project/README.md) for the `laptop` and `full` (publication-scale)
presets.

## License

Code is released under the [MIT License](LICENSE).
