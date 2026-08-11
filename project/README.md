# Rare-Event Sampling for Spin Systems via Transport-Theoretic Variance Reduction

A physics + ML research project connecting **variational autoregressive networks (VAN)**
for the 2D Ising model to **adjoint / weight-window variance reduction** from radiation
transport (CADIS / FW-CADIS). Built incrementally, each stage validated against an exact
reference (Onsager solution, exact enumeration, or a solvable toy model) before moving on.

Everything here runs on a single CPU. No GPU required.

---

## Quickstart

```bash
# 1. clone / unzip this folder, then from inside it:
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# 2. check your environment
python3 check_env.py

# 3. ONE COMMAND runs everything. Start here:
python3 run_all.py --preset smoke          # ~2 min: proves all code + gates work
python3 run_all.py --preset laptop         # ~45-70 min: the real run
python3 run_all.py --preset full --jobs 8  # hours: publication scale
```

`run_all.py` auto-detects your cores, runs every stage in order, streams output, and
writes a combined `run_all_<preset>.log` plus a pass/fail summary. Useful flags:

| flag | meaning |
|---|---|
| `--only 4 5` | run just those stages |
| `--jobs N` | cores for the stages that parallelise (default: cores-1) |
| `--build-docs` | also recompile the LaTeX documents (needs `pdflatex`) |

Presets change problem size, **not** correctness -- every preset runs the same code and the
same validation gates against exact references. `bash run_smoke_tests.sh` is still there as
a lighter alternative that skips stage 5.

---

## Project structure and the story it tells

```
01_van_ising/           VAN learns the 2D Ising Boltzmann distribution; eliminates
                         critical slowing down (validated vs Onsager + exact enumeration)
02_magnetic_phases_doc/ Teaching document: magnetic phases + 5 AI/ML methods for them
03_phase1_framing/      Theory memo: adjoint importance = rare-event committor =
                         "neural importance function" -- the framing that motivates
                         Phase 2, with an exact zero-variance demonstration
04_phase2_core/         Validated technical core: weight windows (1D unit test) and
                         Weighted Ensemble for 2D Ising rare magnetization tails
                         (gated against exact enumeration; L=20 rare-tail result)
05_neural_importance/   LEARNED importance function I_theta(s) in full configuration
                         space, head-to-head by figure of merit against the
                         coordinate baselines, on the Ising model AND the
                         Edwards-Anderson spin glass
```

Read in this order if you're new to the project: `02` (background) -> `01` (the ML tool)
-> `03` (the theoretical bridge) -> `04` (the validated payoff) -> `05` (the frontier).
Each folder has its own `README.md` (or the memo itself, for 02/03) with full details --
this file is the map.

Stages 01-04 are **validated against exact references**. Stage 05 is **implemented and
unbiasedness-gated, but its headline claim is not yet established** -- see its README and
the honest-scope section below.

---

## 01_van_ising -- VAN learns the Ising model

```bash
cd 01_van_ising
python3 run.py                          # default L=8 demo, ~6 min on 1 CPU
python3 run.py --L 16 --steps0 1000 --steps 400   # bigger/slower/better
python3 critical_slowing_down.py        # Metropolis vs VAN autocorrelation comparison
```
**Validated:** L=8 VAN reproduces the exact Onsager free energy (the variational bound
F_q >= F_exact holds at every temperature); Metropolis integrated autocorrelation time
spikes to ~46 near T_c while the VAN's is exactly 1 (i.i.d. sampling by construction).

## 02_magnetic_phases_doc -- background reading
```bash
cd 02_magnetic_phases_doc && pdflatex magnetic_phases_ml.tex   # optional, PDF included
```
11-page document covering magnetic phases and five AI methods used to study them
(CNN classifiers, PCA/VAE, VAN, neural quantum states, PINN/GNN), with real computed
figures (e.g. PCA reproducing the Wang 2016 order-parameter result).

## 03_phase1_framing -- the theoretical bridge
```bash
cd 03_phase1_framing
python3 demo_committor.py               # seconds; regenerates figs/committor_demo.pdf
```
Read `phase1_framing.pdf`. Core result (proved, not just asserted): the transport
adjoint importance function, the rare-event committor, and the "neural importance
function" being reinvented in recent ML papers (Kim & Cai, arXiv:2602.12294) are the
same object. Tilting a Markov chain by the exact importance gives a **zero-variance**
estimator; the demo confirms this numerically (measured relative variance ~1e-30).

## 04_phase2_core -- the validated payoff
```bash
cd 04_phase2_core
python3 ww_1d.py          # ~3-5 min; 1D weight-window unit test (unbiasedness + FOM)
python3 ising_we.py       # ~30s; Weighted Ensemble vs EXACT enumeration at L=4 (gate)
python3 demo_tail.py      # ~3-4 min; L=20 rare-tail result -- the headline figure
```
**Validated:** Weighted Ensemble (= weight windows on the magnetization coordinate)
reproduces exact P(m) at L=4 to ~1-2% across the whole distribution, including rare
tails, in three regimes (above Tc, below Tc double-well, and on |m|).
**Headline result:** at L=20, matched computational cost, naive Monte Carlo has a hard
sampling floor at |m|=0.886 (P~6e-6) -- below it, zero samples, no information. Weighted
Ensemble keeps producing unbiased estimates two decades deeper, to |m|=0.932 (P~2.4e-7).
Full honesty notes (what's solid, what's still noisy, what's next) are in
`04_phase2_core/README.md`.

## 05_neural_importance -- the learned importance function (frontier)
```bash
cd 05_neural_importance
python3 neural_importance.py --gate --preset laptop         # unbiasedness gate, ~1 min
python3 neural_importance.py --preset laptop --jobs 4       # ferromagnet
python3 neural_importance.py --model ea --preset laptop --jobs 4   # spin glass
```
Learns `I_theta(s)` on the full configuration (a small periodic CNN) and uses it as the
weight-window binning coordinate, compared by figure of merit against `naive`, `WE[m]` and
`WE[E]` at matched cost. Two models: the ordinary Ising ferromagnet (where `m` is already a
good coordinate) and the Edwards-Anderson spin glass (where `m` is useless -- the case that
motivates learning a map at all). Targets are **auto-calibrated** from a pilot run to be
rare-but-reachable; use `--beyond 2` to push deeper into the tail.
**Read `05_neural_importance/README.md` before quoting any number from this stage** -- the
implementation is gated and unbiased, but the headline comparison is not yet established.

---

## Honest scope -- what is established and what is not

**Established (validated against exact references):**
- VAN reproduces the exact Onsager free energy and removes critical slowing down (stage 01).
- Adjoint importance = committor, and the exact importance gives a **zero-variance**
  estimator -- confirmed numerically to ~1e-30 relative variance (stage 03).
- Weighted Ensemble reproduces exact P(m) at L=4 to ~1-2% in three regimes, and at L=20
  reaches ~2 decades deeper into the tail than naive MC at matched cost (stage 04).
- The learned-importance implementation is **unbiased** (L=4 gate, ratios 1.000/1.001), and
  the weight-window machinery reaches targets beyond a pilot run's reach on the spin glass,
  where binning on magnetization fails in every replica -- the predicted physics (stage 05).

**NOT established (the open research):**
- **Whether a learned `I_theta(s)` actually beats hand-picked coordinates on a rugged
  landscape.** A preliminary ferromagnet run showed FOM ~13x the `WE[m]` baseline, but an
  audit showed that measurement was taken where `|m|` saturates its ceiling, so the target
  was not rare (pi ~ 4e-3) and variance reduction is not expected to help anyway -- `naive`
  beat `WE[m]` in the same run. On the spin glass at that scale the learned map still
  failed. **This is the experiment to run**, at `--preset laptop` or higher.
- A faithful committor objective. The current labels are a **monotone surrogate** (best
  value reached in short rollouts), because binary committor labels are ~all zero for rare
  targets and collapse the network. Milestoning is the fix.
- Head-to-head against a single-target (Kim & Cai-style) neural biasing or a reverse-KL VAN.
- The FW-CADIS **global** objective -- one map, controlled error across a whole family of
  thresholds -- which the Phase-1 memo identifies as the most defensible novelty.

Full detail in `04_phase2_core/README.md` and `05_neural_importance/README.md`.

---

## Bug fix log

`BUGFIXES.md` documents every bug found in a full audit (lint + manual logic review +
targeted tests), including two severe ones: `--jobs` silently running on a single core,
and a threshold calibrator that produced a non-rare target while printing a claim that it
was rare. All four validation gates reproduce their previous numbers exactly after the
fixes.

## Environment notes

- Python 3.9+ with numpy, matplotlib, torch (CPU build is fine -- see `requirements.txt`).
- `pdflatex` is only needed if you want to recompile the two `.tex` documents; the PDFs
  are already included, so this is optional.
- Everything was developed and tested on a 1-CPU / 4GB-RAM machine; runtimes above are
  from that environment; a modern laptop will typically be faster.
- `demo_tail.py`, `ww_1d.py`'s "POOR tilt" replica loop, and stage 05 are the slowest
  parts (minutes to tens of minutes) because they run many independent Monte Carlo
  replicas for honest error bars -- not because anything is wrong if it looks like it is
  hanging. Stage 05 parallelises across cores with `--jobs N`.
- Developed and tested on a **1-core** sandbox, so the runtime notes above are pessimistic;
  a typical 4-8 core laptop will be substantially faster on stage 05.

## Key references
- D. Wu, L. Wang & P. Zhang, PRL **122**, 080602 (2019) -- VAN for statistical mechanics.
- M. Kim & W. Cai, arXiv:2602.12294 (2026) -- closest neural rare-event prior art.
- J. C. Wagner & A. Haghighat, Nucl. Sci. Eng. **128**, 186 (1998) -- CADIS.
- J. C. Wagner, D. E. Peplow & S. W. Mosher, Nucl. Sci. Eng. **176**, 37 (2014) -- FW-CADIS.
