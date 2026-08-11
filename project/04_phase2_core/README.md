# Phase 2 — technical core (validated baseline)

Weight-window / reaction-coordinate variance reduction for rare events, built and
verified on exactly-solvable references. This is the **non-neural baseline** the
neural importance-function method (next stage) must beat, plus the unit tests that
de-risk the machinery.

## Files
- `ww_1d.py` — 1D deep-penetration walk. Unit test for the weight-window code.
- `ising_we.py` — 2D Ising rare-magnetization sampler by Weighted Ensemble
  (= weight windows on the magnetization coordinate = reaction-coordinate CADIS),
  with exact enumeration for validation. **Run this to reproduce the L=4 gate.**
- `demo_tail.py` — L=20 demonstration on the monotone one-sided tail P(|m|≥m*).
- `figs/ising_tail.pdf` — the result figure. `tail_data.npz` — its data.

## What is validated

**1. The weight-window estimator is correct (1D unit test).**
With the *exact* importance (committor), tilting the kernel gives a **zero-variance**
estimator — measured relative variance ~1e-30, every path carries the identical
weight h(s₀). With a right-shaped *approximate* importance, the tilt-plus-weight-window
scheme is **unbiased** (large-run relative bias 0.01%) and reliable (2.4% spread across
replicas). Pure tilting without windows has *unbounded* weights and is unreliable
(one replica looks great, another blows up) — which is exactly why weight windows exist.

**2. Weighted Ensemble for the 2D Ising model is unbiased (gate vs exact).**
At L=4 the magnetization distribution P(m) is available by exact enumeration. WE
reproduces it to ~1% across the whole range in three regimes:

| regime | mean \|Δ ln P\| |
|---|---|
| T=2.6, signed m (above Tc) | 0.006 |
| T=2.0, double-well (below Tc, incl. barrier) | 0.024 |
| T=2.6, \|m\| coordinate | 0.007 |

The resampling is systematic split/merge to a fixed number of walkers per bin —
weight-conserving, hence unbiased. This is the transport split/Russian-roulette
operation applied along a reaction coordinate.

**3. It reaches ~2 decades deeper into the tail than naive MC (L=20, T=2.6).**
At matched cost (~3.4e8 spin-sweeps, 4 replicas each):

- naive MC has a **hard sampling floor** at |m|=0.886 (P≈6×10⁻⁶); below it, zero samples.
- Weighted Ensemble keeps producing unbiased estimates to |m|=0.932 (P≈2.4×10⁻⁷),
  and agrees with naive everywhere they overlap.

So where naive returns *nothing*, WE returns an accessible, unbiased estimate.

## Honest limitations (do not overclaim)

- **The deep-tail estimate is noisy, not precise.** WE's relative error grows into the
  tail (≈0.16 at |m|=0.84, ≈0.56 at 0.89, ≈2.0 at 0.93 with only 4 replicas × 10
  walkers/bin). "Two decades deeper" means *nonzero and unbiased*, roughly
  order-of-magnitude — not a tight value. More walkers/replicas tighten it; the earlier
  "uniform relative error" framing holds only in the bulk-to-mid-tail, not the deepest bins.
- **This is a 1-D reaction coordinate (magnetization).** It is the principled
  transport-style baseline, *not* the neural contribution. The novelty claimed in the
  Phase-1 memo — a learned importance I_θ(s) in full configuration space, FW-CADIS global
  windows across a *family* of thresholds, and the FOM as training objective — is the
  next stage and is not done here.
- **No head-to-head yet** against a single-target (Kim–Cai-style) biasing or against a
  reverse-KL VAN. The FOM comparison that the paper needs is future work.

## Next steps (Phase 2 continued)
1. Learn I_θ(s) (a small autoregressive / CNN importance) in full config space; compare
   its FOM to this magnetization-coordinate baseline — the neural map should beat the 1-D
   coordinate when the reaction coordinate is a poor descriptor.
2. FW-CADIS *global* windows: one importance map giving controlled error across a family
   of thresholds / the whole rate function; report the global uniform-error metric.
3. Train by maximizing FOM = 1/(σ²_rel·T) under the consistency condition; compare to
   reverse-KL.
4. Robustness on the Edwards–Anderson spin glass (rugged landscape where a 1-D coordinate
   and plain neural samplers both struggle).
5. Also available: the below-Tc barrier-crossing variant (magnetization reversal) as a
   second rare-event testbed.
