# Stage 5 — Learned importance function in full configuration space

This is the component the Phase-1 memo argued for and Phase 2 deliberately did not do.
Phase 2 put weight windows on a **hand-picked 1-D coordinate** (the magnetization).
Here we **learn** an importance function `I_theta(s)` on the whole configuration and use
it as the binning coordinate for the same weight-window machinery, then compare by
figure of merit.

```bash
python3 neural_importance.py --gate --preset laptop        # validation gate, ~1 min
python3 neural_importance.py --preset laptop --jobs 4      # ferromagnet
python3 neural_importance.py --model ea --preset laptop --jobs 4   # spin glass
python3 neural_importance.py --model ea --preset full --jobs 8     # publication scale
```

## What is compared

All four methods estimate the *same* rare probability at matched computational cost:

| method | binning coordinate | role |
|---|---|---|
| `naive` | — | analog Monte Carlo baseline |
| `WE[m]` | magnetization | the Phase-2 baseline |
| `WE[E]` | energy | better baseline for the spin glass |
| `WE[I_theta]` | **learned** | the contribution under test |

`FOM = 1 / (rel_var * cost)`, with `rel_var` measured across independent replicas.

Two models: `ferro` (ordinary 2D Ising, target `|m| >= m*` — here **m is already a good
coordinate**) and `ea` (Edwards–Anderson ±J spin glass, target `E/N <= e*` — here **m is
useless**, since a spin glass has m ≈ 0). The spin glass is the point of the exercise: a
learned map should win precisely where the hand-picked coordinate is a poor descriptor.

## What is validated

**Unbiasedness gate (`--gate`).** At L=4 the answer is available by exact enumeration.
Weight windows are unbiased for *any* binning coordinate, good or bad, so this checks the
implementation while the FOM checks coordinate quality. Measured: `WE[m]` ratio to exact
**1.000**, `WE[E]` **1.001**. Note honestly that a 16-spin system has no truly rare events
(π cannot go far below ~0.1), so the gate validates *unbiasedness only*.

**The machinery reaches targets beyond a pilot run's reach.** On the spin glass with a
target set one discrete step beyond the most extreme energy a pilot of 25,600 samples ever
saw (`E/N <= -1.3125`, rarer than ~4e-5), `WE[E]` returned a finite estimate (~2e-5) while
`WE[m]` failed in **every** replica — exactly the predicted physics, since magnetization
carries no information about a spin glass's low-energy states.

**Targets are auto-calibrated, not hard-coded.** An early version hard-coded
`E/N <= -1.55` for the spin glass. That is **below the ground state** (≈ -1.35 for this
disorder sample), so the probability is exactly zero and every method silently returns 0.
`calibrate_threshold()` now runs a short pilot and sets the target one discrete lattice
step beyond the most extreme value observed, which guarantees rare-but-reachable at
whatever L and T you choose. Use `--beyond 2` (or more) to push further into the tail.

## Preliminary observation — NOT a result

On the ferromagnet at `--preset smoke` (L=8, 3 replicas), the learned coordinate gave
**FOM ≈ 13x the `WE[m]` baseline** with the smallest replica spread (rel.sd 0.044 vs
0.156). Do **not** quote this. A later audit (see `../BUGFIXES.md`, bug #2) established
that this measurement is weaker than it first appeared:

- **The target was not rare at all.** `|m|` is capped at 1.0, and at L=8 the pilot run
  already reaches 1.0, so the "one step beyond the pilot" rule clamped to a threshold the
  pilot had *already hit* — giving π ≈ 4e-3. The calibrator now prints an explicit
  saturation warning in exactly this case; at `--preset smoke` you will always see it.
- Consistent with that, `naive` also beat `WE[m]` (5.9x) in the same run — the signature
  of a target where variance reduction cannot pay for its overhead. Variance reduction is
  not *supposed* to win on a common event.
- **3 replicas** is far too few for a stable variance estimate.

So the 13x was measured in the regime where the whole method is expected to be pointless.
A real comparison needs L >= 12 (where calibration does not saturate) and a target several
decades rarer — use `--preset laptop`/`full`, and raise `--beyond`.
- On the **spin glass** at smoke scale the learned map still failed (0/3 replicas) while
  `WE[E]` succeeded. The network barely trained there (MSE 0.945 vs 1.0 for "predict the
  mean") because the training set was tiny. Whether it wins at `laptop`/`full` scale is
  **an open question this project has not answered.**

So: the component is implemented, gated, and produces sensible numbers, but **the headline
claim — a learned importance map beating hand-picked coordinates on a rugged landscape —
is not established.** That is the experiment to run on your machine. Run it at
`--preset laptop` or higher: the `smoke` preset is a code check only, and its physics is
explicitly degenerate (see the saturation warning it prints).

## How `I_theta` is trained

1. Collect configurations along a crude weight-window run, so the training set covers the
   tail rather than only the bulk.
2. From each configuration, run `n_roll` short rollouts of `roll_K` sweeps and record the
   **best (most extreme) value reached** — a dense, monotone surrogate for the committor.
3. Fit a small periodic-padding CNN by MSE regression on standardized labels.

**Honest caveat on step 2.** The Phase-1 identity is about the *committor* — the
probability of reaching the target before returning to the source. Binary committor labels
are unusable here: for a genuinely rare target essentially no short rollout ever arrives,
so labels are ~all zero and the network collapses to "never" (measured: `frac>0 = 0.003`,
FOM 0, complete failure). The best-value-reached label is a **surrogate**: monotonically
related to the committor and dense enough to train on, but not the committor itself. A
more faithful implementation would use milestoning / nested intermediate thresholds so
each level's committor is estimable. That is the main methodological gap remaining.

## Presets

| preset | L | replicas | rough cost |
|---|---|---|---|
| `smoke` | 8 | 3 | seconds — checks the code runs |
| `laptop` | 12 | 6 | ~20–40 min on 4 cores |
| `full` | 16 | 10 | hours; publication scale |

`--jobs N` parallelises replicas across cores (the naive and fixed-coordinate methods;
the learned-coordinate runs stay single-process because the torch model does not pickle
cheaply across workers).

## Result: Kim & Cai head-to-head (2026-08-04, L=16 EA, `--preset full --beyond 0`)

`--objective selfconsistent` implements the Kim & Cai (arXiv:2602.12294) label-free
training recipe (`train_importance_selfconsistent()`) as an alternative to the surrogate
objective above. **Found and fixed a real bug before this result was trustworthy**: the
training-data collection pass originally used `cfg["collect_iter"]/["collect_walkers"]`
(tuned for the surrogate objective's rollout labels, which only need broad tail coverage)
— far shallower than the actual WE depth, so it never collected a state within one
single-spin-flip of the target. Self-consistency alone can't rule out the trivial
constant solution `I(s)=c` without at least one such boundary state (see the long comment
in `train_importance_selfconsistent`), so training silently collapsed to a near-flat
coordinate (measured sd=0.137 across the training set) while reporting a deceptively
small loss. Confirmed this was structural, not a ladder-depth artifact: failed identically
at L=12 and L=16, `beyond=0` and `beyond=1`. **Fix**: collect at the same depth as the
real WE evaluation (`cfg["we_iter"]` iterations, `cfg["n_bins"]*cfg["n_per_bin"]`
walkers) instead of the cheaper rollout-tuned pass — now finds boundary states reliably
(44 found vs 0 before, at full scale) and trains a properly differentiated coordinate
(sd=0.873, range [-7.7, -0.04]).

**With the fix, in the clean regime (0/10 zero-replicas for every method) at `--beyond 0`:**

| method | pi_hat | rel.sd | FOM | zero-replicas |
|---|---|---|---|---|
| naive | 3.357e-05 | 0.294 | 8.595e-08 | 0/10 |
| WE[m] | 3.944e-05 | 0.553 | 8.814e-08 | 1/10 |
| WE[E] | 3.389e-05 | 0.671 | 1.327e-08 | 0/10 |
| WE[I_theta] surrogate | 2.922e-05 | 0.904 | 8.624e-09 | 1/10 |
| **WE[I_theta] self-consistent (fixed)** | 4.013e-05 | 0.655 | **1.616e-08** | **0/10** |

The point estimate favors the Kim & Cai-style coordinate over both the hand-picked energy
coordinate (`WE[E]`, +22% FOM) and this project's own surrogate-label coordinate (+87%
FOM), and it's the most reliable WE-based method here (0/10 zero-replicas, tied with
`WE[E]` and better than both `WE[m]` and the surrogate). `WE[m]` and `naive` still have the
best raw FOM — expected, `beyond=0` is the shallowest calibrated depth, not yet deep enough
for coordinate quality to dominate cost. Training cost is real: ~4560s (76 min) total at
`full` preset, vs 245s for the surrogate objective (~170x slower per epoch, from the extra
N=L*L forward passes per training configuration the self-consistency loss needs).

**Bootstrapped 2026-08-05 (rerun with per-replica values saved, `bootstrap_fom_ratio` in
`neural_importance.py`): the gap is NOT statistically significant at R=10.** FOM ratio
self-consistent/surrogate = 1.83, 95% CI [0.49, 9.23] — includes 1.0. FOM ratio
self-consistent/WE[E] = 1.19, 95% CI [0.33, 4.95] — also includes 1.0.

**Re-run at R=30 (tripled replicas, same trained coordinate — only the eval step re-runs at
higher R, so this cost ~6 more minutes on top of the ~76min training already paid):
CI tightened a lot, and the point estimate itself shrank.** FOM ratio
self-consistent/surrogate: 1.83 → **1.39**, CI [0.49,9.23] → **[0.65, 2.95]**. FOM ratio
self-consistent/WE[E]: 1.19 → **1.07**, CI [0.33,4.95] → **[0.55, 2.09]**. Still includes
1.0 at R=30 — not yet established — but the shrinking point estimate as replicas increase
is itself informative: it suggests the original R=10 "+87%"/"+22%" numbers were inflated by
noise, and the true effect (if any) is likely much more modest than first measured. Same
lesson as `06_fw_cadis`'s flatness metric, reinforced: a point-estimate FOM ratio from few
replicas is not enough to call a winner, especially for a ratio-of-variances statistic.
**Honest current state: the self-consistent coordinate is still the best point estimate
among WE-based methods and among the most reliable, but a real, established FOM advantage
over the surrogate objective remains unproven — the evidence for it has gotten weaker, not
stronger, as more data came in.**

**`--net-cache PATH` added (2026-08-05)**: training is deterministic (fixed seed=0), so
re-running only to bump `--replicas` was repaying the full ~76min training cost for a
bit-identical network every time. This flag saves the trained net's `state_dict()` on
first run and loads it on subsequent runs with the same path, skipping training entirely
(verified at smoke scale: second run dropped from 14s to 3s, bit-identical results). Makes
further replica increases on the self-consistency objective essentially free.

**Re-run at `--beyond 1` (2026-08-05): confirms the fix holds deeper, and the point-estimate
gap shrank rather than grew.** Collection-depth fix verified again at this deeper target (26
boundary states found immediately, no recurrence of the warning). Result: `WE[I_theta]`
self-consistent FOM=6.068e-09 (1/10 zeros) vs surrogate 5.785e-09 (2/10 zeros) vs `WE[E]`
5.000e-09. Bootstrapped: ratio self-consistent/surrogate = 1.14, 95% CI [0.27, 4.81] — not
significant, and notably *smaller* than the beyond=0/R=10 ratio (1.83) — the opposite of the
"advantage grows where naive/surrogate struggle" hypothesis. What *did* hold at both depths:
self-consistent has the fewest zero-replicas of any WE-based method every time tested — the
one consistently-observed advantage across this whole investigation.

## Next steps
1. Even more replicas (R=30 CI still includes 1.0, though much tighter than R=10) if a
   definitive answer is wanted — now cheap via `--net-cache`, no need to repay training.
2. ~~Re-run at `--beyond 1` or deeper~~ — done 2026-08-05, see above. Could push to `--beyond
   2` next, though naive/WE[m] both start failing badly there per the milestoning results.
3. `04_phase2_core` and `06_fw_cadis` already cover milestoning and the FW-CADIS global
   objective — see those READMEs.
