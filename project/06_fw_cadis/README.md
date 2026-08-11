# Stage 6 — FW-CADIS: global variance reduction across a threshold family

**This is the novelty claim.** The Phase-1 memo identified it, the prior-art recheck
confirmed it is still unoccupied (Kim & Cai are single-threshold by construction), and
nothing else in this project attempts it.

```bash
python3 fw_cadis.py --gate                                  # unbiasedness, seconds
python3 fw_cadis.py --gate --model ea
python3 fw_cadis.py --model ea --preset laptop              # the comparison
python3 fw_cadis.py --model ea --preset full --replicas 30 --jobs 8
```

## The idea

Stages 4 and 5 optimise for **one** rare target. The resulting weight-window ladder is
tuned to that depth — it over-serves the bulk and under-serves anything deeper. Ask for
five thresholds and you need five runs, or one run whose relative error explodes with
depth.

Transport solved this in 2014 (FW-CADIS, Wagner–Peplow–Mosher): weight the adjoint source
by the **inverse of the local forward response**, so every region is sampled to comparable
*relative* precision rather than comparable absolute weight.

Translated here:

| transport | this implementation |
|---|---|
| forward response R(b) | equilibrium bin probability P(b), from a cheap pilot |
| inverse-response weighting | walkers per bin ∝ (1/P(b))^α |
| one map, many detectors | one ladder, **all thresholds tallied from the same walkers** |

`--alpha 0` is uniform allocation — exactly the Stage 4/5 baseline. `--alpha 1` is full
FW-CADIS. **Same code, two settings**, which is what makes the comparison fair.

## The metric is coverage and flatness, not FOM

A single figure of merit answers "how well did you sample one target". That is the wrong
question here. The metric is:

1. **Coverage** — how many thresholds in the family were observed at all. Judged first.
2. **Flatness** — spread (max/min and sd of log) of relative error across the observed
   thresholds.

**A subtlety found while testing, worth knowing about.** A method that fails to observe a
deep threshold has infinite error there. If those are dropped before computing flatness,
the failing method is scored *only on the easy thresholds it could see* — and looks
**flatter** than a method that reached everything with a modest error gradient. That
rewards failure. The smoke test showed exactly this: uniform allocation missed the deepest
threshold and scored max/min = 30, while FW-CADIS, which reached every threshold, scored
213. Coverage is therefore reported and judged first, and the number of unobserved
thresholds is always printed.

## Validated

**Unbiasedness gate (L=4, exact enumeration), both allocations, every threshold:**

| model | α=0 (uniform) ratios | α=1 (FW-CADIS) ratios |
|---|---|---|
| ferro | 0.999, 0.998, 0.989 | 1.005, 1.012, 0.992 |
| ea | 1.009, 1.004, 1.007 | 0.986, 0.980, 0.983 |

All within ~2% of exact. Weight-conserving split/merge is unbiased for *any* allocation,
so this validates the implementation — including that **all thresholds are correctly
tallied from a single run**, which is the core mechanic.

**Preliminary signal (smoke preset, L=8, 4 replicas — a code check, not a result):**
FW-CADIS observed **6/6** thresholds where uniform allocation observed **5/6**, at roughly
**one-sixth the cost** (7.6e5 vs 4.6e6). It reached a threshold with π ≈ 2.1e-5 that
uniform never saw at all.

## Result (2026-08-04, L=16 EA, `--preset full`, every lever tried)

**FW-CADIS does not measurably flatten error better than uniform allocation on this
problem — but runs at roughly 1/8th the cost.** Every variant tried converges on the same
honest answer: bootstrap 95% CIs on the max/min flatness metric always overlap between
`uniform` and `FW-CADIS`, so "flatter" is not established. Point estimates consistently
favor `uniform` (lower max/min = flatter), not `FW-CADIS`, across every config:

| config | uniform max/min [CI] | FW-CADIS max/min [CI] | cost ratio (FW-CADIS/uniform) |
|---|---|---|---|
| `--beyond 1`, R=30, α=1.0 | 298.01 [177,478] | 385.54 [259,700] | ~0.13 |
| `--beyond 1`, R=30, α=0.5 | 298.01 [177,478] | 323.91 [216,626] | — |
| `--beyond 1`, R=90, α=1.0 | — | 405.05 [296,560] | — |
| `--beyond 1`, R=30, α=1.0 + learned coord | — | 509.61 [282,921] | ~2x cost of physical coord |
| `--beyond 0`, R=30, α=1.0 | 250.67 [188,332] | 356.30 [235,581] | — |
| `--beyond 0`, R=90, α=1.0 | 276.09 [224,344] | 297.29 [234,388] | **~0.13** |

**Levers tried, all exhausted:**
- **Bootstrap CIs on flatness** — implemented (`bootstrap_spread`), used throughout.
- **α sweep** (0.5, 1.0) — no config crosses significance.
- **Learned coordinate as the binning axis** (`--coord learned`, wired via `NetCoord`) —
  no improvement, ~2x cost. Gated against exact enumeration at L=4, ratios 0.997–1.010.
- **Common random numbers + paired-difference bootstrap** — implemented
  (`bootstrap_paired_diff`), but CRN doesn't engage: `uniform` and `FW-CADIS` allocate a
  different walker count per bin, so their RNG streams desync almost immediately
  (corr ≈ -0.05 to -0.10 at every ladder depth tried, including the shallowest). The
  paired CI never usefully tightens below the independent-CI estimate.
- **Laddering the threshold depth** (`--beyond 0`, i.e. the shallowest calibrated target)
  — the one lever that measurably works. CI widths on both methods and on the paired
  difference shrink monotonically going `beyond=1` → `beyond=0/R=30` → `beyond=0/R=90`,
  and the point-estimate gap between the two methods shrinks too (106 apart → 21 apart).
  But it does not fully close: paired diff at the best config is still
  `+20.92 [-77.2, +137.6]` (includes zero).
- **CI-width scaling was checked**: going R=30→90 (3x) shrank the paired-diff CI width by
  almost exactly `sqrt(30/90) = 0.577` (actual: 0.597) — the statistic behaves like normal
  1/√R scaling here, not the slower extreme-value convergence originally feared. Fully
  closing the remaining gap under that scaling would need **R ≈ 2300+** — several hours of
  compute for a result that, if it resolved, would most likely confirm `uniform` as
  measurably flatter, not `FW-CADIS`. **Not worth running**; the honest conclusion above is
  the one to report.

**Bottom line for the paper:** FW-CADIS's actual demonstrated advantage here is *cost*,
not flatness — it reaches the same (statistically indistinguishable) error profile at
~1/8th the simulation cost of uniform allocation. That is a real, defensible, already-
established result and does not need more replicas to state.

## Root-cause test (2026-08-05): is the flattening failure spin-glass-specific?

Ran the identical comparison on the **ferromagnet** (`--model ferro`, known-good
coordinate, smooth landscape) instead of the EA spin glass, at `--beyond 0 --replicas 30`:

| config | uniform max/min [CI] | FW-CADIS max/min [CI] | cost |
|---|---|---|---|
| ferro, `--beyond 0`, R=30 | **100.91** [65.8, 160.3] | 287.30 [194.7, 414.8] | 3.09e7 (uniform 1.96e8) |

**Statistically resolved, and the opposite direction from what "the spin glass is just
hard" would predict.** The paired-difference CI (+178.20 [+59.8, +351.3]) *excludes zero*
— on the easy landscape, FW-CADIS is *measurably confirmed worse* than uniform, something
the spin-glass tests never reached in either direction. **This rules out landscape
difficulty as the explanation.** The remaining candidates are the bin-count/threshold-
spacing parameterization or a more general limit of the inverse-response weighting at this
lattice size — distinguishing those needs a parameter sweep independent of α and the model,
not another problem instance.
