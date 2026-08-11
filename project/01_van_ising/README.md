# VAN for the 2D Ising model — joint physics + ML project scaffold

A from-scratch, validated implementation of a **Variational Autoregressive Network (VAN)**
that learns the Boltzmann distribution of the 2D Ising model and, by construction,
**eliminates critical slowing down**. This is the starting point for the joint paper:
you (physics: model, ground truth, interpretation) + your wife (ML: architecture,
optimization, scaling).

Reference method: Wu, Wang & Zhang, *Solving Statistical Mechanics Using Variational
Autoregressive Networks*, Phys. Rev. Lett. **122**, 080602 (2019).

---

## 1. The idea in one paragraph

We approximate the Boltzmann distribution `p(s) = exp(-βE(s))/Z` by a normalized
neural network `q_θ(s)` written **autoregressively**, `q_θ(s) = Π_k q_θ(s_k | s_{<k})`.
Autoregressive form buys two things a Markov chain cannot: we can **sample exactly**
(one sequential pass) and **evaluate `q_θ(s)` exactly**. We then minimize the
**variational free energy**

```
βF_q = E_{s~q}[ βE(s) + ln q(s) ] = βF_exact + KL(q ‖ p)  ≥  βF_exact
```

Because `F_q` is a **rigorous upper bound** on the true free energy, the training curve
is a certified bound — something standard MCMC does not give you directly. And because
samples are drawn i.i.d., the integrated autocorrelation time is `τ = 1` at *every*
temperature, including `T_c`.

Spins are discrete, so we cannot backpropagate through sampling. We use the
**REINFORCE / score-function** gradient with a batch-mean baseline `b`:

```
∇_θ (βF_q) = E_{s~q}[ (βE(s) + ln q(s) − b) · ∇_θ ln q_θ(s) ]
```

> **Bridge to your Geant4 work:** this is importance sampling with a *learned* proposal.
> `q_θ` plays the role of the biased sampling distribution and `ln q` is the log-weight —
> the same weight-based bookkeeping as your weight-window variance reduction, but in
> configuration space instead of particle-transport phase space. The variational bound
> is the analogue of an unbiased weighted estimator.

---

## 2. Files

| file | what it is |
|------|------------|
| `ising_van.py` | Core library: exact Onsager free energy, brute-force enumeration (validation), vectorized Ising energy, `MADE` masked autoregressive net, `VAN` wrapper, REINFORCE trainer, observable estimator. |
| `run.py` | Annealed temperature sweep → thermodynamics figure + CSV, benchmarked against Onsager. |
| `critical_slowing_down.py` | Vectorized Metropolis sampler + Sokal `τ_int` estimator → the critical-slowing-down figure. |
| `results_L8_thermo.png` / `.csv` | Demo output: F, E, C, ⟨\|m\|⟩ vs T for L=8. |
| `critical_slowing_down.png` / `.csv` | Demo output: `τ_int(T)` for Metropolis vs the flat VAN baseline. |

**Dependencies:** `torch`, `numpy`, `scipy`, `matplotlib`.

---

## 3. Run it

```bash
python run.py                       # L=8 demo, ~5 min on 1 CPU
python run.py --L 16 --steps0 1500 --steps 500 --batch 1024   # sharper, GPU recommended
python critical_slowing_down.py     # ~2 min, no neural net needed
```

Correctness is checked automatically at start-up against **exact enumeration** on a
4×4 lattice (`F_q ≥ F_exact` must hold — it does).

---

## 4. What the demo run shows (L=8, 1 CPU, ~5 min)

- **Free energy** tracks the exact Onsager curve and stays *above* it everywhere
  (the variational bound holds), closing to within ~0.005/spin at low T.
- **Energy, specific heat, magnetization** reproduce the ordering crossover; `C/N`
  shows the finite-size-rounded peak near `T_c ≈ 2.269` and `⟨|m|⟩` rises to ~1 below `T_c`.
- **Critical slowing down:** Metropolis `τ_int` spikes to ~46 sweeps at `T_c` (L=32),
  while VAN is `τ = 1` at all T. This is the figure that motivates the whole method.

Known rough edges in the scaffold (good first things to improve, all documented):
- High-T points are slightly under-converged (the hottest, highest-entropy point is
  trained from scratch). Fix: more `--steps0`, or initialize near the analytic high-T
  (uniform) solution.
- Below `T_c` the model can favor one of the two `Z₂` symmetry-broken modes
  (spin-flip symmetry). For ⟨|m|⟩ this is harmless; for a symmetry-faithful study, add
  a spin-flip-symmetrized architecture or data augmentation.

---

## 5. Division of labor (first paper)

**You (physics / ground truth):**
- Define Hamiltonians and the phase diagram to target; guarantee ground truth (Onsager,
  enumeration, and later exact diagonalization).
- Interpret the learned latent/samples physically (does `q_θ` capture the correlator,
  the transition, the correct entropy?).
- Frame the variance-reduction / importance-sampling narrative (your strength).

**Your wife (ML / systems):**
- Architecture: MADE → PixelCNN / masked-conv / transformer; symmetry-equivariant layers.
- Optimization: baselines/control variates for the REINFORCE variance, natural-gradient /
  KFAC, learning-rate and annealing schedules, batch scaling.
- Engineering: GPU batching, sampling throughput, reproducibility, larger `L`.

---

## 6. Roadmap to a publishable result

1. **Paper 1 (this codebase, scaled up).** L = 16, 32, 64; finite-size scaling of the
   free-energy bound and observables against Onsager; the critical-slowing-down
   elimination as the headline. Methodological hook = learned importance sampling +
   certified free-energy bound. Realistic venues: *Mach. Learn.: Sci. Technol.*,
   *Phys. Rev. E*, *SciPost Physics*.
2. **Paper 2 (natural extension).** 2D **XY model** and the **BKT transition** — no local
   order parameter, a topological transition; reuses ~all of this code with a different
   energy function and a `q_θ` over angles (or clock-model discretization). Higher physics
   interest, modest new ML.
3. **Stretch.** Frustrated / quantum (Neural Quantum States). Much harder (sign problem,
   symmetry) and a crowded field — do only if you find a model where a physics insight of
   yours gives an edge.

**Priority note:** keep this strictly behind the deep-penetration buildup-factor
manuscript until that is submitted. This project has working code now and can absorb
small time windows; your PhD deliverables cannot.
