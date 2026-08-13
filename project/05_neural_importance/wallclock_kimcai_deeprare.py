#!/usr/bin/env python3
"""Extend the wall-clock-fair FOM recomputation to Kim & Cai and deep-rare
(beyond=4 self-consistency) -- the two other sections of the paper using
WE[I_theta] FOM numbers, per user request to finish checking before any
manuscript edit."""
import json, re, os, sys
import numpy as np
os.chdir(os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))
sys.path.insert(0, os.getcwd())
from neural_importance import fom_from

method_re = re.compile(r'^\s*(naive|WE\[m\]|WE\[E\]|WE\[I_theta\])\s+.*\[(\d+)s\]\s*$')
order = ['naive', 'WE[m]', 'WE[E]', 'WE[I_theta]']

def parse_log(logfile):
    cum = {}
    for l in open(logfile):
        m = method_re.match(l)
        if m and m.group(1) not in cum:
            cum[m.group(1)] = float(m.group(2))
    deltas = {}
    prev = 0.0
    for meth in order:
        if meth in cum:
            deltas[meth] = cum[meth] - prev
            prev = cum[meth]
    return deltas

def load_rows(jsonfile):
    d = json.load(open(jsonfile))
    return {row['method']: row for row in d['rows']}

def report_pair(label, logfile, jsonfile):
    deltas = parse_log(logfile)
    rows = load_rows(jsonfile)
    est_I = np.array(rows['WE[I_theta]']['est'])
    est_E = np.array(rows['WE[E]']['est'])
    cost_I_mc, cost_E_mc = rows['WE[I_theta]']['cost'], rows['WE[E]']['cost']
    wc_I, wc_E = deltas['WE[I_theta]'], deltas['WE[E]']

    fom_I_mc, _, _ = fom_from(est_I, cost_I_mc)
    fom_E_mc, _, _ = fom_from(est_E, cost_E_mc)
    fom_I_wc, _, _ = fom_from(est_I, wc_I)
    fom_E_wc, _, _ = fom_from(est_E, wc_E)

    print(f"{label:34s} wc(E)={wc_E:7.1f}s wc(I)={wc_I:8.1f}s time_ratio={wc_I/wc_E:5.2f}x  "
          f"FOM ratio I/E: MC-cost={fom_I_mc/fom_E_mc:6.3f}  wall-clock={fom_I_wc/fom_E_wc:6.3f}")
    return dict(label=label, est_I=est_I, est_E=est_E, wc_I=wc_I, wc_E=wc_E,
                ratio_mc=fom_I_mc/fom_E_mc, ratio_wc=fom_I_wc/fom_E_wc)

def bootstrap_single(row, R, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        rI = row['est_I'][rng.integers(0, R, size=R)]
        rE = row['est_E'][rng.integers(0, R, size=R)]
        fI, _, _ = fom_from(rI, row['wc_I'])
        fE, _, _ = fom_from(rE, row['wc_E'])
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    ratios = np.array(ratios)
    lo, med, hi = np.percentile(ratios, [2.5, 50, 97.5])
    return lo, med, hi

print("=== Kim & Cai, beyond=0, R=10 ===")
sc10 = report_pair("self-consistent (kimcai_sc_b0_boot)", "kimcai_sc_b0_boot.log",
                    "results_kimcai_sc_b0_boot_ea_L16_full_selfconsistent.json")
surr10 = report_pair("surrogate (kimcai_surrogate_b0_boot)", "kimcai_surrogate_b0_boot.log",
                      "results_kimcai_surrogate_b0_boot_ea_L16_full.json")
lo, med, hi = bootstrap_single(sc10, 10)
print(f"  self-consistent vs WE[E], wall-clock-fair bootstrap (R=10): median={med:.3f} CI=[{lo:.3f},{hi:.3f}]")

print("\n=== Kim & Cai, beyond=0, R=30 (the paper's headline R=30 comparison) ===")
sc30 = report_pair("self-consistent (kimcai_sc_b0_r30)", "kimcai_sc_b0_r30.log",
                    "results_kimcai_sc_b0_r30_ea_L16_full_selfconsistent.json")
surr30 = report_pair("surrogate (kimcai_surrogate_b0_r30)", "kimcai_surrogate_b0_r30.log",
                      "results_kimcai_surrogate_b0_r30_ea_L16_full.json")
lo, med, hi = bootstrap_single(sc30, 30)
print(f"  self-consistent vs WE[E], wall-clock-fair bootstrap (R=30): median={med:.3f} CI=[{lo:.3f},{hi:.3f}]")
print(f"  (MC-cost basis published: 1.07x, CI [0.55,2.09])")

print("\n=== Kim & Cai, beyond=1, R=10 ===")
sc_b1 = report_pair("self-consistent (kimcai_sc_b1)", "kimcai_sc_b1.log",
                     "results_kimcai_sc_b1_ea_L16_full_selfconsistent.json")
surr_b1 = report_pair("surrogate (kimcai_surrogate_b1)", "kimcai_surrogate_b1.log",
                       "results_kimcai_surrogate_b1_ea_L16_full.json")

print("\n=== Deep-rare beyond=4, self-consistency, R=10, 3 seeds ===")
d0 = report_pair("seed0 (selfconsistent_beyond4)", "selfconsistent_beyond4.log",
                  "results_selfconsistent_deep_ea_L16_full_selfconsistent.json")
d1 = report_pair("seed1 (selfconsistent_beyond4_seed1)", "selfconsistent_beyond4_seed1.log",
                  "results_selfconsistent_deep_seed1_ea_L16_full_selfconsistent.json")
d2 = report_pair("seed2 (selfconsistent_beyond4_seed2)", "selfconsistent_beyond4_seed2.log",
                  "results_selfconsistent_deep_seed2_ea_L16_full_selfconsistent.json")

print("\n=== Deep-rare beyond=4, self-consistency, R=30, 3 seeds (paper's headline) ===")
d0_30 = report_pair("seed0_r30", "selfconsistent_beyond4_seed0_r30.log",
                     "results_selfconsistent_deep_seed0_r30_ea_L16_full_selfconsistent.json")
d1_30 = report_pair("seed1_r30", "selfconsistent_beyond4_seed1_r30.log",
                     "results_selfconsistent_deep_seed1_r30_ea_L16_full_selfconsistent.json")
d2_30 = report_pair("seed2_r30", "selfconsistent_beyond4_seed2_r30.log",
                     "results_selfconsistent_deep_seed2_r30_ea_L16_full_selfconsistent.json")

# two-level bootstrap across the 3 deep-rare R=30 seeds, wall-clock-fair
rng = np.random.default_rng(0)
rows3 = [d0_30, d1_30, d2_30]
N_BOOT = 10000
boot = []
for _ in range(N_BOOT):
    drawn = rng.integers(0, 3, size=3)
    ratios = []
    for i in drawn:
        r = rows3[i]
        rI = r['est_I'][rng.integers(0, 30, size=30)]
        rE = r['est_E'][rng.integers(0, 30, size=30)]
        fI, _, _ = fom_from(rI, r['wc_I'])
        fE, _, _ = fom_from(rE, r['wc_E'])
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    if ratios:
        boot.append(np.exp(np.mean(np.log(ratios))))
boot = np.array(boot)
lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])
print(f"\nDeep-rare beyond=4, 3-seed pooled, wall-clock-fair (geometric mean): "
      f"median={med:.3f} 95% CI=[{lo:.3f},{hi:.3f}]  N={len(boot)}")
print(f"(MC-cost basis published: 1.16x, CI [0.65,2.12])")
print(f"excludes 1.0? {'YES' if lo>1.0 or hi<1.0 else 'NO'}")

# also report reliability (hit-rate) numbers -- unaffected by cost, sanity check they're unchanged
print("\n--- reliability (hit-rate) check, cost-independent, should be unaffected ---")
for label, jf in [("seed0_r30", "results_selfconsistent_deep_seed0_r30_ea_L16_full_selfconsistent.json"),
                   ("seed1_r30", "results_selfconsistent_deep_seed1_r30_ea_L16_full_selfconsistent.json"),
                   ("seed2_r30", "results_selfconsistent_deep_seed2_r30_ea_L16_full_selfconsistent.json")]:
    rows = load_rows(jf)
    est_I = np.array(rows['WE[I_theta]']['est'])
    est_E = np.array(rows['WE[E]']['est'])
    print(f"  {label}: WE[I_theta] hits={np.sum(est_I>0)}/{len(est_I)}  WE[E] hits={np.sum(est_E>0)}/{len(est_E)}")
