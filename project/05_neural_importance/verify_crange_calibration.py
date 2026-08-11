#!/usr/bin/env python3
"""Empirical check of the crange fix's core assumption: that the trained
network's ACTUAL output on real configurations tracks the theoretical
z_bulk/z_target anchors derived from the label normalization (used in
run_milestone.py to calibrate WE[I_theta]'s bin range). Training is not
loss=0 (MSE 0.27-0.68), so this assumption is unverified until checked
directly. Cheap: reuses cached labels, only retrains the net (~20s).
"""
import numpy as np
import torch

from neural_importance import make_couplings, sweep_population, ImportanceNet, PRESETS

torch.set_num_threads(1)

cfg = PRESETS['custom'].copy()
L, T = cfg['L'], cfg['T']
model = 'ea'
net_seed = 1  # matches results_fix_seed1.json / results_seed1.json for direct comparison

Jx, Jy = make_couplings(L, model, seed=0)

d = np.load('.label_cache/labels_ea_L16_T2.6_b1_n2500.npz')
S_train, labels = d['S'], d['labels']
mean_lab, std_lab = labels.mean(), labels.std()
z_bulk = (0.0 - mean_lab) / std_lab
z_target = (1.0 - mean_lab) / std_lab
print(f"theoretical anchors: z_bulk={z_bulk:.3f}  z_target={z_target:.3f}  "
      f"(from mean_lab={mean_lab:.4f}, std_lab={std_lab:.4f})")

torch.manual_seed(net_seed)
net = ImportanceNet(ch=cfg['channels'])
labels_std = (labels - mean_lab) / std_lab
X = torch.tensor(S_train, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(labels_std, dtype=torch.float32)
optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
loss_fn = torch.nn.MSELoss()
net.train()
n_train = len(S_train)
for epoch in range(cfg['epochs']):
    perm = torch.randperm(n_train)
    X_shuf, Y_shuf = X[perm], Y[perm]
    for i in range(0, n_train, cfg['batch']):
        optimizer.zero_grad()
        pred = net(X_shuf[i:i + cfg['batch']])
        l = loss_fn(pred, Y_shuf[i:i + cfg['batch']])
        l.backward()
        optimizer.step()
net.eval()
print("training done")


def coord_net(S):
    with torch.no_grad():
        Xb = torch.tensor(S, dtype=torch.float32).unsqueeze(1)
        return net(Xb).numpy().reshape(-1)


# --- Actual output on S_train (spans label 0.2448-1.0, i.e. shallow-to-deep
# starting configs already used in training -- an in-sample check). ---
I_train = coord_net(S_train)
print(f"\nI_theta(S_train): min={I_train.min():.3f} max={I_train.max():.3f} "
      f"mean={I_train.mean():.3f} sd={I_train.std():.3f}")
print(f"  (label range was [{labels.min():.3f},{labels.max():.3f}], "
      f"so z-range should be roughly "
      f"[{(labels.min()-mean_lab)/std_lab:.3f}, {(labels.max()-mean_lab)/std_lab:.3f}])")

# --- Fresh equilibrium (genuinely bulk / label~0) population, independent of
# training data -- the real "bulk" endpoint of the crange. ---
rng = np.random.default_rng(seed=777)
S_bulk = rng.choice([-1, 1], size=(cfg['collect_walkers'], L, L))
for _ in range(150):
    S_bulk = sweep_population(S_bulk, T, Jx, Jy, rng)
I_bulk = coord_net(S_bulk)
print(f"\nI_theta(fresh equilibrium bulk sample): min={I_bulk.min():.3f} "
      f"max={I_bulk.max():.3f} mean={I_bulk.mean():.3f} sd={I_bulk.std():.3f}")
print(f"  theoretical z_bulk anchor = {z_bulk:.3f}")

lo_I, hi_I = (z_bulk, z_target) if z_bulk < z_target else (z_target, z_bulk)
margin_I = 0.15 * (hi_I - lo_I)
crange_used = (lo_I - margin_I, hi_I + margin_I)
print(f"\ncrange actually used by run_milestone.py: {crange_used}")

actual_lo = min(I_bulk.min(), I_train.min())
actual_hi = max(I_bulk.max(), I_train.max())
print(f"actual observed I_theta range across both samples: "
      f"[{actual_lo:.3f}, {actual_hi:.3f}]")
covered = actual_lo >= crange_used[0] and actual_hi <= crange_used[1]
print(f"\nVERDICT: observed range {'IS' if covered else 'is NOT'} fully inside "
      f"the calibrated crange.")
if not covered:
    print("  -> the crange fix's assumption does not hold cleanly for this seed; "
          "the fixed bins may clip real excursions at one or both ends.")
