#!/usr/bin/env python3
"""Drill into WHY coord_net is 233ms/call for a single tiny CNN forward pass
over a (2000,1,16,16) batch -- isolate tensor conversion, padding, each conv
layer, and check whether single-threaded is actually in effect."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
print("torch.get_num_threads() =", torch.get_num_threads())
print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())

from neural_importance import ImportanceNet

torch.manual_seed(0)
net = ImportanceNet(ch=24).eval()

L, N = 16, 2000
rng = np.random.default_rng(0)
S = rng.choice([-1.0, 1.0], size=(N, L, L)).astype(np.float32)

N_CALLS = 50

# --- full coord_net-style call, timed per-stage ---
t_tensor = t_pad1 = t_conv1 = t_pad2 = t_conv2 = t_head = t_numpy = 0.0
with torch.no_grad():
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        Xb = torch.tensor(S, dtype=torch.float32).unsqueeze(1)
        t1 = time.perf_counter(); t_tensor += t1 - t0

        h = net._pad(Xb)
        t2 = time.perf_counter(); t_pad1 += t2 - t1

        h = torch.relu(net.c1(h))
        t3 = time.perf_counter(); t_conv1 += t3 - t2

        h = net._pad(h)
        t4 = time.perf_counter(); t_pad2 += t4 - t3

        h = torch.relu(net.c2(h))
        t5 = time.perf_counter(); t_conv2 += t5 - t4

        feat = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        out = net.head(feat).squeeze(-1)
        t6 = time.perf_counter(); t_head += t6 - t5

        out_np = out.numpy().reshape(-1)
        t7 = time.perf_counter(); t_numpy += t7 - t6

total = t_tensor + t_pad1 + t_conv1 + t_pad2 + t_conv2 + t_head + t_numpy
print(f"\n{N_CALLS} calls, batch={N}:")
for label, t in [("tensor()", t_tensor), ("pad1", t_pad1), ("conv1+relu", t_conv1),
                  ("pad2", t_pad2), ("conv2+relu", t_conv2), ("head+cat", t_head),
                  (".numpy()", t_numpy)]:
    print(f"  {label:12s} {t:.4f}s total  {1000*t/N_CALLS:.2f}ms/call  {100*t/total:.1f}%")
print(f"  {'TOTAL':12s} {total:.4f}s total  {1000*total/N_CALLS:.2f}ms/call")

# --- vary batch size to see if cost is per-call-fixed or scales with N ---
print("\nscaling with batch size (single conv1 call only, no_grad):")
for bs in [1, 40, 200, 2000]:
    Xb = torch.tensor(S[:bs], dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        # warmup
        _ = net(Xb)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = net(Xb)
        t1 = time.perf_counter()
    print(f"  batch={bs:5d}: {1000*(t1-t0)/20:.3f} ms/call")
