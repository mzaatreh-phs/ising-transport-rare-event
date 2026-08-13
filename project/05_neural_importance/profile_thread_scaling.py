import os, sys, time
sys.path.insert(0, os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))
import numpy as np
import torch
from neural_importance import ImportanceNet

torch.manual_seed(0)
net = ImportanceNet(ch=24).eval()
L, N = 16, 2000
rng = np.random.default_rng(0)
S = rng.choice([-1.0, 1.0], size=(N, L, L)).astype(np.float32)
Xb = torch.tensor(S).unsqueeze(1)

for nt in [1, 2, 4, 8]:
    torch.set_num_threads(nt)
    with torch.no_grad():
        _ = net(Xb)  # warmup
        t0 = time.perf_counter()
        for _ in range(20):
            _ = net(Xb)
        t1 = time.perf_counter()
    print(f"threads={nt}: {1000*(t1-t0)/20:.2f} ms/call")

# mkldnn path
torch.set_num_threads(1)
net_md = net.to(memory_format=torch.contiguous_format)
try:
    with torch.no_grad():
        Xb_md = Xb.to_mkldnn()
        t0 = time.perf_counter()
        _ = torch.mkldnn_convolution
    print("mkldnn_convolution symbol present, but ImportanceNet uses F.pad(circular) which mkldnn tensors likely don't support directly -- not a drop-in fix without rewriting the model.")
except Exception as e:
    print("mkldnn path check failed:", e)
