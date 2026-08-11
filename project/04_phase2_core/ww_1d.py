"""
ww_1d.py -- Phase 2, Step 1: get weight windows genuinely working.

Goal: show that an APPROXIMATE importance map, which makes *pure* importance
tilting WORSE than doing nothing (the fragile amber bar of Phase 1), gives
large, ROBUST variance reduction once the transport community's fix -- weight
windows (splitting + Russian roulette) enforcing the CADIS consistency
condition -- is layered on top.

Model: biased nearest-neighbour walk on {0,...,N}, absorbing at 0 and N.
pi = P(reach N before 0 | s0) = h(s0), h the exact committor. We treat the
committor computed with a WRONG drift as the cheap/approximate adjoint.

Everything is validated: (a) unbiasedness (estimate == pi within error),
(b) figure of merit FOM = 1/(sigma_rel^2 * cost).
"""
import numpy as np


def committor(N, p):
    q = 1 - p
    s = np.arange(N + 1)
    if abs(p - 0.5) < 1e-12:
        return s / N
    r = q / p
    return (1 - r ** s) / (1 - r ** N)


# --------------------------------------------------------------------------- #
def naive(N, p, s0, M, rng):
    hits, steps = 0, 0
    for _ in range(M):
        s = s0
        while 0 < s < N:
            s += 1 if rng.random() < p else -1
            steps += 1
        hits += (s == N)
    pi = hits / M
    return pi, pi * (1 - pi) / M, steps


def pure_tilt(N, p, s0, M, h, rng):
    """Importance sampling by tilting the kernel with (approximate) h. No windows."""
    q = 1 - p
    est = np.empty(M); steps = 0
    for m in range(M):
        s, w = s0, 1.0
        while 0 < s < N:
            a, b = p * h[s + 1], q * h[s - 1]
            pr = a / (a + b)
            if rng.random() < pr:
                w *= p / pr; s += 1
            else:
                w *= q / (1 - pr); s -= 1
            steps += 1
        est[m] = w if s == N else 0.0
    return est.mean(), est.var(ddof=1) / M, steps


def tilt_plus_ww(N, p, s0, M, h, k=5.0, split_cap=8, rng=None):
    """CADIS: tilt the kernel with approximate importance h AND enforce weight
    windows consistent with the same h. Target weight wt(s)=h(s0)/h(s) is exactly
    what the weight would be under an *exact* tilt; the window [wt/k, wt*k] snaps
    the drifting approximate weight back by splitting (w too high) or Russian
    roulette (w too low). Unbiased; variance bounded even when h is wrong."""
    q = 1 - p
    wt = h[s0] / np.maximum(h, 1e-300)                  # consistent target weight
    scores = np.zeros(M); steps = 0
    for m in range(M):
        stack = [(s0, 1.0)]
        while stack:
            s, w = stack.pop()
            while 0 < s < N and w > 0.0:
                lo, hi = wt[s] / k, wt[s] * k
                if w > hi:                              # split
                    n = min(int(np.ceil(w / wt[s])), split_cap)
                    w /= n
                    for _ in range(n - 1):
                        stack.append((s, w))
                elif w < lo:                            # Russian roulette
                    if rng.random() < w / wt[s]:
                        w = wt[s]
                    else:
                        w = 0.0; break
                a, b = p * h[s + 1], q * h[s - 1]       # tilted step
                pr = a / (a + b)
                if rng.random() < pr:
                    w *= p / pr; s += 1
                else:
                    w *= q / (1 - pr); s -= 1
                steps += 1
            if s == N:
                scores[m] += w
    return scores.mean(), scores.var(ddof=1) / M, steps


def fom(rel_var, steps):
    return 1.0 / (rel_var * steps) if rel_var > 0 else np.inf


def replicas(method, R, N, p, s0, imp, M, rng, **kw):
    """Run R independent replicas; return per-replica estimates and total steps."""
    ests, steps = np.empty(R), 0
    for r in range(R):
        e, _, st = method(N, p, s0, M, imp, rng=rng, **kw) if imp is not None \
            else method(N, p, s0, M, rng)
        ests[r] = e; steps += st
    return ests, steps


def report(name, ests, steps, pi):
    """Reliability = spread of the estimate across INDEPENDENT replicas."""
    mean, sd = ests.mean(), ests.std(ddof=1)
    rel_var = (sd / mean) ** 2 if mean > 0 else np.inf
    steps_per = steps / len(ests)
    print(f"{name:28s} mean={mean:.4e}  rel.bias={ (mean-pi)/pi:+.3f}  "
          f"across-replica rel.sd={sd/mean:6.3f}  worst|bias|={np.abs(ests-pi).max()/pi:5.2f}  "
          f"FOM={fom(rel_var, steps_per):.2e}")
    return fom(rel_var, steps_per)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, p, s0 = 15, 0.40, 1
    h = committor(N, p); pi = h[s0]
    # A genuinely POOR cheap adjoint: guessed an UNBIASED walk (linear committor),
    # missing the drift entirely -- the realistic "our fast solve had wrong physics".
    h_bad = committor(N, 0.60); h_bad[0], h_bad[-1] = 0.0, 1.0
    print(f"Deep-penetration walk N={N}, p={p}; exact pi = {pi:.6e}")
    print("Reliability measured as the spread of the estimate over 60 independent "
          "replicas.")
    print("Approximate adjoint uses p=0.60 -- a strongly WRONG drift.\n")

    R = 60
    f0 = report("naive analog MC",
                *replicas(naive, R, N, p, s0, None, 8_000, rng), pi=pi)
    fE = report("exact-importance tilt",
                *replicas(pure_tilt, R, N, p, s0, h, 200, rng), pi=pi)
    fP = report("POOR tilt (no windows)",
                *replicas(pure_tilt, R, N, p, s0, h_bad, 2_000, rng), pi=pi)
    fW = report("POOR tilt + WEIGHT WINDOWS",
                *replicas(tilt_plus_ww, R, N, p, s0, h_bad, 2_000, rng), pi=pi)

    print("\nSame poor importance map, with vs without weight windows:")
    print(f"  pure tilt  : FOM {fP:.2e}   (unbounded weights -> unreliable)")
    print(f"  + windows  : FOM {fW:.2e}   (bounded weights   -> robust)")
    print(f"  window gain over naive: {fW/f0:.0f}x")
