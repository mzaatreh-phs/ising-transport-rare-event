"""
ising_van.py
============
Variational Autoregressive Networks (VAN) for the 2D Ising model.

Method (Wu, Wang & Zhang, PRL 122, 080602 (2019)):
  Represent the Boltzmann distribution p(s) = exp(-beta E(s)) / Z by a
  normalized, autoregressive neural network q_theta(s) that we can (a) sample
  from exactly and (b) evaluate the probability of exactly.  Train by minimizing
  the *variational free energy*

        beta F_q = E_{s~q}[ beta E(s) + ln q(s) ]
                 = beta F_exact + KL(q || p)   >=   beta F_exact.

  Because F_q is a rigorous upper bound on the true free energy, the training
  curve gives a certified bound, and because samples are drawn i.i.d. from
  q_theta there is *no Markov chain and no critical slowing down*.

The gradient uses the REINFORCE / score-function estimator with a batch-mean
baseline (spins are discrete, so we cannot reparameterize):

        grad(beta F_q) = E_q[ (beta E(s) + ln q(s) - b) * grad ln q_theta(s) ].

Author: scaffold for Moh & spouse joint project. MIT-style, extend freely.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  1. Exact references (to validate the network)                              #
# --------------------------------------------------------------------------- #
def onsager_free_energy_per_spin(T: float, J: float = 1.0) -> float:
    """Exact 2D Ising free energy per spin in the thermodynamic limit (Onsager).

    f(T) = -(1/beta) * [ ln2/2 + (1/(2 pi^2)) * I ],
    I = \\int_0^pi \\int_0^pi ln[ cosh^2(2K) - sinh(2K)(cos x + cos y) ] dx dy,
    with K = beta J.  Computed by 2D numerical quadrature.
    """
    beta = 1.0 / T
    K = beta * J
    c = math.cosh(2 * K) ** 2
    s = math.sinh(2 * K)
    n = 400
    grid = (np.arange(n) + 0.5) * math.pi / n           # midpoints in (0, pi)
    cx = np.cos(grid)
    integrand = np.log(c - s * (cx[:, None] + cx[None, :]))
    I = integrand.mean() * math.pi * math.pi            # <f> * area(pi^2)
    lnZ_per_spin = math.log(2) + I / (2 * math.pi * math.pi)
    return -lnZ_per_spin / beta


def exact_enumeration(L: int, T: float, J: float = 1.0):
    """Brute-force exact free energy / energy / |m| for a small L x L torus.

    Only feasible for L <= 4 (2^16 states).  Used as a ground-truth unit test:
    the variational free energy must satisfy  F_q >= F_exact.
    """
    N = L * L
    assert N <= 20, "exact enumeration only for tiny lattices (2^N states)"
    beta = 1.0 / T
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)   # (2^N, N)
    spins = (2 * bits - 1).reshape(-1, L, L).astype(np.float64)
    E = -J * (
        (spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2))
        + (spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2))
    )
    w = np.exp(-beta * (E - E.min()))
    Z = w.sum()
    F = -T * (np.log(Z) - beta * E.min())
    p = w / Z
    E_mean = (p * E).sum()
    m = np.abs(spins.sum(axis=(1, 2))) / N
    m_mean = (p * m).sum()
    return dict(F_per_spin=F / N, E_per_spin=E_mean / N, m_abs=m_mean)


# --------------------------------------------------------------------------- #
#  2. Ising energy (vectorized, periodic boundary conditions)                 #
# --------------------------------------------------------------------------- #
def ising_energy(spins: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    """Energy of a batch of configs. spins: (B, L, L) with values in {-1,+1}.

    E = -J * sum over nearest-neighbour bonds (each bond counted once) via a
    shift in +x and +y with periodic wrap. Returns (B,).
    """
    right = spins * torch.roll(spins, shifts=-1, dims=2)
    down = spins * torch.roll(spins, shifts=-1, dims=1)
    return -J * (right.sum(dim=(1, 2)) + down.sum(dim=(1, 2)))


# --------------------------------------------------------------------------- #
#  3. MADE: a masked autoregressive network over the N lattice bits           #
# --------------------------------------------------------------------------- #
class MaskedLinear(nn.Linear):
    """Linear layer with a fixed binary mask on the weight matrix."""

    def __init__(self, in_f, out_f, bias=True):
        super().__init__(in_f, out_f, bias)
        self.register_buffer("mask", torch.ones(out_f, in_f))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.as_tensor(mask, dtype=self.mask.dtype))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation (Germain et al. 2015).

    Autoregressive over N sites in raster order.  Output k is a logit for
    P(x_k = 1 | x_{<k}); masks guarantee output k sees inputs 1..k-1 only.
    """

    def __init__(self, N: int, hidden=(256, 256)):
        super().__init__()
        self.N = N
        sizes = [N] + list(hidden) + [N]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [MaskedLinear(a, b), nn.ReLU()]
        layers.pop()                                    # drop last ReLU
        self.net = nn.Sequential(*layers)

        # --- degree assignment (natural ordering of the N inputs/outputs) ---
        deg = [np.arange(1, N + 1)]                     # input degrees 1..N
        for h in hidden:
            deg.append(np.arange(h) % (N - 1) + 1)      # hidden degrees 1..N-1
        deg.append(np.arange(1, N + 1))                 # output degrees 1..N

        masked = [m for m in self.net if isinstance(m, MaskedLinear)]
        for li, layer in enumerate(masked):
            d_in, d_out = deg[li], deg[li + 1]
            if li < len(masked) - 1:                    # hidden: m_out >= m_in
                mask = (d_out[:, None] >= d_in[None, :]).astype(np.float32)
            else:                                       # output: strict >
                mask = (d_out[:, None] > d_in[None, :]).astype(np.float32)
            layer.set_mask(mask)

    def logits(self, x):
        return self.net(x)

    def log_prob(self, x):
        """log q(x) for a batch of bit configs x in {0,1}, shape (B, N)."""
        logits = self.logits(x)
        # log P(x_k) = -BCEwithlogits(logit_k, x_k), summed over k
        return -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)

    @torch.no_grad()
    def sample(self, n_samples: int, device="cpu"):
        """Draw n i.i.d. configs by N sequential Bernoulli steps. Returns bits."""
        x = torch.zeros(n_samples, self.N, device=device)
        for k in range(self.N):
            logit_k = self.logits(x)[:, k]
            x[:, k] = torch.bernoulli(torch.sigmoid(logit_k))
        return x


# --------------------------------------------------------------------------- #
#  4. VAN wrapper (bits <-> spins, energy, free energy)                       #
# --------------------------------------------------------------------------- #
class VAN(nn.Module):
    def __init__(self, L: int, hidden=(256, 256), J: float = 1.0):
        super().__init__()
        self.L, self.N, self.J = L, L * L, J
        self.made = MADE(self.N, hidden)

    def bits_to_spins(self, x):
        return (2 * x - 1).view(-1, self.L, self.L)

    def sample(self, n, device="cpu"):
        return self.made.sample(n, device)

    def free_energy_terms(self, n, beta, device="cpu"):
        """Sample and return (beta*E + ln q) per config and the differentiable
        log q for the REINFORCE surrogate."""
        x = self.sample(n, device)                      # no grad through sampling
        log_q = self.made.log_prob(x)                   # differentiable
        with torch.no_grad():
            E = ising_energy(self.bits_to_spins(x), self.J)
            f = beta * E + log_q.detach()               # beta*F contribution
        return f, log_q, E, x


# --------------------------------------------------------------------------- #
#  5. Training (REINFORCE + baseline), one temperature                        #
# --------------------------------------------------------------------------- #
def train_one_temperature(van, T, steps, batch, lr=1e-3, device="cpu",
                          beta_anneal=False, verbose=False):
    """Minimize variational free energy at temperature T. Returns history dict."""
    beta = 1.0 / T
    opt = torch.optim.Adam(van.parameters(), lr=lr)
    hist = {"betaF_per_spin": [], "F_per_spin": []}
    for step in range(steps):
        # optional inverse-temperature warm-up stabilizes early training
        b = beta * min(1.0, (step + 1) / (0.2 * steps)) if beta_anneal else beta
        f, log_q, E, x = van.free_energy_terms(batch, b, device)
        baseline = f.mean()
        # surrogate whose gradient equals the REINFORCE estimator of d(betaF)
        loss = ((f - baseline).detach() * log_q).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(van.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            betaF = (beta * E + log_q).mean().item()
            hist["betaF_per_spin"].append(betaF / van.N)
            hist["F_per_spin"].append(betaF / beta / van.N)
        if verbose and (step % max(1, steps // 5) == 0 or step == steps - 1):
            print(f"    T={T:.3f} step {step:4d}  F/N={betaF/beta/van.N:+.5f}")
    return hist


@torch.no_grad()
def measure(van, T, n=20000, device="cpu"):
    """Estimate observables from i.i.d. samples (no autocorrelation)."""
    beta = 1.0 / T
    x = van.sample(n, device)
    spins = van.bits_to_spins(x)
    E = ising_energy(spins, van.J)
    log_q = van.made.log_prob(x)
    N = van.N
    betaF = (beta * E + log_q).mean().item()
    return dict(
        T=T,
        F_per_spin=betaF / beta / N,
        E_per_spin=E.mean().item() / N,
        C_per_spin=(beta ** 2) * E.var(unbiased=True).item() / N,   # C = beta^2 Var(E) / N
        m_abs=spins.sum(dim=(1, 2)).abs().mean().item() / N,
        S_per_spin=(E.mean().item() / N - betaF / beta / N) / T,    # S = (E - F)/T
    )
