"""Core solver for 'Stackelberg security resource allocation under shared
real-time constraints'. Implements Algorithm 1 (SSE by candidate bisection)
for every timing model used in the paper, the epsilon-margin variant, the
minimax benchmark, attacker best-response evaluation and the per-asset cap
rules (equal, delay-proportional, risk-weighted, oracle).

Notation follows the paper: D (damage), G (attacker gain), cA (attack cost),
p (baseline breach prob.), cH (hardening cost), a (delay per unit hardening),
A (R x N loop-asset incidence), s (loop slack), B (budget), m (breach shape),
C (hardening caps).
"""
import numpy as np

INF = np.inf
K_DEFAULT = 100
FEAS_TOL = 1e-9       # absolute tolerance used when re-checking feasibility
TIE_REL = 1e-9        # utilities within TIE_REL * max(G p) are treated as tied


# ----------------------------------------------------------------------------
# basic functions
# ----------------------------------------------------------------------------
def P(h, p, m):
    return p / (1.0 + np.power(h, m))


def U(h, inst):
    return inst["G"] * P(h, inst["p"], inst["m"]) - inst["cA"]


def L(h, inst):
    return inst["D"] * P(h, inst["p"], inst["m"])


def hhat(nu, G, p, cA, m):
    """Lemma 1: smallest h >= 0 with G P(h) - cA <= nu (broadcasts)."""
    qbar = (nu + cA) / G
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(qbar <= 0, INF,
                     np.power(np.maximum(p / np.where(qbar > 0, qbar, 1.0) - 1.0, 0.0), 1.0 / m))
    return np.where(G * p - cA <= nu, 0.0, h)


def scale(inst):
    return float(np.max(inst["G"] * inst["p"]))


# ----------------------------------------------------------------------------
# constraint sets
# ----------------------------------------------------------------------------
def caps_from_delta(inst, delta):
    a = inst["a"]
    with np.errstate(divide="ignore", invalid="ignore"):
        hc = np.where(a > 0, delta / np.where(a > 0, a, 1.0), INF)
    return np.minimum(inst["C"], hc)


def split_caps(inst, weights):
    """delta_i = min_{r: i in V_r} s_r w_i / sum_{k in V_r} w_k (equal split if all w = 0)."""
    A, s = inst["A"], inst["s"]
    N = A.shape[1]
    delta = np.full(N, INF)
    for r in range(A.shape[0]):
        idx = np.flatnonzero(A[r])
        if idx.size == 0:
            continue
        w = weights[idx].astype(float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        delta[idx] = np.minimum(delta[idx], s[r] * w / w.sum())
    return caps_from_delta(inst, delta)


def relaxed_caps(inst):
    """Model B': each asset gets the whole slack of its tightest loop."""
    A, s = inst["A"], inst["s"]
    N = A.shape[1]
    delta = np.full(N, INF)
    for r in range(A.shape[0]):
        idx = np.flatnonzero(A[r])
        delta[idx] = np.minimum(delta[idx], s[r])
    return caps_from_delta(inst, delta)


def oracle_caps(inst, hA):
    A, s, a = inst["A"], inst["s"], inst["a"]
    N = A.shape[1]
    delta = np.full(N, INF)
    use = a * hA
    for r in range(A.shape[0]):
        idx = np.flatnonzero(A[r])
        if idx.size == 0:
            continue
        resid = max(s[r] - use[idx].sum(), 0.0)
        delta[idx] = np.minimum(delta[idx], use[idx] + resid / idx.size)
    return caps_from_delta(inst, delta)


def risk_weights(inst):
    return inst["a"] * np.maximum(U(np.zeros_like(inst["a"]), inst), 0.0)


def model(inst, name, hA=None):
    """Return a constraint specification for a named model."""
    if name == "A":
        return ("loops", None)
    if name == "none":
        return ("caps", inst["C"].copy())
    if name == "B":
        return ("caps", split_caps(inst, np.ones_like(inst["a"])))
    if name == "Bp":
        return ("caps", relaxed_caps(inst))
    if name == "delay":
        return ("caps", split_caps(inst, inst["a"]))
    if name == "risk":
        return ("caps", split_caps(inst, risk_weights(inst)))
    if name == "oracle":
        return ("caps", oracle_caps(inst, hA))
    raise ValueError(name)


def feasible(H, inst, cons, tol=0.0):
    """H: (k, N) candidate allocations -> bool (k,)."""
    kind, caps = cons
    fin = np.all(np.isfinite(H), axis=1)
    Hs = np.where(np.isfinite(H), H, 0.0)
    ok = fin & (Hs @ inst["cH"] <= inst["B"] + tol)
    if kind == "loops":
        ok &= np.all(Hs <= inst["C"] + tol, axis=1)
        ok &= np.all((Hs * inst["a"]) @ inst["A"].T <= inst["s"] + tol, axis=1)
    else:
        ok &= np.all(Hs <= caps + tol, axis=1)
    return ok


# ----------------------------------------------------------------------------
# Algorithm 1
# ----------------------------------------------------------------------------
def solve(inst, cons, eps=0.0, K=K_DEFAULT):
    """Strong Stackelberg equilibrium (eps = 0) or eps-margin value (eps > 0).
    Returns (loss, allocation h, target index or -1 if the attacker abstains)."""
    G, p, cA, m = inst["G"], inst["p"], inst["cA"], inst["m"]
    N = G.size
    if eps > 0:                                        # enforce the margin strictly (paper, Sec. 4.3)
        eps = eps * (1 + 1e-6)
    h0 = hhat(-eps, G, p, cA, m)                       # deterrence: all U_i <= -eps
    if feasible(h0[None, :], inst, cons)[0]:
        return 0.0, h0, -1
    ub = inst["C"].copy() if cons[0] == "loops" else cons[1].copy()
    lo, hi = np.zeros(N), np.minimum(ub, 1e6)
    eye = np.eye(N, dtype=bool)

    def build(x):
        u = G * P(x, p, m) - cA                        # attacker utility of candidate j at h_j = x_j
        Hm = hhat(u[:, None] - eps, G[None, :], p[None, :], cA[None, :], m)
        Hm[eye] = x
        return Hm

    ok0 = feasible(build(lo), inst, cons)
    for _ in range(K):
        x = 0.5 * (lo + hi)
        f = feasible(build(x), inst, cons)
        lo = np.where(f, x, lo)
        hi = np.where(f, hi, x)
    loss = np.where(ok0, L(lo, inst), INF)
    j = int(np.argmin(loss))
    if not np.isfinite(loss[j]):
        return INF, np.full(N, np.nan), -2
    return float(loss[j]), build(lo)[j], j


def minimax(inst, cons, K=K_DEFAULT):
    """min_h max_i L_i(h_i) by bisection on the epigraph level theta."""
    D, p, m = inst["D"], inst["p"], inst["m"]
    zero = np.zeros_like(D)
    lo, hi = 0.0, float(np.max(D * p))
    for _ in range(K):
        th = 0.5 * (lo + hi)
        h = hhat(th, D, p, zero, m)
        if feasible(h[None, :], inst, cons)[0]:
            hi = th
        else:
            lo = th
    return hi, hhat(hi, D, p, zero, m)


# ----------------------------------------------------------------------------
# evaluation of a given allocation against the attacker
# ----------------------------------------------------------------------------
def attacker_eval(h, inst, eps=None):
    """Nominal loss with defender-favourable tie-breaking (eps=None), or the
    eps-robust loss (worst defender loss over options within eps of the best)."""
    u = U(h, inst)
    l = L(h, inst)
    opts_u = np.append(u, 0.0)            # last option = abstain
    opts_l = np.append(l, 0.0)
    best = opts_u.max()
    if eps is None:
        S = opts_u >= best - TIE_REL * scale(inst)
        k = np.flatnonzero(S)[np.argmin(opts_l[S])]
    else:
        S = opts_u >= best - eps
        k = np.flatnonzero(S)[np.argmax(opts_l[S])]
    return float(opts_l[k]), (int(k) if k < h.size else -1)


def near_tie(h, inst, eps):
    """True if some other option is within eps of the attacker's best option."""
    u = np.append(U(h, inst), 0.0)
    srt = np.sort(u)[::-1]
    return bool(srt[0] - srt[1] <= eps)


def loop_use(h, inst):
    return (h * inst["a"]) @ inst["A"].T


def binding(h, inst, cons):
    spend = float(h @ inst["cH"])
    budget = spend >= inst["B"] * (1 - 1e-7) - 1e-9
    if cons[0] == "loops":
        timing = bool(np.any(loop_use(h, inst) >= inst["s"] * (1 - 1e-7) - 1e-9))
    else:
        timing = bool(np.any(h >= cons[1] * (1 - 1e-7) - 1e-12) & np.any(np.isfinite(cons[1])))
    return budget, timing


def rel(lb, la):
    if la <= 0:
        return 0.0 if lb <= 0 else INF
    return (lb - la) / la
