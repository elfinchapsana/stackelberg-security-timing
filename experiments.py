"""Experiments E1-E5 and solver validation. Each function writes CSV files to
results/ and returns a pandas DataFrame. Run everything with run_all.py."""
import os, time, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool

import solver as S

RES = "results"
NAMES = ["SCADA", "PLC", "Servo", "Switch", "HMI"]


# ----------------------------------------------------------------------------
# instances
# ----------------------------------------------------------------------------
def e1_instance(B=10.0, m=1.0, slack_scale=1.0, cA_scale=1.0):
    D = np.array([80, 120, 60, 70, 40.0]); G = np.array([60, 90, 40, 70, 50.0])
    cA = np.array([5, 8, 4, 3, 2.0]) * cA_scale; p = np.array([0.5, 0.6, 0.4, 0.7, 0.8])
    cH = np.array([1.0, 1.5, 1.0, 0.8, 0.5]); a = np.array([0.5, 1.0, 0.8, 0.6, 0.3])
    tau0 = np.array([4.0, 2.0, 1.5, 1.0, 3.0])
    A = np.array([[0, 1, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=float)   # motion, supervisory
    s = np.array([3.5, 6.0]) * slack_scale
    return dict(D=D, G=G, cA=cA, p=p, cH=cH, a=a, tau0=tau0, A=A, s=s, B=float(B), m=float(m), C=np.full(5, 10.0))


def e4_instance(u, crypto="software", T2=10_000.0, B=10.0, m=1.0):
    """Timing-anchored five-asset scenario (delays in microseconds per protected frame)."""
    inst = e1_instance(B=B, m=m)
    # software: PLC 10 us (software auth.), servo 100 us (MCU); hardware: PLC and servo use the
    # 2.6 us AES-GCM engine. Switch: MACsec 0.21 us. SCADA/HMI: 10 us (assumption).
    plc, servo = (10.0, 100.0) if crypto == "software" else (2.6, 2.6)
    inst["a"] = np.array([10.0, plc, servo, 0.21, 10.0])             # SCADA, PLC, Servo, Switch, HMI
    T = np.array([250.0, T2])
    inst["s"] = (1.0 - u) * T
    return inst


def base_draws(seed, N=8, R=3):
    rng = np.random.default_rng(seed)
    return dict(D=rng.uniform(20, 150, N), g=rng.uniform(0.5, 1.5, N), p=rng.uniform(0.2, 0.9, N),
                c=rng.uniform(0, 0.2, N), cH=rng.uniform(0.5, 2, N), tau0=rng.uniform(0.5, 3, N),
                ua=rng.uniform(-1, 1, N), ul=rng.uniform(0, 1, N), memb=rng.random((R, N)))


def random_instance(seed, sigma, beta, rho, m, eta, delays="synthetic", N=8, R=3):
    d = base_draws(seed, N, R)
    G = d["D"] * d["g"]
    if delays == "synthetic":
        a = 0.8 * (1 + eta * d["ua"])
    else:                                                               # E4: log-uniform over [0.2, 100] us
        a = np.exp(np.log(0.2) + d["ul"] * (np.log(100.0) - np.log(0.2)))
    A = (d["memb"] < rho).astype(float)
    # every loop must contain at least two assets: redraw memberships until it does
    t = 0
    while A.sum(1).min() < 2:
        t += 1
        A = (np.random.default_rng([seed, int(1000 * rho), t]).random(A.shape) < rho).astype(float)
    s = sigma * (A @ a)
    return dict(D=d["D"], G=G, cA=G * d["p"] * d["c"], p=d["p"], cH=d["cH"], a=a, tau0=d["tau0"],
                A=A, s=s, B=beta * d["cH"].sum(), m=float(m), C=np.full(N, 10.0))


# ----------------------------------------------------------------------------
# E1
# ----------------------------------------------------------------------------
def run_e1():
    inst = e1_instance()
    rows, alloc = [], {}
    for mo in ["A", "B", "Bp", "none"]:
        cons = S.model(inst, mo)
        l, h, j = S.solve(inst, cons)
        use = S.loop_use(h, inst) / inst["s"]
        u = S.U(h, inst)
        rows.append(dict(model=mo, loss=l, target=NAMES[j] if j >= 0 else "abstain",
                         budget_used=h @ inst["cH"] / inst["B"], loop1_use=use[0], loop2_use=use[1],
                         common_attacker_utility=u[j] if j >= 0 else 0.0))
        alloc[mo] = h
    df = pd.DataFrame(rows)
    df["rel_to_A"] = df["loss"] / df.loc[df.model == "A", "loss"].item() - 1
    pd.DataFrame(alloc, index=NAMES).to_csv(f"{RES}/E1_allocations.csv")
    df.to_csv(f"{RES}/E1_losses.csv", index=False)
    return df, pd.DataFrame(alloc, index=NAMES)


# ----------------------------------------------------------------------------
# E2
# ----------------------------------------------------------------------------
def _e2_point(kw, models=("A", "B", "Bp", "none"), eps=0.0):
    inst = e1_instance(**kw)
    out = dict(kw)
    for mo in models:
        l, h, j = S.solve(inst, S.model(inst, mo), eps=eps * S.scale(inst))
        out[f"L_{mo}"] = l; out[f"tgt_{mo}"] = j
    out["relPoD"] = S.rel(out["L_B"], out["L_A"])
    return out


def run_e2():
    sig = np.geomspace(0.05, 8, 45)
    slack = pd.DataFrame([_e2_point(dict(m=m, slack_scale=s)) for m in [0.5, 1, 2, 3] for s in sig])
    budget = pd.DataFrame([_e2_point(dict(B=b), models=("A", "B")) for b in range(1, 41)])
    grid = pd.DataFrame([_e2_point(dict(B=b, slack_scale=s), models=("A", "B"))
                         for b in np.geomspace(1, 40, 25) for s in np.geomspace(0.05, 8, 23)])
    cost = pd.DataFrame([_e2_point(dict(cA_scale=c), models=("A", "B")) for c in [0, 0.5, 1, 2, 4]])
    epsr = pd.DataFrame([dict(eps=e, **_e2_point(dict(), models=("A", "B"), eps=e)) for e in [0.001, 0.01, 0.05]])
    for n, d in [("slack", slack), ("budget", budget), ("grid", grid), ("attack_cost", cost), ("eps", epsr)]:
        d.to_csv(f"{RES}/E2_{n}.csv", index=False)
    n_solves = 4 * len(slack) + 2 * (len(budget) + len(grid) + len(cost) + len(epsr))
    return dict(slack=slack, budget=budget, grid=grid, cost=cost, eps=epsr, n_solves=n_solves)


# ----------------------------------------------------------------------------
# E3 / E4 random sweeps
# ----------------------------------------------------------------------------
def _sweep_instance(args):
    seed, sigma, beta, rho, m, eta, delays = args
    inst = random_instance(seed, sigma, beta, rho, m, eta, delays)
    sc = S.scale(inst)
    out = dict(seed=seed, sigma=sigma, beta=beta, rho=rho, m=m, eta=eta)
    sol = {}
    for mo in ["A", "B", "Bp", "none", "delay", "risk"]:
        sol[mo] = S.solve(inst, S.model(inst, mo))
    sol["oracle"] = S.solve(inst, S.model(inst, "oracle", sol["A"][1]))
    for mo, (l, h, j) in sol.items():
        out[f"L_{mo}"] = l; out[f"tgt_{mo}"] = j
    lA, hA, jA = sol["A"]
    out["A_budget_binds"], out["A_timing_binds"] = S.binding(hA, inst, S.model(inst, "A"))
    out["B_budget_binds"], _ = S.binding(sol["B"][1], inst, S.model(inst, "B"))
    # re-evaluation check
    out["reeval_err_A"] = abs(S.attacker_eval(hA, inst)[0] - lA)
    out["feas_A"] = bool(S.feasible(hA[None, :], inst, S.model(inst, "A"), tol=S.FEAS_TOL)[0])
    # robustness
    e1 = 0.01 * sc
    out["near_tie"] = S.near_tie(hA, inst, e1)
    out["robust_of_sse_A"] = S.attacker_eval(hA, inst, eps=e1)[0]
    for ef in [0.01, 0.05]:
        e = ef * sc
        lAe, hAe, jAe = S.solve(inst, S.model(inst, "A"), eps=e)
        out[f"L_A_e{ef}"] = lAe; out[f"tgt_A_e{ef}"] = jAe
        out[f"robust_A_e{ef}"] = S.attacker_eval(hAe, inst, eps=e)[0] if np.all(np.isfinite(hAe)) else np.inf
        for mo in ["B", "delay", "risk"]:
            l, h, j = S.solve(inst, S.model(inst, mo), eps=e)
            out[f"L_{mo}_e{ef}"] = l; out[f"tgt_{mo}_e{ef}"] = j
        out[f"L_oracle_e{ef}"] = S.solve(inst, S.model(inst, "oracle", hAe), eps=e)[0]
    # minimax allocation against the strategic attacker
    _, hmm = S.minimax(inst, S.model(inst, "A"))
    out["L_minimax_vs_strategic"] = S.attacker_eval(hmm, inst)[0]
    return out


def run_sweep(delays="synthetic", workers=None, seeds=40):
    etas = [0.0, 0.8] if delays == "synthetic" else [0.0]
    grid = list(itertools.product(range(seeds), [0.25, 0.5, 1, 2, 4], [1, 2, 4], [0.3, 0.6, 0.9], [0.5, 1, 2], etas))
    args = [g + (delays,) for g in grid]
    t0 = time.time()
    with Pool(workers or os.cpu_count()) as pool:
        rows = pool.map(_sweep_instance, args, chunksize=20)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/{'E3' if delays == 'synthetic' else 'E4'}_random_sweep.csv", index=False)
    print(f"  {len(df)} instances in {time.time() - t0:.0f}s")
    return df


# ----------------------------------------------------------------------------
# E4 anchored scenario
# ----------------------------------------------------------------------------
def run_e4_anchored():
    rows = []
    for crypto in ["software", "hardware"]:
        for T2 in [10_000.0, 1_000.0]:
            for u in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
                inst = e4_instance(u, crypto, T2)
                r = dict(crypto=crypto, T2_us=T2, u=u)
                sol = {mo: S.solve(inst, S.model(inst, mo)) for mo in ["A", "B", "delay", "risk"]}
                sol["oracle"] = S.solve(inst, S.model(inst, "oracle", sol["A"][1]))
                for mo, (l, h, j) in sol.items():
                    r[f"L_{mo}"] = l
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/E4_anchored.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def _slsqp_best(inst, cons, starts=10, rng=None):
    """Multi-start SLSQP over candidate programs: min L_j s.t. U_i <= U_j, constraints."""
    from scipy.optimize import minimize
    rng = rng or np.random.default_rng(0)
    N = inst["G"].size
    ub = inst["C"] if cons[0] == "loops" else np.minimum(cons[1], 1e3)
    best = np.inf
    lin = []
    lin.append(dict(type="ineq", fun=lambda h: inst["B"] - h @ inst["cH"], jac=lambda h: -inst["cH"]))
    if cons[0] == "loops":
        M = inst["A"] * inst["a"]
        lin.append(dict(type="ineq", fun=lambda h: inst["s"] - M @ h, jac=lambda h: -M))
    for j in range(N):
        cj = lin + [dict(type="ineq", fun=lambda h, j=j: S.U(h, inst)[j] - S.U(h, inst))]
        for _ in range(starts):
            x0 = rng.uniform(0, 1, N) * np.minimum(ub, 3.0)
            try:
                r = minimize(lambda h, j=j: S.L(h, inst)[j], x0, method="SLSQP",
                             bounds=[(0, b) for b in ub], constraints=cj, options=dict(maxiter=300, ftol=1e-12))
            except Exception:
                continue
            h = np.clip(r.x, 0, ub)
            if S.feasible(h[None, :], inst, cons)[0]:                 # strictly feasible, as in Algorithm 1
                best = min(best, S.attacker_eval(h, inst)[0])         # same attacker model and tie tolerance
    return best


def _random_best(inst, cons, n=20000, rng=None):
    rng = rng or np.random.default_rng(1)
    N = inst["G"].size
    ub = inst["C"] if cons[0] == "loops" else np.minimum(cons[1], inst["C"])
    H = rng.random((n, N)) ** 2 * ub
    # scale each draw down to the feasibility boundary
    spend = H @ inst["cH"]
    f = np.minimum(1.0, inst["B"] / np.maximum(spend, 1e-12))
    if cons[0] == "loops":
        use = (H * inst["a"]) @ inst["A"].T
        f = np.minimum(f, np.min(np.where(use > 0, inst["s"] / np.maximum(use, 1e-12), np.inf), axis=1))
    H = H * f[:, None] * (1 - 1e-12)
    return min(S.attacker_eval(h, inst)[0] for h in H)


def run_validation():
    rows = []
    # (i) zero-sum special case: G = D, cA = 0 -> SSE = minimax
    for seed in range(51):
        inst = random_instance(10_000 + seed, sigma=[0.25, 0.5, 1, 2][seed % 4], beta=[1, 2, 4][seed % 3], rho=0.6, m=1.0, eta=0.8)
        inst["G"] = inst["D"].copy(); inst["cA"] = np.zeros_like(inst["D"])
        for mo in ["A", "B", "Bp", "none"]:
            cons = S.model(inst, mo)
            l = S.solve(inst, cons)[0]; mm = S.minimax(inst, cons)[0]
            rows.append(dict(check="zero_sum", model=mo, sse=l, other=mm, rel_diff=abs(l - mm) / max(mm, 1e-300)))
    # (ii) SLSQP multistart and 20,000 random feasible allocations
    cases = [(e1_instance(), mo) for mo in ["A", "B", "Bp", "none"]]
    for seed in range(29):
        inst = random_instance(20_000 + seed, sigma=[0.25, 0.5, 1, 2][seed % 4], beta=[1, 2, 4][seed % 3],
                               rho=[0.3, 0.6, 0.9][seed % 3], m=[0.5, 1, 2][seed % 3], eta=0.8)
        cases += [(inst, "A"), (inst, "B")]
    for k, (inst, mo) in enumerate(cases):
        cons = S.model(inst, mo)
        l, h, j = S.solve(inst, cons)
        sl = _slsqp_best(inst, cons, rng=np.random.default_rng(k))
        rb = _random_best(inst, cons, rng=np.random.default_rng(10_000 + k))
        rows.append(dict(check="slsqp", model=mo, sse=l, other=sl, diff=sl - l))
        rows.append(dict(check="random", model=mo, sse=l, other=rb, diff=rb - l))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/validation.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# E5 scalability
# ----------------------------------------------------------------------------
def scal_instance(N, R, seed, rho=0.3, loop_size=None, sigma=1.0, beta=2.0, m=1.0, eta=0.8):
    rng = np.random.default_rng(seed)
    D = rng.uniform(20, 150, N); G = D * rng.uniform(0.5, 1.5, N)
    p = rng.uniform(0.2, 0.9, N); cA = G * p * rng.uniform(0, 0.2, N)
    cH = rng.uniform(0.5, 2, N); a = 0.8 * (1 + eta * rng.uniform(-1, 1, N))
    if loop_size is None:
        A = (rng.random((R, N)) < rho).astype(float)
    else:
        A = np.zeros((R, N))
        for r in range(R):
            A[r, rng.choice(N, size=min(loop_size, N), replace=False)] = 1.0
    return dict(D=D, G=G, cA=cA, p=p, cH=cH, a=a, A=A, s=sigma * (A @ a), B=beta * cH.sum(), m=m, C=np.full(N, 10.0))


def run_e5():
    rows = []
    for regime in ["dense", "sparse"]:
        for N in [10, 25, 50, 100, 250, 500, 1000]:
            for R in [5, 10, 20, 50]:
                if regime == "sparse" and R != 50:
                    continue
                for sd in range(5):
                    inst = scal_instance(N, R, 1000 + sd, loop_size=(10 if regime == "sparse" else None))
                    rec = dict(regime=regime, N=N, R=R, seed=sd, Vmax=int(inst["A"].sum(1).max()))
                    for mo in ["A", "B"]:
                        cons = S.model(inst, mo)
                        t0 = time.perf_counter()
                        l, h, j = S.solve(inst, cons)
                        rec[f"wall_{mo}"] = time.perf_counter() - t0
                        rec[f"L_{mo}"] = l; rec[f"tgt_{mo}"] = j
                        rec[f"feas_{mo}"] = bool(S.feasible(h[None, :], inst, cons, tol=S.FEAS_TOL)[0])
                    rows.append(rec)
                    print(f"  E5 {regime:6s} N={N:5d} R={R:3d} seed={sd}  A {rec['wall_A']:.3f}s  B {rec['wall_B']:.3f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/E5_scalability.csv", index=False)
    return df
