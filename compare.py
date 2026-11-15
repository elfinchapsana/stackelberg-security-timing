"""Compare every number quoted in the paper with the reproduced results.
Writes results/comparison.csv and prints a table."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = "results"


def rel(d, b, a):
    lb, la = d[b].to_numpy(float), d[a].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(la > 0, (lb - la) / la, np.where(lb > 1e-12, np.inf, 0.0))


def pct(x):
    return 100 * float(x)


def build():
    rows = []
    add = lambda sec, item, paper, repro, unit="%": rows.append(dict(section=sec, item=item, paper=paper, reproduced=repro, unit=unit))

    e1 = pd.read_csv(f"{RES}/E1_losses.csv").set_index("model")
    add("E1 Table 5", "Loss Model A", 10.17, e1.loc["A", "loss"], "")
    add("E1 Table 5", "Loss Model B", 15.14, e1.loc["B", "loss"], "")
    add("E1 Table 5", "Loss Model B' / no timing", 7.64, e1.loc["Bp", "loss"], "")
    add("E1 Table 5", "B above A", 48.9, pct(e1.loc["B", "rel_to_A"]))
    add("E1 Table 5", "Budget used A / B", "70 / 38", f"{pct(e1.loc['A','budget_used']):.0f} / {pct(e1.loc['B','budget_used']):.0f}")
    add("E1 text", "Common attacker utility A / B", "10.71 / 16.92",
        f"{e1.loc['A','common_attacker_utility']:.2f} / {e1.loc['B','common_attacker_utility']:.2f}", "")
    add("E1 text", "B' violates motion loop by", 42, pct(e1.loc["Bp", "loop1_use"] - 1))

    sl = pd.read_csv(f"{RES}/E2_slack.csv")
    for m, paper in [(1.0, 50), (2.0, 113), (3.0, 237)]:
        d = sl[sl.m == m]; r = d.relPoD.replace(np.inf, np.nan)
        add("E2 text", f"Peak relative PoD, m={m:g}", paper, pct(r.max()))
    d = sl[sl.m == 3.0]; k = (d.slack_scale - 1.3).abs().idxmin()
    add("E2 text", "m=3, sigma~1.3: loss A / B", "1.63 / 5.48", f"{d.L_A[k]:.2f} / {d.L_B[k]:.2f}", "")
    bu = pd.read_csv(f"{RES}/E2_budget.csv").set_index("B")
    add("E2 text", "Budget gap at B=5 / 6 / 7+", "18 / 34 / 48.9",
        f"{pct(bu.relPoD[5]):.0f} / {pct(bu.relPoD[6]):.0f} / {pct(bu.relPoD[7]):.1f}")

    e3 = pd.read_csv(f"{RES}/E3_random_sweep.csv")
    pod = rel(e3, "L_B", "L_A")
    add("E3 text", "Instances", 10800, len(e3), "")
    add("E3 text", "Share L_B > L_A", 90.6, pct(np.mean(e3.L_B > e3.L_A + 1e-9)))
    add("E3 text", "Share PoD > 5%", 89.1, pct(np.mean(pod > 0.05)))
    add("E3 text / abstract", "Median relative PoD", 50.9, pct(np.median(pod)))
    add("E3 text", "A deters, B does not", 0.9, pct(np.mean((e3.L_A <= 0) & (e3.L_B > 0))))
    c = e3[(e3.sigma == 4) & (e3.beta == 1)]
    add("E3 text", "sigma=4, beta=1: median PoD", 0, pct(np.median(rel(c, "L_B", "L_A"))))
    add("E3 text", "sigma=4, beta=1: budget binds", 99.7, pct(c.A_budget_binds.mean()))
    cells = e3[e3.sigma <= 1].groupby(["sigma", "beta", "rho", "m", "eta"]).A_timing_binds.mean()
    add("E3 text", "sigma<=1: A timing binds per cell", "77-100", f"{pct(cells.min()):.0f}-{pct(cells.max()):.0f}")
    tb = e3.A_timing_binds.astype(bool)
    add("E3 text", "Timing binds => PoD > 0", 100, pct(np.mean(pod[tb] > 1e-9)))
    tdiff = [np.mean(e3.tgt_A != e3.tgt_B), np.mean(e3["tgt_A_e0.01"] != e3["tgt_B_e0.01"]), np.mean(e3["tgt_A_e0.05"] != e3["tgt_B_e0.05"])]
    add("E3 text / abstract", "Target differs A vs B", "29-33", f"{pct(min(tdiff)):.0f}-{pct(max(tdiff)):.0f}")
    mm = e3.L_minimax_vs_strategic / e3.L_A
    add("E3 text", "Minimax vs SSE loss (median ratio)", 1.41, float(np.median(mm[e3.L_A > 0])), "x")
    for f, lv, paper in [("sigma", [0.25, 0.5, 1, 2, 4], [46, 60, 69, 55, 17]), ("beta", [1, 2, 4], [36, 52, 65]),
                         ("rho", [0.3, 0.6, 0.9], [38, 52, 65]), ("m", [0.5, 1, 2], [35, 60, 85]), ("eta", [0.0, 0.8], [49, 54])]:
        rep = [pct(np.median(pod[e3[f] == v])) for v in lv]
        add("E3 Table 6", f"Median PoD by {f} {lv}", " / ".join(map(str, paper)), " / ".join(f"{x:.0f}" for x in rep))
    # robustness
    add("E3 6.3", "Near-tie flag raised", 99.7, pct(e3.near_tie.mean()))
    add("E3 6.3", "eps-robust / nominal, SSE allocation (median)", 2.08, float(np.median((e3.robust_of_sse_A / e3.L_A)[e3.L_A > 0])), "x")
    add("E3 6.3", "Achievable eps-robust / nominal (median)", 1.027, float(np.median((e3["robust_A_e0.01"] / e3.L_A)[e3.L_A > 0])), "x")
    for ef, paper in [("0.01", (49.4, 90.3, 89.2)), ("0.05", (42.0, 90.2, 88.2))]:
        pe = rel(e3, f"L_B_e{ef}", f"L_A_e{ef}")
        add("E3 Table 7", f"eps={ef}: median PoD / share>1% / share>5%", " / ".join(map(str, paper)),
            f"{pct(np.median(pe)):.1f} / {pct(np.mean(pe > 0.01)):.1f} / {pct(np.mean(pe > 0.05)):.1f}")
    add("E3 Table 7", "Nominal: share > 1%", 90.2, pct(np.mean(pod > 0.01)))
    pe05 = rel(e3, "L_B_e0.05", "L_A_e0.05")
    ok = np.isfinite(pod) & np.isfinite(pe05)
    add("E3 6.3", "Spearman nominal vs eps=0.05 PoD", 0.91, spearmanr(pod[ok], pe05[ok]).correlation, "")
    # split rules
    for mo, paper_med, paper_worse in [("delay", 47.7, 16), ("risk", 18.1, 6)]:
        r = rel(e3, f"L_{mo}", "L_A")
        add("E3 Table 8", f"{mo}: median above A / worse than equal", f"{paper_med} / {paper_worse}",
            f"{pct(np.median(r)):.1f} / {pct(np.mean(e3[f'L_{mo}'] > e3.L_B + 1e-9)):.0f}")
    add("E3 Table 8", "Oracle: max |L_oracle - L_A|", "2e-14", float(np.max(np.abs(e3.L_oracle - e3.L_A))), "")
    g = e3[e3.L_B > e3.L_A + 1e-9]
    closes = (g.L_B - g.L_risk) / (g.L_B - g.L_A)
    add("E3 6.4", "Risk rule closes median share of gap", 64, pct(np.median(closes)))
    add("E3 6.4", "Target mismatch vs A: equal / risk", "31 / 15",
        f"{pct(np.mean(e3.tgt_B != e3.tgt_A)):.0f} / {pct(np.mean(e3.tgt_risk != e3.tgt_A)):.0f}")
    # E4
    an = pd.read_csv(f"{RES}/E4_anchored.csv")
    sw = an[(an.crypto == "software") & (an.T2_us == 10_000)].set_index("u")
    add("E4 text", "Software: gap at u=0.9", 79, pct(sw.L_B[0.9] / sw.L_A[0.9] - 1))
    add("E4 text", "Software: median gap over u grid", 34, pct(np.median(sw.L_B / sw.L_A - 1)))
    add("E4 text", "Software: median above A, risk / delay", "54 / 136",
        f"{pct(np.median(sw.L_risk / sw.L_A - 1)):.0f} / {pct(np.median(sw.L_delay / sw.L_A - 1)):.0f}")
    hw = an[an.crypto == "hardware"]
    first = lambda d: d[d.L_B > d.L_A + 1e-9].u.min()
    add("E4 text", "Hardware: gap first appears (10 ms / 1 ms loop)", "0.95 / 0.9",
        f"{first(hw[hw.T2_us == 10_000]):g} / {first(hw[hw.T2_us == 1_000]):g}", "u")
    e4 = pd.read_csv(f"{RES}/E4_random_sweep.csv")
    add("E4 Table 8", "Instances", 5400, len(e4), "")
    add("E4 Table 8", "Equal split median above A (nominal / eps=0.05)", "56.5 / 48.5",
        f"{pct(np.median(rel(e4, 'L_B', 'L_A'))):.1f} / {pct(np.median(rel(e4, 'L_B_e0.05', 'L_A_e0.05'))):.1f}")
    for mo, pm, pw in [("delay", 38.7, 37), ("risk", 13.4, 12)]:
        add("E4 Table 8", f"{mo}: median above A / worse than equal", f"{pm} / {pw}",
            f"{pct(np.median(rel(e4, f'L_{mo}', 'L_A'))):.1f} / {pct(np.mean(e4[f'L_{mo}'] > e4.L_B + 1e-9)):.0f}")
    # validation
    v = pd.read_csv(f"{RES}/validation.csv")
    z = v[v.check == "zero_sum"]
    add("Validation", "Zero-sum: solves / max rel. diff", "204 / 2.6e-16", f"{len(z)} / {z.rel_diff.max():.1e}", "")
    s = v[v.check == "slsqp"]; r = v[v.check == "random"]
    add("Validation", "SLSQP: solves / min (SLSQP - bisection)", "62 / -8.7e-9", f"{len(s)} / {s['diff'].min():.1e}", "")
    add("Validation", "Random 20,000 allocations: min (random - bisection)", ">= 0", f"{r['diff'].min():.2e}", "")
    add("Validation", "All E3 allocations feasible / re-evaluate exactly", "yes",
        f"{bool(e3.feas_A.all())} / max err {e3.reeval_err_A.max():.1e}", "")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv(f"{RES}/comparison.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))
