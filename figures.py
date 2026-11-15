"""Regenerate Figures 1-8 of the paper from the CSV files in results/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

FIG = "figures"
RES = "results"
BLUE, ORANGE, GREEN, YELLOW, PURPLE, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6b5bd6", "#e87ba4"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.7, "axes.axisbelow": True})
LAB = {"A": "A: coupled loop constraints", "B": "B: per-asset caps, equal split (restriction)",
       "Bp": "B': per-asset caps, relaxation (diagnostic)", "none": "No timing constraints (reference)"}


def fig1():
    al = pd.read_csv(f"{RES}/E1_allocations.csv", index_col=0)
    ls = pd.read_csv(f"{RES}/E1_losses.csv").set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(al)); w = 0.26
    for k, (mo, c) in enumerate([("A", BLUE), ("B", ORANGE), ("Bp", GREEN)]):
        ax[0].bar(x + (k - 1) * w, al[mo], w * 0.92, color=c, label=LAB[mo])
        ax[1].bar(np.arange(2) + (k - 1) * w, [ls.loc[mo, "loop1_use"], ls.loc[mo, "loop2_use"]], w * 0.92, color=c)
    ax[0].set_xticks(x, al.index); ax[0].set_ylabel("hardening level $h_i$"); ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("(a) Hardening level by model", loc="left"); ax[0].grid(axis="x", visible=False)
    ax[1].axhline(1.0, color="black", lw=1.2); ax[1].text(1.45, 1.02, "loop deadline", ha="right", va="bottom", fontsize=9)
    ax[1].set_xticks([0, 1], ["Loop 1", "Loop 2"]); ax[1].set_ylabel("share of loop slack used")
    ax[1].set_title("(b) Use of the true loop slack", loc="left"); ax[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/F1_alloc.png", dpi=200); plt.close(fig)


def fig2():
    d = pd.read_csv(f"{RES}/E2_slack.csv")
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.3))
    sty = [("A", BLUE, "-", "o"), ("B", ORANGE, "-", "s"), ("Bp", GREEN, "--", "^"), ("none", YELLOW, ":", "D")]
    for a, m in zip(ax, [0.5, 1.0, 2.0, 3.0]):
        dm = d[d.m == m].sort_values("slack_scale")
        for mo, c, ls, mk in sty:
            a.plot(dm.slack_scale, dm[f"L_{mo}"], ls, color=c, lw=2.2, marker=mk, ms=5, markevery=4, label=LAB[mo])
        a.axvline(1.0, color="#52514e", ls="--", lw=1); a.set_xscale("log"); a.set_title(f"m = {m}", loc="left")
        a.set_xlabel("loop slack scale  (1 = base case)")
    ax[0].set_ylabel("defender loss at Stackelberg eq.")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FIG}/F2_loss_vs_slack.png", dpi=200); plt.close(fig)


def fig3():
    d = pd.read_csv(f"{RES}/E2_grid.csv")
    Bs = np.sort(d.B.unique()); ss = np.sort(d.slack_scale.unique())
    Z = np.full((len(Bs), len(ss)), np.nan)
    for i, b in enumerate(Bs):
        for j, s in enumerate(ss):
            v = d[(d.B == b) & (d.slack_scale == s)].relPoD.item()
            Z[i, j] = 100 * min(v, 10.0)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#d6e6fb", "#2a78d6", "#0d3366"]), vmin=0, vmax=50)
    yt = list(range(0, len(Bs), 3)); ax.set_yticks(yt, [f"{Bs[i]:.3g}" for i in yt])
    xt = [int(np.argmin(abs(ss - v))) for v in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]]
    ax.set_xticks(xt, ["0.05", "0.1", "0.2", "0.4", "0.8", "1.6", "3.2", "6.4"])
    ax.set_xlabel("loop slack scale"); ax.set_ylabel("budget B"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="price of decoupling, % of coupled loss")
    fig.tight_layout(); fig.savefig(f"{FIG}/F3_pod_heatmap.png", dpi=200); plt.close(fig)


def _rel(df, b, a):
    lb, la = df[b].to_numpy(), df[a].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(la > 0, (lb - la) / la, np.where(lb > 0, np.inf, 0.0))
    return r


def fig4():
    d = pd.read_csv(f"{RES}/E3_random_sweep.csv"); d["pod"] = 100 * _rel(d, "L_B", "L_A")
    d = d[d.sigma <= 1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, f, lab in [(ax[0], "rho", r"loop overlap probability $\rho$"), (ax[1], "m", "breach-curve shape $m$")]:
        lv = sorted(d[f].unique())
        data = [np.clip(d[d[f] == v].pod.replace(np.inf, 1e9), None, 200) for v in lv]
        a.boxplot(data, widths=0.5, showfliers=False, whis=1.5, patch_artist=True,
                  boxprops=dict(facecolor="#a3c6f3", edgecolor="#1f5fb0"), medianprops=dict(color="black", lw=2))
        a.set_xticks(range(1, len(lv) + 1), [f"{v:g}" for v in lv]); a.set_xlabel(lab); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("price of decoupling, %")
    fig.tight_layout(); fig.savefig(f"{FIG}/F4_pod_by_rho_m.png", dpi=200); plt.close(fig)


def fig5():
    d = pd.read_csv(f"{RES}/E3_random_sweep.csv"); d["pod"] = 100 * _rel(d, "L_B", "L_A")
    sig = [0.25, 0.5, 1, 2, 4]; bet = [1, 2, 4]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#d6e6fb", "#2a78d6", "#0d3366"])
    Ms = []
    for eta in [0.0, 0.8]:
        M = np.array([[d[(d.eta == eta) & (d.sigma == s) & (d.beta == b)].pod.median() for s in sig] for b in bet])
        Ms.append(M)
    vmax = max(np.nanmax(M[np.isfinite(M)]) for M in Ms)
    for a, eta, M in zip(ax, [0.0, 0.8], Ms):
        im = a.imshow(M, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(3):
            for j in range(5):
                a.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9,
                       color="white" if M[i, j] > 0.55 * vmax else "black")
        a.set_xticks(range(5), [f"{s:g}" for s in sig]); a.set_yticks(range(3), bet); a.grid(False)
        a.set_xlabel(r"timing slack $\sigma$ (uniform hardening level loops absorb)")
        a.set_title(f"delay heterogeneity eta = {eta}", loc="left")
    ax[0].set_ylabel(r"budget $\beta$ (uniform level budget buys)")
    fig.colorbar(im, ax=ax, label="median price of decoupling, %", fraction=0.025)
    fig.savefig(f"{FIG}/F5_pod_grid.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig6():
    d = pd.read_csv(f"{RES}/E3_random_sweep.csv")
    sig = [0.25, 0.5, 1, 2, 4]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for a, suf, title in [(ax[0], "", "nominal SSE loss"), (ax[1], "_e0.01", "eps-margin loss, eps = 0.01"),
                          (ax[2], "_e0.05", "eps-margin loss, eps = 0.05")]:
        for mo, c, ls, mk, lab in [("B", ORANGE, "-", "s", "equal split (Model B)"),
                                   ("delay", PURPLE, "--", "v", "split proportional to a_i"),
                                   ("risk", GREEN, "-.", "^", "split proportional to a_i x U_i(0)")]:
            med = [100 * np.median(_rel(d[d.sigma == s], f"L_{mo}{suf}", f"L_A{suf}")) for s in sig]
            a.plot(range(5), med, ls, color=c, lw=2.2, marker=mk, ms=6, label=lab)
        a.axhline(0, color=BLUE, lw=2.5, label="A: coupled (= oracle split)")
        a.set_xticks(range(5), [f"{s:g}" for s in sig]); a.set_xlabel(r"timing slack $\sigma$"); a.set_title(title, loc="left")
    ax[0].set_ylabel("median loss above Model A, %")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/F6_split_rules.png", dpi=200); plt.close(fig)


def fig7():
    d = pd.read_csv(f"{RES}/E4_anchored.csv"); d = d[d.T2_us == 10_000]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, cr, title in [(ax[0], "software", "software crypto (servo a = 100 us/frame)"),
                         (ax[1], "hardware", "hardware crypto (servo a = 2.6 us/frame)")]:
        dc = d[d.crypto == cr]; u = 100 * dc.u
        for mo, c, ls, mk, lab in [("A", BLUE, "-", "o", "A: coupled (= oracle split)"), ("B", ORANGE, "-", "s", "equal split (Model B)"),
                                   ("delay", PURPLE, "--", "v", "split proportional to a_i"),
                                   ("risk", GREEN, "-.", "^", "split proportional to a_i x U_i(0)")]:
            a.plot(u, dc[f"L_{mo}"], ls, color=c, lw=2.2, marker=mk, ms=6, label=lab)
        a.set_xlabel("baseline loop utilisation before hardening, %"); a.set_title(title, loc="left")
    ax[0].set_ylabel("defender loss at Stackelberg eq.")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/F7_anchored.png", dpi=200); plt.close(fig)


def fig8():
    d = pd.read_csv(f"{RES}/E5_scalability.csv")
    med = d.groupby(["regime", "N", "R"]).wall_A.median()
    Ns = sorted(d.N.unique()); C = [BLUE, ORANGE, GREEN, YELLOW]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for c, R in zip(C, [5, 10, 20, 50]):
        ax[0].plot(Ns, [med["dense", n, R] for n in Ns], "-o", color=c, lw=2, ms=5, label=f"R = {R}, dense loops")
    ax[0].plot(Ns, [med["sparse", n, 50] for n in Ns], "--s", color=PINK, lw=2, ms=5, label="R = 50, loops of 10 assets")
    big = [n for n in Ns if n >= 100]
    ax[0].plot(big, med["dense", 1000, 50] * (np.array(big) / 1000) ** 2, ":", color="#52514e", lw=1.2, label="slope 2 reference")
    ax[0].set(xscale="log", yscale="log", xlabel="Number of assets N", ylabel="Wall-clock time per solve (s)")
    ax[0].legend(frameon=False, fontsize=7.5)
    for c, N in zip(C, [100, 500, 1000]):
        ax[1].plot([5, 10, 20, 50], [med["dense", N, R] for R in [5, 10, 20, 50]], "-o", color=c, lw=2, ms=5, label=f"N = {N}")
    ax[1].set(xscale="log", yscale="log", xlabel="Number of control loops R", ylabel="Wall-clock time per solve (s)")
    ax[1].xaxis.set_minor_locator(mt.NullLocator()); ax[1].set_xticks([5, 10, 20, 50], [5, 10, 20, 50])
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/F11_scalability.png", dpi=220); plt.close(fig)


def all_figures(include_e5=True):
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7] + ([fig8] if include_e5 else []):
        f()
