"""Statistics and figures for the revised study.

Reads results/main_results.csv (one row per pattern x controller x seed, identical demand for every controller within
a seed) and writes: stats table (paired comparison DQN vs each benchmark, per pattern and metric, Holm-corrected),
aggregate table, and figures.  Usage: python analyze_results.py [--results results/main_results.csv]
"""
import argparse, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

METRICS = ["queue", "speed", "travel_time", "delay", "throughput"]
LOWER_BETTER = {"queue": True, "speed": False, "travel_time": True, "delay": True, "throughput": False, "unfinished": True}

def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o): run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def paired(df, a, b, metric):
    rows = []
    for pat, g in df.groupby("pattern"):
        w = g.pivot_table(index="seed", columns="controller", values=metric)
        if a not in w or b not in w: continue
        w = w.dropna(subset=[a, b]); x, y = w[a].values, w[b].values; n = len(x)
        if n < 3: continue
        d = x - y; sd = d.std(ddof=1)
        ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=sd / np.sqrt(n)) if sd > 0 else (d.mean(), d.mean())
        p = stats.ttest_rel(x, y).pvalue if sd > 0 else 1.0
        rows.append(dict(pattern=pat, metric=metric, a=a, b=b, n=n, a_mean=x.mean(), a_sd=x.std(ddof=1), b_mean=y.mean(),
                         b_sd=y.std(ddof=1), diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1], pct=100 * d.mean() / y.mean() if y.mean() else np.nan,
                         p=p, d=d.mean() / sd if sd > 0 else np.nan))
    t = pd.DataFrame(rows)
    if len(t): t["p_holm"] = holm(t["p"])
    return t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results/main_results.csv")
    ap.add_argument("--curves", default="results/learning_curves.csv"); ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.results)
    ctrls = [c for c in df.controller.unique() if c != "dqn"]
    allp = pd.concat([paired(df, "dqn", b, m) for b in ctrls for m in METRICS + ["unfinished"]], ignore_index=True)
    allp.to_csv(f"{a.out}/paired_stats.csv", index=False)
    # aggregate over patterns: mean of per-pattern means (+ SD across patterns), by controller
    pm = df.groupby(["pattern", "controller"])[METRICS + ["unfinished"]].mean().reset_index()
    agg = pm.groupby("controller")[METRICS + ["unfinished"]].agg(["mean", "std"])
    agg.to_csv(f"{a.out}/aggregate_by_controller.csv")
    # figure 1: delay by pattern for each controller (mean +- SD over seeds)
    order = ["fixed_time", "rule_original", "rule_corrected", "sumo_actuated", "max_pressure", "dqn"]
    order = [c for c in order if c in ctrls + ["dqn"]]
    pats = [p for p in pd.unique(df.pattern)]
    fig, ax = plt.subplots(figsize=(12, 4.6)); w = 0.8 / len(order)
    for i, c in enumerate(order):
        g = df[df.controller == c].groupby("pattern").delay.agg(["mean", "std"]).reindex(pats)
        ax.bar(np.arange(len(pats)) + i * w, g["mean"], w, yerr=g["std"], capsize=1.5, label=c.replace("_", " "))
    ax.set_xticks(np.arange(len(pats)) + 0.4 - w / 2); ax.set_xticklabels(pats, rotation=45, ha="right"); ax.set_ylabel("Mean delay (s)")
    ax.legend(ncol=3, fontsize=8); ax.set_title("Delay by headway pattern (mean ± SD over evaluation seeds)"); fig.tight_layout()
    fig.savefig(f"{a.out}/fig_delay_by_pattern.png", dpi=170); plt.close(fig)
    # figure 2: % change of DQN vs best benchmark per pattern (delay)
    base = [c for c in ctrls if c != "fixed_time" and c != "rule_original"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for b in ["rule_corrected", "sumo_actuated", "max_pressure"]:
        if b in ctrls:
            t = allp[(allp.b == b) & (allp.metric == "delay")].set_index("pattern").reindex(pats)
            ax.plot(range(len(pats)), -t.pct, "o-", label=f"vs {b.replace('_',' ')}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(pats))); ax.set_xticklabels(pats, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Delay reduction by DQN (%)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{a.out}/fig_improvement.png", dpi=170); plt.close(fig)
    # figure 3: learning curves
    if os.path.exists(a.curves):
        lc = pd.read_csv(a.curves); lc["ep"] = lc.controller.str.extract(r"ep(\d+)").astype(int)
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        for pat, g in lc.groupby("pattern"):
            g = g.sort_values("ep"); axs[0].plot(g.ep, g.delay, lw=0.7, alpha=0.5)
        m = lc.groupby("ep").delay.agg(["mean", "std"]); axs[1].plot(m.index, m["mean"], "k-", label="mean over patterns")
        axs[1].fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.2)
        for ax, t in zip(axs, ["Per pattern", "Mean ± SD across patterns"]): ax.set_xlabel("Training episode"); ax.set_ylabel("Mean delay (s)"); ax.set_title(t)
        fig.tight_layout(); fig.savefig(f"{a.out}/fig_learning_curves.png", dpi=170); plt.close(fig)
    print(agg.round(2).T.to_string())
    sig = allp[allp.p_holm < 0.05].groupby(["b", "metric"]).size()
    print(sig.to_string())

if __name__ == "__main__":
    main()
