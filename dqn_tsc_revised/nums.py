import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
M = pd.read_csv(f"{R}/main_results.csv"); P = pd.read_csv(f"{R}/analysis/paired_stats.csv")
LC = pd.read_csv(f"{R}/learning_curves.csv")
PATS = ["uniform","random","platoon","morning_peak","evening_peak","oversaturated","free_flow","mixed_fleet","lognormal","gamma","weibull","erlang","urban_arterial","highway_merge","incident","bimodal"]
CTRL = ["fixed_time","rule_original","rule_corrected","sumo_actuated","max_pressure","dqn"]
NAME = {"fixed_time":"Fixed-time","rule_original":"Rule-based (original logic)","rule_corrected":"Rule-based (corrected)","sumo_actuated":"SUMO actuated","max_pressure":"Max-pressure","dqn":"DQN"}
LABEL = {p: p.replace("_"," ").capitalize().replace("Free flow","Free-flow").replace("Lognormal","Log-normal") for p in PATS}
PM = M.groupby(["pattern","controller"]).agg(["mean","std"])
def pm(pat, c, m): return PM.loc[(pat,c),(m,"mean")], PM.loc[(pat,c),(m,"std")]
def agg(c, m):
    v = M[M.controller==c].groupby("pattern")[m].mean(); return v.mean(), v.std()
def vs(b, m="delay"):
    t = P[(P.b==b)&(P.metric==m)]
    return dict(pct=t.pct.mean(), better=int(((t.p_holm<.05)&(t["diff"]<0)).sum()), worse=int(((t.p_holm<.05)&(t["diff"]>0)).sum()),
                ns=int((t.p_holm>=.05).sum()), n=len(t))
def vs_dir(b, m):  # sign convention: 'better' respects metric direction
    t = P[(P.b==b)&(P.metric==m)]; low = m not in ("speed","throughput")
    good = (t["diff"]<0) if low else (t["diff"]>0)
    return dict(pct=t.pct.mean(), better=int(((t.p_holm<.05)&good).sum()), worse=int(((t.p_holm<.05)&~good).sum()), n=len(t))
def curves():
    lc = LC.copy(); lc["ep"] = lc.controller.str.extract(r"ep(\d+)").astype(int)
    e = lc.groupby(["pattern","ep"]).delay.mean().unstack()
    return dict(first5=e.loc[:,1:5].mean().mean(), last5=e.loc[:,36:40].mean().mean(),
                last10_cv=(e.loc[:,31:40].std(axis=1)/e.loc[:,31:40].mean(axis=1)).mean()*100,
                drift=((e.loc[:,36:40].mean(axis=1)-e.loc[:,26:35].mean(axis=1))/e.loc[:,26:35].mean(axis=1)*100).abs().mean())
def gen():
    g = pd.read_csv(f"{R}/generalization.csv"); a = pd.concat([g, M])
    w = a.pivot_table(index=["pattern","seed"], columns="controller", values="delay").reset_index()
    seen = ["uniform","platoon","gamma","bimodal","morning_peak","oversaturated"]; w["seen"] = w.pattern.isin(seen)
    return w, seen
if __name__ == "__main__":
    for c in CTRL: print(c, [round(agg(c,m)[0],2) for m in ["queue","speed","travel_time","delay","throughput","unfinished"]])
    for b in CTRL[:-1]: print(b, vs(b), vs_dir(b,"queue"), vs_dir(b,"speed"))
    print(curves())
    w, seen = gen(); print(w.groupby("seen")[["dqn_mixed","dqn","sumo_actuated"]].mean())
    t = P[(P.b=="sumo_actuated")&(P.metric=="delay")][["pattern","a_mean","b_mean","pct","p_holm"]]; print(t.round(3).to_string())
