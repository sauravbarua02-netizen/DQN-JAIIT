"""Train / evaluate controllers. Results are appended to a tidy CSV (one row per run) so that runs can be resumed
and analysed with analyze_results.py.

Examples
  python run_experiments.py --patterns uniform platoon --train-episodes 40 --eval-seeds 5 --out results/run1.csv
"""
import argparse, os, random, time, csv, json, tempfile
import numpy as np, torch
import config as C, traffic_patterns as tp
from sumo_env import Episode
import controllers as K
from dqn import DQNController

METRICS = ["queue", "speed", "travel_time", "delay", "throughput", "unfinished"]

def run_episode(name, pattern, seed, dqn=None, workdir=None, train=False):
    d = workdir or tempfile.mkdtemp(prefix="route_")
    route = os.path.join(d, f"{pattern}_{seed}_{os.getpid()}.rou.xml")
    tp.write_route_file(pattern, seed, route)
    ep = Episode(route, seed, use_sumo_actuated=(name == "sumo_actuated"), workdir=d)
    ep.start()
    if name == "dqn": ctl = dqn; ctl.training = train; ctl.begin_episode()
    elif name == "rule_original": ctl = K.RuleBased("original")
    elif name == "rule_corrected": ctl = K.RuleBased("corrected")
    elif name == "max_pressure": ctl = K.MaxPressure()
    else: ctl = None                       # fixed_time and sumo_actuated are driven by SUMO programs
    log = []
    for step in range(C.DURATION):
        ep.step()
        if ctl is not None:
            ctl.step(step, final=(step == C.DURATION - 1)) if name == "dqn" else ctl.step(step)
    m = ep.close(); os.remove(route)
    return m

def append(path, rows):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "controller", "seed"] + METRICS)
        if new: w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", default=C.PATTERNS)
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--train-seed", type=int, default=0, help="seed of the DQN initialisation / exploration")
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--controllers", nargs="+", default=["fixed_time", "rule_original", "rule_corrected", "sumo_actuated", "max_pressure", "dqn"])
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--curves", default="results/learning_curves.csv")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)
    for pattern in a.patterns:
        t0 = time.time(); dqn = None
        done = set()
        if os.path.exists(a.out):
            import pandas as pd; d0 = pd.read_csv(a.out); done = set(d0[d0.pattern == pattern].seed)
        mpath = os.path.join(a.models, f"dqn_{pattern}.pt")
        if "dqn" in a.controllers and os.path.exists(mpath):          # resume: reuse the trained agents
            dqn = DQNController(total_train_steps=1)
            for t, sd in torch.load(mpath).items(): dqn.agents[t].q.load_state_dict(sd)
        elif "dqn" in a.controllers:
            random.seed(a.train_seed); np.random.seed(a.train_seed); torch.manual_seed(a.train_seed)
            dqn = DQNController(total_train_steps=a.train_episodes * (C.DURATION // C.DECISION_INTERVAL))
            for e in range(a.train_episodes):
                m = run_episode("dqn", pattern, 10_000 + e, dqn, train=True)   # training seeds 10000+
                append(a.curves, [dict(pattern=pattern, controller=f"dqn_train_ep{e+1}", seed=10_000 + e, **m)])
                print(f"[{pattern}] train ep {e+1}/{a.train_episodes} eps={dqn.epsilon():.2f} delay={m['delay']:.1f} queue={m['queue']:.2f}", flush=True)
            torch.save({t: ag.q.state_dict() for t, ag in dqn.agents.items()}, os.path.join(a.models, f"dqn_{pattern}.pt"))
        for seed in range(1, a.eval_seeds + 1):           # evaluation seeds 1..K, identical demand for every controller
            if seed in done: continue
            rows = []
            for name in a.controllers:
                m = run_episode(name, pattern, seed, dqn)
                rows.append(dict(pattern=pattern, controller=name, seed=seed, **m))
            append(a.out, rows)
            print(f"[{pattern}] eval seed {seed}: " + " ".join(f"{r['controller']}={r['delay']:.0f}" for r in rows), flush=True)
        if dqn is not None:
            json.dump(dqn.action_counts, open(os.path.join(a.models, f"actions_{pattern}.json"), "w"))
        print(f"[{pattern}] done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
