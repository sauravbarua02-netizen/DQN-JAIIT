"""Cross-pattern generalisation: ONE set of agents trained on a mixture of patterns, evaluated on all sixteen
(including patterns never seen in training). Evaluation seeds are the same as in run_experiments.py, so results
can be paired with the pattern-specific agents and the baselines."""
import argparse, random, numpy as np, torch
import config as C
import run_experiments as R
from dqn import DQNController

TRAIN_SET = ["uniform", "platoon", "gamma", "bimodal", "morning_peak", "oversaturated"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=60); ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out", default="results/generalization.csv"); ap.add_argument("--curves", default="results/generalization_curves.csv")
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dqn = DQNController(total_train_steps=a.train_episodes * (C.DURATION // C.DECISION_INTERVAL))
    for e in range(a.train_episodes):
        p = TRAIN_SET[e % len(TRAIN_SET)]
        m = R.run_episode("dqn", p, 20_000 + e, dqn, train=True)
        R.append(a.curves, [dict(pattern=p, controller=f"mixed_train_ep{e+1}", seed=20_000 + e, **m)])
        print(f"mixed train ep {e+1} [{p}] delay={m['delay']:.1f}", flush=True)
    for p in C.PATTERNS:
        for s in range(1, a.eval_seeds + 1):
            m = R.run_episode("dqn", p, s, dqn, train=False)
            R.append(a.out, [dict(pattern=p, controller="dqn_mixed", seed=s, **m)])
        print("eval", p, "seen" if p in TRAIN_SET else "UNSEEN", flush=True)

if __name__ == "__main__":
    main()
