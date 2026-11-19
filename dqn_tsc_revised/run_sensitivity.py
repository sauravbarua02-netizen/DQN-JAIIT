"""Reward-weight sensitivity: scale one reward weight at a time (x0.5, x2) or drop the imbalance term, retrain, evaluate."""
import argparse, random, numpy as np, torch
import config as C
import run_experiments as R
from dqn import DQNController

BASE = dict(W_QUEUE=C.W_QUEUE, W_DELAY=C.W_DELAY, W_SPEED=C.W_SPEED, W_IMBALANCE=C.W_IMBALANCE)
CONFIGS = {"default": {}, "queue_x0.5": {"W_QUEUE": 0.5}, "queue_x2": {"W_QUEUE": 2.0}, "delay_x0.5": {"W_DELAY": 0.5},
           "delay_x2": {"W_DELAY": 2.0}, "speed_x0.5": {"W_SPEED": 0.5}, "speed_x2": {"W_SPEED": 2.0}, "no_imbalance": {"W_IMBALANCE": 0.0}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", default=["uniform", "oversaturated"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--train-episodes", type=int, default=30); ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--out", default="results/sensitivity.csv")
    a = ap.parse_args()
    for name in a.configs:
        for k, v in BASE.items(): setattr(C, k, v * CONFIGS[name].get(k, 1.0))
        for p in a.patterns:
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            dqn = DQNController(total_train_steps=a.train_episodes * (C.DURATION // C.DECISION_INTERVAL))
            for e in range(a.train_episodes): R.run_episode("dqn", p, 10_000 + e, dqn, train=True)
            for s in range(1, a.eval_seeds + 1):
                m = R.run_episode("dqn", p, s, dqn, train=False)
                R.append(a.out, [dict(pattern=p, controller=f"dqn_{name}", seed=s, **m)])
            print("done", name, p, flush=True)
    for k, v in BASE.items(): setattr(C, k, v)

if __name__ == "__main__":
    main()
