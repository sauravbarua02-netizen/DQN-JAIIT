"""Revised multi-agent dueling Double-DQN. One agent per intersection, actions {keep, switch}."""
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try:
    import libsumo as traci
except ImportError:
    import traci
import config as C
from controllers import PhaseMachine, approach_queue

class DuelingDQN(nn.Module):
    def __init__(self, s=C.STATE_DIM, a=C.ACTION_DIM):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(s, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.v = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.a = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, a))
    def forward(self, x):
        h = self.f(x); a = self.a(h); return self.v(h) + a - a.mean(-1, keepdim=True)

class Agent:
    def __init__(self, tls):
        self.tls = tls; self.q = DuelingDQN(); self.tq = DuelingDQN(); self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=C.LR)
        self.mem = []; self.pos = 0; self.updates = 0; self.last = None
    def state(self, pm):
        f = []
        for lanes in C.APPROACHES[self.tls]:
            n = len(lanes)
            halt = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
            spd = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes]) * 3.6
            occ = np.mean([traci.lane.getLastStepOccupancy(l) for l in lanes]) / 100.0
            cnt = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            f += [min(halt / (C.MAX_QUEUE_PER_APPROACH), 1), min(wait / (C.MAX_WAIT * n), 1),
                  min(spd / C.FREE_SPEED_KMH, 1), min(occ, 1), min(cnt / (C.MAX_QUEUE_PER_APPROACH * 1.5), 1)]
        g = pm.green[self.tls]; f += [float(g == 0), float(g == 2), min(pm.elapsed(self.tls) / C.MAX_GREEN, 1)]
        return np.array(f, np.float32)
    def reward(self):
        qs = [sum(traci.lane.getLastStepHaltingNumber(l) for l in a) for a in C.APPROACHES[self.tls]]
        w = sum(traci.lane.getWaitingTime(l) for a in C.APPROACHES[self.tls] for l in a)
        sp = np.mean([traci.lane.getLastStepMeanSpeed(l) for a in C.APPROACHES[self.tls] for l in a]) * 3.6
        qn = min(sum(qs) / (4 * C.MAX_QUEUE_PER_APPROACH), 1); wn = min(w / C.MAX_WAIT, 1); sn = min(sp / C.FREE_SPEED_KMH, 1)
        cv = np.std(qs) / (np.mean(qs) + 1e-6) if np.mean(qs) > 0 else 0
        imb = C.W_IMBALANCE * (cv - 0.7) if cv > 0.7 else 0.0
        thr = min(sum(traci.lane.getLastStepVehicleNumber(l) for a in C.APPROACHES[self.tls] for l in a) / 5, 1)
        return C.W_QUEUE * qn + C.W_DELAY * wn + C.W_SPEED * sn + C.W_THROUGHPUT * thr + imb
    def remember(self, *x):
        if len(self.mem) < C.MEMORY: self.mem.append(x)
        else: self.mem[self.pos] = x
        self.pos = (self.pos + 1) % C.MEMORY
    def learn(self):
        if len(self.mem) < max(C.BATCH, C.TRAIN_START): return
        s, a, r, s2, d = map(np.array, zip(*random.sample(self.mem, C.BATCH)))
        s, s2 = torch.tensor(s), torch.tensor(s2); a = torch.tensor(a).long().unsqueeze(1)
        r, d = torch.tensor(r, dtype=torch.float32), torch.tensor(d, dtype=torch.float32)
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)              # Double DQN
            y = r + C.GAMMA * self.tq(s2).gather(1, a2).squeeze(1) * (1 - d)
        loss = F.smooth_l1_loss(self.q(s).gather(1, a).squeeze(1), y)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 1.0); self.opt.step()
        self.updates += 1
        if self.updates % C.TARGET_SYNC_UPDATES == 0: self.tq.load_state_dict(self.q.state_dict())

class DQNController:
    def __init__(self, total_train_steps=1, agents=None):
        self.agents = agents or {t: Agent(t) for t in C.TLS_IDS}
        self.training = True; self.total = max(total_train_steps, 1); self.n = 0; self.pm = None
        self.action_counts = {t: [0, 0] for t in C.TLS_IDS}
    def epsilon(self):
        frac = min(self.n / (C.EPS_FRACTION * self.total), 1.0)
        return C.EPS_START + frac * (C.EPS_END - C.EPS_START)
    def begin_episode(self):
        self.pm = PhaseMachine()
        for a in self.agents.values(): a.last = None
    def step(self, step, final=False):
        self.pm.tick()
        if step % C.DECISION_INTERVAL: return
        eps = self.epsilon() if self.training else 0.0
        for t, ag in self.agents.items():
            if not self.pm.in_green(t): continue
            s = ag.state(self.pm)
            if ag.last is not None and self.training:
                ps, pa = ag.last; ag.remember(ps, pa, ag.reward(), s, float(final))
                if self.n % C.TRAIN_FREQ == 0: ag.learn()
            el = self.pm.elapsed(t)
            if el < C.MIN_GREEN: act = 0
            elif el >= C.MAX_GREEN: act = 1
            elif random.random() < eps: act = random.randint(0, 1)
            else:
                with torch.no_grad(): act = int(ag.q(torch.tensor(s).unsqueeze(0)).argmax(1))
            if el >= C.MIN_GREEN and el < C.MAX_GREEN or el >= C.MAX_GREEN:
                ag.last = (s, act)
            else: ag.last = None
            self.action_counts[t][act] += 1
            if act == 1: self.pm.request_switch(t)
        if self.training: self.n += 1
