"""Signal controllers. All manual controllers share one phase state machine (green -> yellow -> green) so that
no controller can skip the yellow phase. Phase 0 = E/W green, 2 = N/S green, 1 and 3 = yellow."""
import numpy as np
try:
    import libsumo as traci
except ImportError:
    import traci
import config as C

def _count(lanes, fn): return sum(fn(l) for l in lanes)
def approach_queue(tls, idx):  # idx 0,1 = N,S ; 2,3 = E,W
    return sum(_count(C.APPROACHES[tls][i], traci.lane.getLastStepHaltingNumber) for i in idx)
NS, EW = (0, 1), (2, 3)

class PhaseMachine:
    """Holds green until the controller asks for a switch; inserts the yellow."""
    def __init__(self):
        self.green = {t: 0 for t in C.TLS_IDS}; self.start = {t: 0.0 for t in C.TLS_IDS}
        self.yellow_end = {t: None for t in C.TLS_IDS}
        for t in C.TLS_IDS:
            traci.trafficlight.setPhase(t, 0); traci.trafficlight.setPhaseDuration(t, 1e5)
    def elapsed(self, t): return traci.simulation.getTime() - self.start[t]
    def in_green(self, t): return self.yellow_end[t] is None
    def request_switch(self, t):
        if not self.in_green(t): return
        y = self.green[t] + 1
        traci.trafficlight.setPhase(t, y); traci.trafficlight.setPhaseDuration(t, C.YELLOW)
        self.yellow_end[t] = traci.simulation.getTime() + C.YELLOW
    def tick(self):  # call every simulation step
        now = traci.simulation.getTime()
        for t in C.TLS_IDS:
            if self.yellow_end[t] is not None and now >= self.yellow_end[t]:
                g = (self.green[t] + 2) % 4
                traci.trafficlight.setPhase(t, g); traci.trafficlight.setPhaseDuration(t, 1e5)
                self.green[t], self.start[t], self.yellow_end[t] = g, now, None

class RuleBased:
    """Queue-based adaptive rule. mode='original' reproduces the logic of the submitted Algorithm 1 (switch as
    soon as minimum green is reached and the extension condition is false). mode='corrected' holds the green
    while the served direction has a queue ratio advantage or the opposing direction is empty, up to MAX_GREEN."""
    def __init__(self, mode="corrected"):
        self.mode, self.pm = mode, PhaseMachine()
    def step(self, step):
        self.pm.tick()
        if step % C.DECISION_INTERVAL: return
        for t in C.TLS_IDS:
            if not self.pm.in_green(t): continue
            el = self.pm.elapsed(t)
            if el < C.MIN_GREEN: continue
            cur, opp = (EW, NS) if self.pm.green[t] == 0 else (NS, EW)
            qc, qo = approach_queue(t, cur), approach_queue(t, opp)
            extend = qc > 1.2 * qo
            if self.mode == "original":
                switch = not extend or el >= C.MAX_GREEN
            else:
                switch = el >= C.MAX_GREEN or (not extend and qo > 0)
            if switch: self.pm.request_switch(t)

class MaxPressure:
    """Max-pressure control (Varaiya 2013): serve the phase with the larger pressure (upstream minus downstream queue)."""
    def __init__(self):
        self.pm = PhaseMachine(); self.links = {}
        for t in C.TLS_IDS:
            self.links[t] = [l[0] for l in traci.trafficlight.getControlledLinks(t) if l]
    def pressure(self, t, green):
        state = traci.trafficlight.getAllProgramLogics(t)[0].phases[green].state
        p = 0.0
        for i, s in enumerate(state):
            if s in "Gg" and i < len(self.links[t]):
                inl, outl, _ = self.links[t][i]
                p += traci.lane.getLastStepHaltingNumber(inl) - traci.lane.getLastStepHaltingNumber(outl)
        return p
    def step(self, step):
        self.pm.tick()
        if step % C.DECISION_INTERVAL: return
        for t in C.TLS_IDS:
            if not self.pm.in_green(t): continue
            el = self.pm.elapsed(t)
            if el < C.MIN_GREEN: continue
            cur = self.pm.green[t]; oth = (cur + 2) % 4
            if el >= C.MAX_GREEN or self.pressure(t, oth) > self.pressure(t, cur): self.pm.request_switch(t)
