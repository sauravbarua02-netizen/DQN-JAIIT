"""Thin SUMO wrapper: one episode = one simulated hour under one controller. Metrics come from SUMO's own
tripinfo output (travel time, time loss) instead of hand-rolled bookkeeping."""
import os, tempfile, xml.etree.ElementTree as ET
import numpy as np
try:
    import libsumo as traci
except ImportError:
    import traci
import config as C

def actuated_additional():
    """tlLogic of type 'actuated' with the same phases/min/max green as the other controllers (SUMO built-in)."""
    if os.path.exists(C.ACTUATED_ADD): return
    with open(C.ACTUATED_ADD, "w") as f:
        f.write("<additional>\n")
        for t in C.TLS_IDS:
            f.write(f'  <tlLogic id="{t}" type="actuated" programID="act" offset="0">\n'
                    f'    <phase duration="30" minDur="{C.MIN_GREEN}" maxDur="{C.MAX_GREEN}" state="rrrrGGGGgrrrrGGGGg"/>\n'
                    f'    <phase duration="{C.YELLOW}" state="rrrryyyyyrrrryyyyy"/>\n'
                    f'    <phase duration="30" minDur="{C.MIN_GREEN}" maxDur="{C.MAX_GREEN}" state="GGGgrrrrrGGGgrrrrr"/>\n'
                    f'    <phase duration="{C.YELLOW}" state="yyyyrrrrryyyyrrrrr"/>\n  </tlLogic>\n')
        f.write("</additional>\n")

class Episode:
    def __init__(self, route_file, seed, use_sumo_actuated=False, workdir=None):
        self.route_file, self.seed, self.act = route_file, seed, use_sumo_actuated
        self.workdir = workdir or tempfile.mkdtemp(prefix="sumo_")
        self.tripinfo = os.path.join(self.workdir, f"trip_{os.getpid()}.xml")
    def start(self):
        cmd = ["sumo", "-n", C.NET, "-r", self.route_file, "--seed", str(self.seed), "--no-warnings", "true",
               "--no-step-log", "true", "--tripinfo-output", self.tripinfo, "--end", str(C.DURATION),
               "--time-to-teleport", "300"]
        if self.act:
            actuated_additional(); cmd += ["-a", C.ACTUATED_ADD]
        traci.start(cmd)
        if self.act:
            for t in C.TLS_IDS: traci.trafficlight.setProgram(t, "act")
        self.halt_samples, self.speed_samples, self.arrived = [], [], 0
        self.lanes = [l for t in C.TLS_IDS for a in C.APPROACHES[t] for l in a]
        self.edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    def step(self):
        traci.simulationStep()
        t = traci.simulation.getTime()
        if t >= C.WARMUP:
            self.arrived += traci.simulation.getArrivedNumber()
            if int(t) % 10 == 0:
                self.halt_samples.append(np.mean([traci.lane.getLastStepHaltingNumber(l) for l in self.lanes]))
                n = np.array([traci.edge.getLastStepVehicleNumber(e) for e in self.edges], float)
                v = np.array([traci.edge.getLastStepMeanSpeed(e) for e in self.edges])
                if n.sum() > 0: self.speed_samples.append((n * v).sum() / n.sum() * 3.6)
    def close(self):
        unfinished = traci.simulation.getMinExpectedNumber()
        teleports = 0
        traci.close()
        tt, tl = [], []
        for trip in ET.parse(self.tripinfo).getroot().iter("tripinfo"):
            if float(trip.get("depart")) >= C.WARMUP:
                tt.append(float(trip.get("duration"))); tl.append(float(trip.get("timeLoss")) + float(trip.get("departDelay")))
        try: os.remove(self.tripinfo)
        except OSError: pass
        return dict(queue=float(np.mean(self.halt_samples)), speed=float(np.mean(self.speed_samples)),
                    travel_time=float(np.mean(tt)), delay=float(np.mean(tl)),
                    throughput=self.arrived * 3600.0 / (C.DURATION - C.WARMUP), unfinished=int(unfinished))
