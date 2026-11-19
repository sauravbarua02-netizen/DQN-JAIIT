"""Headway-pattern generators with *matched mean demand*.

Original code drew a fixed number of vehicles (= flow * 1 h) whose headways had pattern-specific means
(e.g. 2.5 s for 'random'), so patterns differed in intensity and the demand often ended long before
t = 3600 s. Here each pattern keeps its original *shape* (distribution family, relative variability,
bunching) but the headway sequence of every route stream is rescaled so that exactly
n = flow * DURATION / 3600 vehicles depart in [0, DURATION). All patterns therefore have the same mean flow
per route (except the explicit demand-level patterns oversaturated / free_flow).
"""
import numpy as np
import config as C

def _raw_headways(pattern, n, rng):
    """Raw headway sequence of length n, in seconds, taken from the original generator definitions."""
    if pattern == "uniform":      return rng.exponential(1.0, n)
    if pattern == "random":       return np.clip(np.abs(rng.normal(2.5, 1.2, n)), 0.5, 10)
    if pattern in ("platoon", "urban_arterial", "highway_merge"):
        out = []
        while len(out) < n:
            if pattern == "platoon":
                if rng.random() < 0.3: out += list(rng.uniform(0.8, 1.5, rng.integers(3, 7)))
                else:                  out.append(rng.uniform(3.0, 6.0))
            elif pattern == "urban_arterial":
                if rng.random() < 0.4:
                    base = [0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.8]; out += base[:min(rng.integers(4, 11), 7)]
                else: out.append(rng.uniform(4.0, 8.0))
            else:
                if rng.random() < 0.25: out += list(rng.uniform(1.8, 2.2, rng.integers(2, 5)))
                else:                   out.append(rng.uniform(4.0, 10.0))
        return np.array(out[:n])
    if pattern == "lognormal":    return rng.lognormal(0.8, 0.5, n)
    if pattern == "gamma":        return rng.gamma(2.0, 1.2, n)
    if pattern == "weibull":      return rng.weibull(1.5, n) * 2.5
    if pattern == "erlang":       return rng.gamma(3.0, 0.8, n)
    if pattern in ("oversaturated",): return rng.exponential(1.0, n)
    if pattern == "free_flow":    return rng.exponential(1.0, n)
    if pattern == "mixed_fleet":
        truck = rng.random(n) < 0.10
        return np.where(truck, rng.uniform(3.0, 5.0, n), rng.uniform(1.5, 3.0, n))
    if pattern == "incident":
        h = rng.exponential(2.5, n); a, b = n // 3, 2 * n // 3; h[a:b] = rng.exponential(1.2, b - a); return h
    if pattern == "bimodal":
        cong = rng.random(n) < 0.6
        return np.where(cong, rng.exponential(1.8, n), rng.exponential(4.5, n))
    if pattern in ("morning_peak", "evening_peak"):
        lo, hi = (0.3, 0.6) if pattern == "morning_peak" else (0.4, 0.7)
        i = np.arange(n); peak = (i >= lo * n) & (i < hi * n)
        return np.where(peak, rng.uniform(1.2, 2.0, n), rng.uniform(3.0, 5.0, n))
    raise ValueError(pattern)

def departures(pattern, flow_vph, rng):
    """Departure times (s) of one route stream: exactly round(flow*DURATION/3600) vehicles in [0, DURATION)."""
    n = int(round(flow_vph * C.DURATION / 3600))
    raw = _raw_headways(pattern, n, rng)
    k = (C.DURATION - 5.0) / raw.sum()          # rescale so the n-th vehicle departs at ~DURATION-5 s
    t = rng.uniform(0, 5.0) + np.cumsum(raw * k)
    return t[t < C.DURATION]

def write_route_file(pattern, seed, path):
    rng = np.random.default_rng([sum(map(ord, pattern)), seed])
    scale = C.FLOW_SCALE.get(pattern, 1.0)
    rows = []
    for route in C.ROUTES:
        flow = C.BASE_FLOW[route] * scale
        for i, t in enumerate(departures(pattern, flow, rng)):
            vt = "truck" if (pattern == "mixed_fleet" and rng.random() < 0.10) else "car"
            rows.append((t, f'    <vehicle id="{route}_{i}" type="{vt}" route="{route}" depart="{t:.2f}"/>'))
    rows.sort(key=lambda r: r[0])
    with open(path, "w") as f:
        f.write('<routes>\n    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>\n')
        f.write('    <vType id="truck" accel="1.5" decel="3.0" sigma="0.3" length="12" maxSpeed="40"/>\n')
        for r, e in C.ROUTES.items(): f.write(f'    <route id="{r}" edges="{e}"/>\n')
        f.write("\n".join(r[1] for r in rows) + "\n</routes>\n")
    return len(rows)

def describe(pattern, seed=0, route="EB"):
    """Realised statistics for Table 1 / Appendix A."""
    rng = np.random.default_rng(seed)
    flow = C.BASE_FLOW[route] * C.FLOW_SCALE.get(pattern, 1.0)
    t = departures(pattern, flow, rng); h = np.diff(t)
    return dict(pattern=pattern, flow_vph=flow, n=len(t), mean_h=h.mean(), sd_h=h.std(), cv=h.std() / h.mean(),
                min_h=h.min(), p95_h=np.percentile(h, 95), max_h=h.max())
