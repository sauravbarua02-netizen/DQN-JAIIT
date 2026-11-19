"""Central configuration for the revised framework (all values that appear in the paper's tables)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(HERE, "sumo", "net.net.xml")
ACTUATED_ADD = os.path.join(HERE, "sumo", "actuated.add.xml")

DURATION = 3600          # s, simulated horizon per episode
WARMUP = 300             # s, vehicles departing before this are excluded from metrics
TLS_IDS = ["J1", "J2", "J3"]
DECISION_INTERVAL = 5    # s between control decisions
MIN_GREEN, MAX_GREEN, YELLOW = 20, 60, 3

# Approach lanes per intersection: [N, S, E, W]; phase 0 = E/W green, phase 2 = N/S green
APPROACHES = {
    "J1": [["E4_0", "E4_1"], ["-E5_0", "-E5_1"], ["-E1_0", "-E1_1", "-E1_2"], ["E0_0", "E0_1", "E0_2"]],
    "J2": [["E6_0", "E6_1"], ["-E7_0", "-E7_1"], ["-E2_0", "-E2_1", "-E2_2"], ["E1_0", "E1_1", "E1_2"]],
    "J3": [["E8_0", "E8_1"], ["-E9_0", "-E9_1"], ["-E3_0", "-E3_1", "-E3_2"], ["E2_0", "E2_1", "E2_2"]],
}
# NOTE: the submitted route definitions "E4 E1 E6" / "-E5 -E1 -E7" are not connected in net.net.xml (SUMO >= 1.28 aborts with
# "no valid route"). The revised set below uses only connected edge sequences: the arterial through movements plus a
# through minor-street movement (N->S and S->N) at each of the three intersections.
ROUTES = {"EB": "E0 E1 E2 E3", "WB": "-E3 -E2 -E1 -E0",
          "S1": "E4 E5", "N1": "-E5 -E4", "S2": "E6 E7", "N2": "-E7 -E6", "S3": "E8 E9", "N3": "-E9 -E8"}

# Base mean demand per route stream (veh/h); identical for every pattern unless a demand-level pattern says otherwise
BASE_FLOW = {"EB": 600, "WB": 600, "S1": 200, "N1": 200, "S2": 200, "N2": 200, "S3": 200, "N3": 200}
FLOW_SCALE = {"oversaturated": 1.5, "free_flow": 0.5}

PATTERNS = ["uniform", "random", "platoon", "lognormal", "gamma", "weibull", "erlang",
            "oversaturated", "free_flow", "mixed_fleet", "urban_arterial", "highway_merge",
            "incident", "bimodal", "morning_peak", "evening_peak"]

# DQN
STATE_DIM = 23
ACTION_DIM = 2           # 0 = keep current green, 1 = switch (yellow inserted automatically)
LR = 5e-4
GAMMA = 0.95
BATCH = 64
MEMORY = 50000
TRAIN_START = 500
TRAIN_FREQ = 2
TARGET_SYNC_UPDATES = 250
EPS_START, EPS_END, EPS_FRACTION = 1.0, 0.05, 0.6   # epsilon reaches EPS_END after 60% of training steps

# Reward (unchanged weights from the original submission; every term normalised to [0,1])
W_QUEUE, W_DELAY, W_SPEED, W_THROUGHPUT, W_IMBALANCE = -5.0, -3.0, 4.0, 0.1, -2.0
MAX_QUEUE_PER_APPROACH, MAX_WAIT, FREE_SPEED_KMH = 15, 200, 50.0
