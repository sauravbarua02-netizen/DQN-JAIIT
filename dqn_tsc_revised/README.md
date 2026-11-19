# Revised DQN traffic-signal framework (JAIIT resubmission)

Corrections relative to the submitted code (all.headways.py): connected routes; matched-demand headway patterns
(traffic_patterns.py); keep/switch actions with automatic yellow (controllers.PhaseMachine); 23 normalised state features;
Double DQN with Huber loss; linear epsilon schedule; metrics from SUMO tripinfo; identical demand realizations for all
controllers per seed; corrected rule-based controller plus SUMO actuated, max-pressure and fixed-time benchmarks.

Requirements: SUMO (tested with 1.28.0, `pip install eclipse-sumo traci`), Python 3.12, torch, numpy, pandas, scipy, matplotlib.
Set SUMO_HOME and add SUMO's bin directory to PATH (the `sumo` binary must be callable).

    python run_experiments.py --train-episodes 40 --eval-seeds 5      # main study  -> results/main_results.csv
    python run_generalization.py --train-episodes 60 --eval-seeds 5   # mixed-training generalization
    python run_sensitivity.py --configs default queue_x0.5 queue_x2 speed_x0.5 no_imbalance
    python analyze_results.py                                         # paired stats (Holm) and figures

run_experiments.py is resumable (skips finished patterns/seeds). results/ contains the raw per-run results and the
learning curves produced for the manuscript; models/ the trained agents. Only one training run per pattern was made.
Note: the first-version NB/SB routes are not connected in net.net.xml; ROUTES in config.py defines the replacement set.
