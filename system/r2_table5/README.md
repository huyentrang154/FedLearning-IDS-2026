# Table 5 results (round-2 revision)

Sample-level label noise on 30% of clients (rc = 0.3, poisoned clients 0..5), strength eps = 0.4,
Asymmetric and Symmetric transitions; `noniid20` = pat partition (kappa = 2), 20 clients;
8 methods x 3 datasets (NSL-KDD, UNSW-NB15, CIC-IDS17) x 3 seeds (0, 1, 2) = 144 runs.
Hyper-parameters = `system/main.py` defaults (Table 2).

- `summary/summary_mean_std.{csv,xlsx}`: mean / std over seeds (metric used in the paper: `f1-normal`,
  the sample-weighted mean of per-client macro-F1 over benign clients, best round).
- `summary/summary_per_seed.csv`: one row per run.
- `<dataset>/noniid20/<algorithm>/<experiment>/seed_<k>/tracking.log`: full per-round log of each run.

FedNoRo was re-run with the macro-F1 client code (`clientNoRo.py` from origin/main); the old weighted-F1
FedNoRo runs are not included. Model checkpoints (.pt / .pkl, ~28 GB) are not committed.
