"""Summarize the timing runs (main.py --timing_out) into one table.

usage: python summarize_timing.py [--dir ../results/timing]
Prints, per method: wall-clock training time as if the clients ran in parallel (the number for the
cost table), the sequential simulation time, server time, and evaluation / bookkeeping time (both excluded).
"""
import argparse, glob, json, os

ORDER = ['fedavg', 'fedgh', 'fedrep', 'fedhome', 'lg_fedavg', 'fedcorr', 'fednoro', 'fedus']
NAMES = {'fedavg': 'FedAvg', 'fedgh': 'FedGH', 'fedrep': 'FedRep', 'fedhome': 'FedHome',
         'lg_fedavg': 'LG-FedAvg', 'fedcorr': 'FedCorr', 'fednoro': 'FedNoRo', 'fedus': 'FedPURE'}

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='../results/timing')
a = ap.parse_args()
rows = {}
for p in glob.glob(os.path.join(a.dir, '*.json')):
    d = json.load(open(p))
    rows[d['algorithm']] = d
print(f"{'Method':10s} {'parallel(s)':>12s} {'sequential(s)':>14s} {'server(s)':>10s} {'eval(s)':>9s} {'bookk.(s)':>9s} {'groups':>7s}")
for m in ORDER:
    if m in rows:
        d = rows[m]
        print(f"{NAMES[m]:10s} {d['parallel_s']:12.1f} {d['sequential_s']:14.1f} {d['server_s']:10.1f} "
              f"{d['eval_s']:9.1f} {d.get('bookkeeping_s', 0):9.1f} {d['n_groups']:7d}")
print("\nLaTeX rows (Training time = parallel wall-clock, s):")
for m in ORDER:
    if m in rows:
        print(f"{NAMES[m]} & {rows[m]['parallel_s']:,.0f} \\\\".replace(',', '{,}'))
