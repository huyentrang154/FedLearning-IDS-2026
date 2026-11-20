#!/usr/bin/env python3
"""Regenerate every mean +- std of Tables 3-5 of the paper from the run logs.

The file results/table_runs.csv lists, for every table entry and seed, the run
directory whose tracking.log gives the seed-level score. This script reads the
score from each log (the "f1-normal" line of the "Best performance metrics"
block; "F1-score" for the centralized runs), checks it against the value stored
in the CSV, and prints the mean and the sample standard deviation (n-1) over
the seeds of every entry.

Run from the repository root or from system/:
    python system/summarize_tables.py            # print all entries
    python system/summarize_tables.py --latex    # also print LaTeX cells
"""
import argparse
import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(".") if Path("results").is_dir() else Path("..")
CSV_PATH = ROOT / "results" / "table_runs.csv"


def read_score(run_dir):
    text = (ROOT / run_dir / "tracking.log").read_text(encoding="utf-8", errors="ignore")
    found = re.findall(r"f1-normal:\s*([0-9.]+)", text)
    if not found:
        found = re.findall(r"F1-score\s*:\s*([0-9.]+)", text)
    if not found:
        raise ValueError(f"no F1 score in {run_dir}/tracking.log")
    return float(found[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="print LaTeX cells")
    args = parser.parse_args()

    entries = OrderedDict()
    mismatches = 0
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["table"], row["setting"], row["dataset"], row["method"])
            entries.setdefault(key, [])
            if not row["run_dir"]:
                entries[key].append((int(row["seed"]), None, row["note"]))
                continue
            score = read_score(row["run_dir"])
            if row["f1"] and abs(score - float(row["f1"])) > 5e-5:
                mismatches += 1
                print(f"MISMATCH {key} seed {row['seed']}: log {score:.4f}, csv {row['f1']}")
            entries[key].append((int(row["seed"]), score, row["note"]))

    print(f"{'table':8} {'setting':12} {'dataset':8} {'method':12} seeds  mean    std")
    for key, seeds in entries.items():
        values = [s for _, s, _ in sorted(seeds) if s is not None]
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else float("nan")
        note = "  (" + "; ".join(sorted({x for _, s, x in seeds if s is None and x})) + ")" if n < len(seeds) else ""
        print(f"{key[0]:8} {key[1]:12} {key[2]:8} {key[3]:12} {n}/{len(seeds)}   {mean:.4f}  {std:.4f}{note}")
        if args.latex:
            print(f"    {mean:.4f} {{\\scriptsize$\\pm${std:.4f}}}")
    print(f"\n{len(entries)} entries, {mismatches} mismatches between logs and {CSV_PATH}")


if __name__ == "__main__":
    main()
