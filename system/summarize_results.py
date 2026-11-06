#!/usr/bin/env python3
"""Summarize tracking.log files for seed_0, seed_1, and seed_2 (or any seeds present).

Run from the repository root (or system/ directory):
    python summarize_results.py

Expected layout:
    results/<dataset>/<data_type>/<algorithm>/<exp_name>/seed_<seed>/tracking.log
"""
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results") if Path("results").is_dir() else Path("../results")
OUTPUT_DIR = RESULTS_DIR / "summary"
VALID_SEEDS = {0, 1, 2, 3, 4}
METRICS = [
    "acc-flipping", "acc-normal",
    "f1-flipping", "f1-normal",
    "recall-flipping", "recall-normal",
    "precision-flipping", "precision-normal",
]


def search_value(pattern, text, default=None):
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    parts = log_path.parts

    # Parse standard directory layout.
    dataset = data_type = algorithm = exp_name = scenario = ""
    seed = None
    try:
        result_idx = parts.index("results")
        tail = list(parts[result_idx + 1:-1])  # remove tracking.log
        if len(tail) >= 4:
            dataset, data_type, algorithm, exp_name = tail[:4]
        if tail and re.fullmatch(r"seed_\d+", tail[-1]):
            seed = int(tail[-1].split("_", 1)[1])
    except ValueError:
        pass

    # Metadata in the log can override/fill path metadata.
    dataset = search_value(r"^\s*Dataset\s*:\s*([^|\r\n]+)", text, dataset) or dataset
    data_type = search_value(r"^\s*Data\s*type\s*:\s*([^|\r\n]+)", text, data_type) or data_type
    algorithm = search_value(r"^\s*Algorithm\s*:\s*([^|\r\n]+)", text, algorithm) or algorithm
    scenario = search_value(r"^\s*Scenario\s*:\s*([^|\r\n]+)", text, scenario) or scenario
    if seed is None:
        seed_text = search_value(r"\bSeed\s*:\s*(\d+)", text)
        if seed_text is not None:
            seed = int(seed_text)

    noise_type = search_value(r"Noise\s*type\s*:\s*([^|\r\n]+)", text, "") or ""
    num_poisoned_workers = search_value(
        r"\bnum_of_poisoned_workers\s*:\s*([0-9.eE+-]+)",
        text, "",
    ) or ""
    noise_ratio = search_value(r"\bratio\s*:\s*([0-9.eE+-]+)", text, "") or ""
    noise_parameter = search_value(
        r"\b(?:num_of_poisoned_workers|noise\s*parameter|parameter|epsilon|rc|rs)\s*:\s*([0-9.eE+-]+)",
        text, num_poisoned_workers,
    ) or ""

    # Extract the last "Best performance metrics" block.
    matches = list(re.finditer(r"Best\s+performance\s+metrics", text, re.I))
    if not matches:
        print(f"[SKIP] No Best performance metrics block: {log_path}")
        return None

    start = matches[-1].end()
    remaining = text[start:]
    endings = [
        m.start() for pat in (
            r"\n\s*-{5,}\s*Round\s+number\s*:",
            r"\n\s*Round\s+number\s*:",
            r"\n\s*Best\s+performance\s+metrics",
        )
        for m in re.finditer(pat, remaining, re.I)
    ]
    block = remaining[:min(endings)] if endings else remaining

    values = {}
    for metric in METRICS:
        pattern = (
            r"^\s*" + re.escape(metric)
            + r"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        )
        value = search_value(pattern, block)
        values[metric] = float(value) if value is not None else None

    if all(v is None for v in values.values()):
        print(f"[SKIP] Metric labels not recognized in {log_path}")
        return None

    return {
        "dataset": dataset,
        "data_type": data_type,
        "algorithm": algorithm,
        "scenario": scenario,
        "exp_name": exp_name,
        "seed": seed,
        "noise_type": noise_type,
        "num_of_poisoned_workers": num_poisoned_workers,
        "noise_ratio": noise_ratio,
        "noise_parameter": noise_parameter,
        **values,
        "log_file": str(log_path),
    }


def main():
    if not RESULTS_DIR.is_dir():
        raise SystemExit("Không tìm thấy thư mục results/. Hãy kiểm tra lại đường dẫn.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for log_path in RESULTS_DIR.rglob("tracking.log"):
        if OUTPUT_DIR in log_path.parents:
            continue
        try:
            record = parse_log(log_path)
        except OSError as exc:
            print(f"[ERROR] {log_path}: {exc}")
            continue
        if record is None:
            continue
        if record["seed"] not in VALID_SEEDS:
            print(f"[SKIP] Seed không hợp lệ: {log_path} (seed={record['seed']})")
            continue
        records.append(record)

    if not records:
        raise SystemExit(
            "Không tìm thấy log hợp lệ. Hãy kiểm tra cấu trúc thư mục, seed "
            "và tên các metric trong tracking.log."
        )

    details = pd.DataFrame(records)
    group_cols = [
        "dataset", "data_type", "algorithm", "scenario",
        "noise_type", "num_of_poisoned_workers", "noise_ratio",
    ]

    duplicate_cols = group_cols + ["seed"]
    duplicate_mask = details.duplicated(duplicate_cols, keep=False)
    if duplicate_mask.any():
        print(
            "\n[CẢNH BÁO] Có nhiều log cùng cấu hình và seed. "
            "Các log đó đều được tính vào tổng hợp; hãy kiểm tra exp_name:"
        )
        print(details.loc[duplicate_mask, duplicate_cols + ["exp_name", "log_file"]].to_string(index=False))

    detail_columns = [
        "dataset", "data_type", "algorithm", "scenario", "exp_name", "seed",
        "noise_type", "num_of_poisoned_workers", "noise_ratio", "noise_parameter",
        *METRICS, "log_file",
    ]
    details = details.reindex(columns=detail_columns)
    details.to_csv(OUTPUT_DIR / "summary_per_seed.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for keys, group in details.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["num_logs"] = len(group)
        row["seeds_present"] = ",".join(
            str(int(s)) for s in sorted(group["seed"].dropna().unique())
        )
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = vals.mean() if len(vals) else None
            row[f"{metric}_std"] = vals.std(ddof=1) if len(vals) >= 2 else None
            row[f"{metric}_count"] = int(vals.count())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    excel_path = OUTPUT_DIR / "summary_mean_std.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Mean_Std", index=False)
            details.to_excel(writer, sheet_name="Per_Log", index=False)
    except Exception as exc:
        print(f"[WARN] Không xuất được file Excel ({exc}), đã lưu đầy đủ CSV.")

    print(f"\nĐã đọc {len(details)} log hợp lệ.")
    print(f"Chi tiết từng log: {OUTPUT_DIR / 'summary_per_seed.csv'}")
    print(f"Tổng hợp CSV:      {OUTPUT_DIR / 'summary_mean_std.csv'}")
    print(f"Tổng hợp Excel:    {excel_path}")


if __name__ == "__main__":
    main()
