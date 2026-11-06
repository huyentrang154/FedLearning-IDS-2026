"""Helpers for saving model checkpoints from FL servers.

Output files (inside ``server.result_path``):
    models{suffix}.pkl       {round: [global, client_0, ..., client_{N-1}]}
                             nn.Module deep copies on CPU (same layout as before,
                             compatible with predict.ipynb: model_list[r][idx]).
                             global = model after aggregation of round r,
                             client_i = local model of client i at the end of round r
                             (clients not selected in round r keep their previous state).
    final_global{suffix}.pt  state_dict of the server model after training.
    best_model{suffix}.pt    {'round', 'score', 'metric', 'global', 'clients'} state_dicts
                             of the round with the best evaluation score.

``suffix`` is empty for the first run (times == 0) and ``_run{times}`` otherwise, so
repeated runs with the same seed (main.py --times) no longer overwrite each other.
"""
import copy
import os
import pickle
import time

import numpy as np
import torch
from utils.data_utils import resolve_data_type


def normalize_noise_type(noise_type):
    """Chuẩn hóa tên loại nhiễu (hỗ trợ cả viết thường/sai chính tả phổ biến)."""
    if noise_type is None:
        return None
    nt = str(noise_type).strip()
    nt_lower = nt.lower()
    if nt_lower in ("", "none", "null", "clean"):
        return None
    if nt_lower in ("label_flipping", "labelflipping", "flipping"):
        return "label_flipping"
    if nt_lower in ("asymmetric", "asymetric", "asym"):
        return "Asymmetric"
    if nt_lower in ("symmetric", "symetric", "sym"):
        return "Symmetric"
    if nt_lower in ("gaussian_noise", "gaussian"):
        return "gaussian_noise"
    return nt


def select_poisoned_workers(num_clients, noise_type, num_of_poisoned_workers):
    """Chọn danh sách client bị nhiễu cố định (không random) để thống nhất giữa các thuật toán và các seed.
    Ví dụ với 20 clients:
      - num_of_poisoned_workers = 0.3 -> [0, 1, 2, 3, 4, 5]
      - num_of_poisoned_workers = 0.5 -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    norm_noise = normalize_noise_type(noise_type)
    if (
        norm_noise in ("label_flipping", "gaussian_noise", "Asymmetric", "Symmetric")
        and num_of_poisoned_workers is not None
        and num_of_poisoned_workers > 0
    ):
        num_poisoned = int(num_clients * num_of_poisoned_workers)
        if num_poisoned > 0:
            return list(range(num_poisoned))
    return []


def format_scenario_name(data_type, noise_type, num_of_poisoned_workers, poisoned_sample_ratio):
    """Tạo tên kịch bản tường minh từ cấu hình thực nghiệm."""
    norm_noise = normalize_noise_type(noise_type)
    if (
        norm_noise is None
        or num_of_poisoned_workers is None
        or num_of_poisoned_workers <= 0
        or poisoned_sample_ratio is None
        or poisoned_sample_ratio <= 0
    ):
        return f"{data_type}_clean"
    return f"{data_type}_{norm_noise}_npw{num_of_poisoned_workers}_psr{poisoned_sample_ratio}"


def setup_experiment_logging(server, args, extra_lines=None):
    """Thiết lập đường dẫn lưu kết quả và ghi header tường minh vào tracking.log."""
    server.data_type = resolve_data_type(getattr(args, "data_type", "noniid"), server.num_clients)
    server.noise_type = normalize_noise_type(getattr(args, "noise_type", None))
    if server.noise_type is None or getattr(server, "num_of_poisoned_workers", 0) <= 0:
        server.num_of_poisoned_workers = 0.0
        server.poisoned_sample_ratio = 0.0

    scenario_name = format_scenario_name(
        server.data_type,
        server.noise_type,
        server.num_of_poisoned_workers,
        server.poisoned_sample_ratio,
    )
    server.scenario_name = scenario_name

    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    raw_exp_name = str(getattr(args, "exp_name", "") or "").strip()
    if not raw_exp_name:
        exp_folder = f"{timestamp_str}_{scenario_name}"
    elif scenario_name not in raw_exp_name:
        exp_folder = f"{raw_exp_name}_{scenario_name}"
    else:
        exp_folder = raw_exp_name

    server.exp_name = exp_folder
    server.result_path = os.path.join(
        "..",
        "results",
        server.dataset,
        server.data_type,
        server.algo,
        server.exp_name,
        f"seed_{server.seed}",
    )
    os.makedirs(server.result_path, exist_ok=True)

    server.log_file = os.path.join(server.result_path, "tracking.log")
    noise_label = server.noise_type if server.noise_type is not None else "None"
    header_lines = [
        "=" * 60 + "\n",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Experiment: {server.exp_name}\n",
        f"Scenario: {server.scenario_name}\n",
        f"Algorithm: {server.algo}\n",
        f"Dataset: {server.dataset}\n",
        f"Data type: {server.data_type}\n",
        f"Seed: {server.seed}\n",
        f"Num clients: {server.num_clients} | Join ratio: {getattr(server, 'join_ratio', 1.0)}\n",
        f"Global rounds: {getattr(args, 'global_rounds', 'N/A')} | Local steps: {getattr(args, 'local_steps', 'N/A')} | Batch size: {getattr(args, 'batch_size', 'N/A')} | LR: {getattr(args, 'local_learning_rate', 'N/A')}\n",
        f"Noise type: {noise_label} | num_of_poisoned_workers: {server.num_of_poisoned_workers} | ratio: {server.poisoned_sample_ratio}\n",
        f"Poisoned Worker IDs: {getattr(server, 'poisoned_worker_ids', [])}\n",
        f"Result path: {server.result_path}\n",
    ]
    if extra_lines:
        for line in extra_lines:
            header_lines.append(line if line.endswith("\n") else line + "\n")
    header_lines.append("=" * 60 + "\n")

    with open(server.log_file, "w", encoding="utf-8") as f:
        f.writelines(header_lines)


def write_final_summary(server):
    """In và ghi khối tổng kết chuẩn hóa vào cuối tracking.log."""
    def _safe_max(vals):
        return max(vals) if vals else 0.0

    acc_f = _safe_max(getattr(server, "rs_test_acc_flipping", []))
    acc_n = _safe_max(getattr(server, "rs_test_acc_normal", []))
    f1_f = _safe_max(getattr(server, "rs_test_f1_flipping", []))
    f1_n = _safe_max(getattr(server, "rs_test_f1_normal", []))
    rec_f = _safe_max(getattr(server, "rs_test_recall_flipping", []))
    rec_n = _safe_max(getattr(server, "rs_test_recall_normal", []))
    pre_f = _safe_max(getattr(server, "rs_test_precision_flipping", []))
    pre_n = _safe_max(getattr(server, "rs_test_precision_normal", []))
    avg_t = mean_round_time(getattr(server, "budget", []))
    noise_label = server.noise_type if getattr(server, "noise_type", None) is not None else "None"

    summary_lines = [
        "\n" + "=" * 60 + "\n",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Experiment: {getattr(server, 'exp_name', '')}\n",
        f"Scenario: {getattr(server, 'scenario_name', '')}\n",
        f"Algorithm: {server.algo} | Dataset: {server.dataset} | Data type: {getattr(server, 'data_type', '')} | Seed: {getattr(server, 'seed', 0)}\n",
        f"Noise type: {noise_label} | num_of_poisoned_workers: {getattr(server, 'num_of_poisoned_workers', 0.0)} | ratio: {getattr(server, 'poisoned_sample_ratio', 0.0)}\n",
        f"Poisoned Worker IDs: {getattr(server, 'poisoned_worker_ids', [])}\n",
        "\nBest performance metrics\n",
        f"\tacc-flipping: {acc_f:.4f}\n",
        f"\tacc-normal: {acc_n:.4f}\n",
        f"\tf1-flipping: {f1_f:.4f}\n",
        f"\tf1-normal: {f1_n:.4f}\n",
        f"\trecall-flipping: {rec_f:.4f}\n",
        f"\trecall-normal: {rec_n:.4f}\n",
        f"\tprecision-flipping: {pre_f:.4f}\n",
        f"\tprecision-normal: {pre_n:.4f}\n",
        f"\tAvg Time/Round: {avg_t:.4f} sec\n",
        "=" * 60 + "\n",
    ]

    for line in summary_lines:
        if line.strip():
            print(line.rstrip())
        with open(server.log_file, "a", encoding="utf-8") as f:
            f.write(line)


def run_suffix(times):
    return "" if not times else f"_run{times}"


def cpu_copy(model):
    """Independent deep copy of ``model`` on CPU (never a reference to the live model)."""
    if model is None:
        return None
    return copy.deepcopy(model).cpu()


def snapshot(model):
    """Independent CPU copy of ``model.state_dict()``."""
    if model is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def round_models(global_model, clients):
    """[global, client_0, ..., client_{N-1}] as CPU deep copies."""
    return [cpu_copy(global_model)] + [cpu_copy(c.model) for c in clients]


def _atomic_write(path, write_fn):
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def save_torch(obj, path):
    _atomic_write(path, lambda p: torch.save(obj, p))


def save_pickle(obj, path):
    def _w(p):
        with open(p, "wb") as f:
            pickle.dump(obj, f)
    _atomic_write(path, _w)


def mean_round_time(budget):
    """Average round time excluding round 0; safe when there are < 2 rounds."""
    vals = budget[1:] if len(budget) > 1 else budget
    return sum(vals) / len(vals) if len(vals) > 0 else 0.0


def track_best(server, epoch, score, global_model, clients, metric="f1-normal"):
    """Save a checkpoint whenever ``score`` improves (written to disk immediately)."""
    if score is None:
        return
    best = getattr(server, "best_score", None)
    if best is not None and score <= best:
        return
    server.best_score = score
    server.best_round = epoch
    ckpt = {
        "round": epoch,
        "score": score,
        "metric": metric,
        "global": snapshot(global_model),
        "clients": {c.id: snapshot(c.model) for c in clients},
    }
    save_torch(ckpt, os.path.join(server.result_path,
                                  f"best_model{run_suffix(getattr(server, 'times', 0))}.pt"))


def save_model_artifacts(server, final_model, model_list=None):
    """Write models.pkl (if model_list is given) and final_global.pt."""
    suffix = run_suffix(getattr(server, "times", 0))
    if model_list is not None:
        save_pickle(model_list, os.path.join(server.result_path, f"models{suffix}.pkl"))
    if final_model is not None:
        save_torch(snapshot(final_model),
                   os.path.join(server.result_path, f"final_global{suffix}.pt"))
    print(f"Models saved to {server.result_path}")

