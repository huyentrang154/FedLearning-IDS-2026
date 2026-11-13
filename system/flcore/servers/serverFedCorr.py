# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import os
# import pickle
# import time
# from sklearn.mixture import GaussianMixture
# from scipy.spatial.distance import cdist

# from flcore.clients.clientFedCorr import ClientFedCorr
# from utils.data_utils import read_client_data


# # ------------------------------------------------------------------
# # LID estimator (verbatim from FedCorr util/util.py)
# # ------------------------------------------------------------------

# def _lid_term(outputs, k=20):
#     """Local Intrinsic Dimensionality for each sample in the output matrix.

#     Computes pairwise distances in the softmax-probability space, then
#     applies the MLE estimator:  LID_i = -k / sum_j log(d_j / d_k)
#     where d_1 ≤ … ≤ d_k are the k nearest-neighbour distances.
#     Matches Algorithm 1 of the FedCorr paper (arXiv:2204.04677).
#     """
#     eps = 1e-6
#     X = np.asarray(outputs, dtype=np.float32)
#     k = min(k, len(X) - 1)
#     if k <= 0:
#         return np.zeros(len(X))

#     def f(v):
#         d_max = v[-1] + eps
#         log_ratios = np.log(np.maximum(v / d_max, eps))
#         denom = np.sum(log_ratios) + eps
#         return -k / denom

#     distances = cdist(X, X)
#     sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
#     m, n = sort_indices.shape
#     idx = list(np.ogrid[:m, :n])
#     idx[1] = sort_indices
#     distances_ = distances[tuple(idx)]
#     lids = np.apply_along_axis(f, axis=1, arr=distances_)
#     # Guard against any residual NaN/inf
#     lids = np.where(np.isfinite(lids), lids, 0.0)
#     return lids


# # ------------------------------------------------------------------
# # FedAvg helper
# # ------------------------------------------------------------------

# def _fedavg(w_list, sizes):
#     total = float(sum(sizes))
#     w_avg = copy.deepcopy(w_list[0])
#     for k in w_avg:
#         w_avg[k] = torch.zeros_like(w_avg[k], dtype=torch.float32)
#     for w, s in zip(w_list, sizes):
#         for k in w_avg:
#             w_avg[k] += w[k].float() * (s / total)
#     return w_avg


# # ------------------------------------------------------------------
# # FedCorr Server
# # ------------------------------------------------------------------

# class FedCorr(object):
#     """Three-stage noise-robust federated learning.

#     Stage 1 – Noise Detection  (iteration1 outer iterations):
#         Every iteration all clients train locally with a proximal term
#         (beta * mu * ||w - w_g||, where mu is the client's estimated noise
#         level from the previous iteration).  After local training, softmax
#         outputs are fed into the LID estimator; per-client mean LID is
#         accumulated across iterations.  At each iteration end a 2-component
#         GMM partitions clients into clean / noisy by their cumulative LID
#         (lower LID ⟹ cleaner data).  A second GMM on per-sample losses
#         within each noisy client estimates that client's noise level.
#         Optionally, mislabelled samples are relabelled using high-confidence
#         global-model predictions (correction step).

#     Stage 2 – Fine-tuning  (rounds1 rounds, optional):
#         Only clients whose estimated noise level ≤ clean_set_thres
#         participate.  Optionally performs a second relabelling pass.

#     Stage 3 – Final training  (rounds2 rounds):
#         Standard FedAvg on all clients (no proximal term, no correction).

#     Reference: Xu et al., "FedCorr: Multi-Stage Federated Learning for
#     Label Noise Robustness", CVPR 2022.  arXiv:2204.04677.
#     """

#     def __init__(self, args, times):
#         self.device = args.device
#         self.dataset = args.dataset
#         self.num_clients = args.num_clients
#         self.join_ratio = args.join_ratio
#         self.global_model = copy.deepcopy(args.model)
#         self.num_classes = args.num_classes
#         self.algo = args.algorithm
#         self.times = times
#         self.data_type = args.data_type
#         self.noise_type = args.noise_type
#         self.num_of_poisoned_workers = args.num_of_poisoned_workers
#         self.poisoned_sample_ratio = args.poisoned_sample_ratio
#         self.attack_noise_std = args.attack_noise_std
#         self.poisoned_worker_ids = []
#         self.initialize_poisoned_workers()

#         # ── FedCorr hyper-parameters ──────────────────────────────────
#         self.iteration1 = args.iteration1          # Stage 1 outer iterations
#         self.rounds1 = args.rounds1                # Stage 2 fine-tuning rounds
#         self.rounds2 = args.rounds2                # Stage 3 final rounds
#         self.frac1 = args.frac1                    # client fraction per sub-round (Stage 1)
#         self.frac2 = args.frac2                    # client fraction (Stage 2 / 3)
#         self.beta = args.beta_fedcorr              # proximal term coefficient
#         self.relabel_ratio = args.relabel_ratio    # fraction of noisy samples to relabel
#         self.confidence_thres = args.confidence_thres  # min confidence to accept relabelling
#         self.clean_set_thres = args.clean_set_thres    # max noise level to be "clean" in Stage 2
#         self.correction = args.correction          # enable label correction
#         self.fine_tuning = args.fine_tuning        # enable Stage 2 fine-tuning

#         # ── Metrics ───────────────────────────────────────────────────
#         self.rs_train_loss = []
#         self.rs_test_acc = []
#         self.rs_test_f1 = []
#         self.rs_test_precision = []
#         self.rs_test_recall = []
#         self.train_losses = {i: [] for i in range(self.num_clients)}

#         # ── Client partition state ────────────────────────────────────
#         self.noisy_clients = []
#         self.clean_clients = list(range(self.num_clients))
#         self.estimated_noisy_level = np.zeros(self.num_clients)

#         # ── Create clients ────────────────────────────────────────────
#         self.clients = []
#         self.set_clients(args, ClientFedCorr)

#         print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
#         print(f"Algorithm: {self.algo} | Dataset: {self.dataset}")
#         print(f"FedCorr | iteration1={self.iteration1}, rounds1={self.rounds1}, "
#               f"rounds2={self.rounds2}")
#         print(f"         beta={self.beta}, relabel_ratio={self.relabel_ratio}, "
#               f"confidence_thres={self.confidence_thres}")
#         print(f"         clean_set_thres={self.clean_set_thres}, "
#               f"correction={self.correction}, fine_tuning={self.fine_tuning}")
#         print("Finished creating server and clients.")

#         # ── Result path ───────────────────────────────────────────────
#         self.result_path = os.path.join(
#             '..', 'results', self.dataset, self.data_type,
#             self.algo, time.strftime("%Y%m%d-%H%M%S")
#         )
#         os.makedirs(self.result_path, exist_ok=True)
#         self.log_file = f"{self.result_path}/tracking.log"
#         with open(self.log_file, 'w') as f:
#             f.write(f"Algorithm: {self.algo} | Dataset: {self.dataset}\n")
#             f.write(f"iteration1={self.iteration1}, rounds1={self.rounds1}, "
#                     f"rounds2={self.rounds2}, beta={self.beta}\n")
#         self.budget = []

#     def initialize_poisoned_workers(self):
#             """
#             Xác định các client bị tấn công.
#             """
    
#             # Các kiểu nhiễu áp dụng lên toàn bộ client
#             if self.noise_type in ["Asymmetric", "Symmetric"]:
#                 self.poisoned_worker_ids = list(range(self.num_clients))
    
#             # Label flipping / Gaussian chỉ áp dụng theo tỷ lệ
#             elif self.noise_type in ["label_flipping", "gaussian_noise"] \
#                     and self.num_of_poisoned_workers > 0:
    
#                 num_poisoned = int(self.num_clients * self.num_of_poisoned_workers)
    
#                 self.poisoned_worker_ids = np.random.choice(
#                     range(self.num_clients),
#                     size=num_poisoned,
#                     replace=False
#                 ).tolist()
    
#             else:
#                 self.poisoned_worker_ids = []
    
#             print(f"Noise Type: {self.noise_type}")
#             print(f"Poisoned Worker IDs: {self.poisoned_worker_ids}")
    
#     # ==================================================================
#     # Main training loop
#     # ==================================================================

#     def train(self):

#         # ==============================================================
#         # Stage 1 – Noise Detection
#         # ==============================================================
#         print("\n===== Stage 1: Noise Detection =====")
#         self._log("===== Stage 1: Noise Detection =====\n")

#         LID_accumulative = np.zeros(self.num_clients)
#         # Per-client per-sample loss accumulator (list length = train_samples)
#         loss_accumulative = {
#             i: np.zeros(len(self.clients[i].train_data))
#             for i in range(self.num_clients)
#         }
#         estimated_noisy_level = np.zeros(self.num_clients)

#         for iteration in range(self.iteration1):
#             s_t = time.time()
#             print(f"\n--- Stage 1  iter {iteration + 1}/{self.iteration1} ---")
#             self._log(f"--- Stage 1 iter {iteration + 1}/{self.iteration1} ---\n")

#             # mu_list: 0 on first iter, estimated noise level after
#             mu_list = estimated_noisy_level.copy()

#             LID_client = np.zeros(self.num_clients)
#             loss_iter = {
#                 i: np.zeros(len(self.clients[i].train_data))
#                 for i in range(self.num_clients)
#             }

#             # ── Sub-rounds: cover all clients in groups of frac1 ──────
#             # Mirrors the original: for _ in range(int(1/frac1)): sample frac1 clients
#             n_sub = max(1, round(1.0 / self.frac1))
#             n_per_sub = max(1, int(self.num_clients * self.frac1))
#             perm = np.random.permutation(self.num_clients)
#             # Build non-overlapping groups; remainder goes into last group
#             groups = []
#             for g in range(n_sub):
#                 start, end = g * n_per_sub, (g + 1) * n_per_sub
#                 groups.append(perm[start:end].tolist())
#             if n_sub * n_per_sub < self.num_clients:
#                 groups[-1] += perm[n_sub * n_per_sub:].tolist()

#             for group in groups:
#                 w_locals, w_sizes = [], []
#                 for idx in group:
#                     client = self.clients[idx]
#                     # Broadcast current global model
#                     client.set_parameters(self.global_model)
#                     # Supply proximal reference and noise estimate
#                     client.global_model = copy.deepcopy(self.global_model).to(self.device)
#                     client.mu = float(mu_list[idx])
#                     # Local training
#                     client.train(beta=self.beta)

#                     # Compute LID from LOCAL model outputs (as in original code)
#                     outputs, losses = client.get_outputs_and_loss(model=client.model)
#                     lids = _lid_term(outputs)
#                     LID_client[idx] = float(np.mean(lids))
#                     loss_iter[idx] = losses

#                     w_locals.append(client.model.state_dict())
#                     w_sizes.append(client.train_samples)

#                 # FedAvg after each sub-group → update global model
#                 w_glob = _fedavg(w_locals, w_sizes)
#                 self.global_model.load_state_dict(w_glob)

#             # ── Accumulate LID and losses ─────────────────────────────
#             LID_accumulative += LID_client
#             for i in range(self.num_clients):
#                 loss_accumulative[i] += loss_iter[i]

#             # ── GMM on cumulative LID to identify clean / noisy clients
#             lid_for_gmm = np.nan_to_num(LID_accumulative, nan=0.0, posinf=0.0, neginf=0.0)
#             gmm_lid = GaussianMixture(n_components=2, random_state=0, max_iter=200).fit(
#                 lid_for_gmm.reshape(-1, 1)
#             )
#             lid_pred = gmm_lid.predict(lid_for_gmm.reshape(-1, 1))
#             # Component with LOWER mean → cleaner clients (lower LID = cleaner)
#             clean_comp = int(np.argmin(gmm_lid.means_[:, 0]))
#             self.clean_clients = [i for i in range(self.num_clients) if lid_pred[i] == clean_comp]
#             self.noisy_clients = [i for i in range(self.num_clients) if lid_pred[i] != clean_comp]

#             # ── Estimate per-client noise level via loss-based GMM ────
#             estimated_noisy_level = np.zeros(self.num_clients)
#             for cid in self.noisy_clients:
#                 loss_c = loss_accumulative[cid]
#                 if len(loss_c) < 2:
#                     continue
#                 gmm_loss = GaussianMixture(n_components=2, random_state=0, max_iter=200).fit(
#                     loss_c.reshape(-1, 1)
#                 )
#                 loss_pred = gmm_loss.predict(loss_c.reshape(-1, 1))
#                 clean_comp_loss = int(np.argmin(gmm_loss.means_[:, 0]))
#                 n_noisy_samples = int(np.sum(loss_pred != clean_comp_loss))
#                 estimated_noisy_level[cid] = n_noisy_samples / len(loss_c)

#             print(f"  LID accum:     {np.round(LID_accumulative, 2).tolist()}")
#             print(f"  Clean clients: {self.clean_clients}")
#             print(f"  Noisy clients: {self.noisy_clients}")
#             print(f"  Noise levels:  {np.round(estimated_noisy_level, 3).tolist()}")
#             self._log(
#                 f"Clean: {self.clean_clients}\n"
#                 f"Noisy: {self.noisy_clients}\n"
#                 f"Noise levels: {np.round(estimated_noisy_level, 3).tolist()}\n"
#             )

#             # ── Optional label correction ─────────────────────────────
#             if self.correction:
#                 self._correct_labels(estimated_noisy_level, loss_accumulative)

#             self._evaluate_and_log(f"[Stage 1 iter {iteration + 1}]")
#             self.budget.append(time.time() - s_t)

#         # Save final Stage 1 estimates for Stage 2 selection
#         self.estimated_noisy_level = estimated_noisy_level

#         # ==============================================================
#         # Stage 2 – Fine-tuning on clean clients (optional)
#         # ==============================================================
#         if self.fine_tuning:
#             print("\n===== Stage 2: Fine-tuning (clean clients only) =====")
#             self._log("===== Stage 2: Fine-tuning =====\n")

#             # Clients whose estimated noise level is below threshold
#             selected_clean = [
#                 i for i in range(self.num_clients)
#                 if estimated_noisy_level[i] <= self.clean_set_thres
#             ]
#             if not selected_clean:
#                 selected_clean = list(self.clean_clients)

#             m2 = max(1, min(int(self.frac2 * self.num_clients), len(selected_clean)))

#             for rnd in range(self.rounds1):
#                 s_t = time.time()
#                 chosen = list(np.random.choice(selected_clean, m2, replace=False))
#                 w_locals, w_sizes = [], []
#                 for idx in chosen:
#                     client = self.clients[idx]
#                     client.set_parameters(self.global_model)
#                     client.global_model = None
#                     client.mu = 0.0
#                     client.train(beta=0.0)
#                     w_locals.append(client.model.state_dict())
#                     w_sizes.append(client.train_samples)
#                 w_glob = _fedavg(w_locals, w_sizes)
#                 self.global_model.load_state_dict(w_glob)

#                 # Second correction pass: relabel noisy clients with fine-tuned model
#                 if self.correction:
#                     self._correct_labels_stage2()

#                 self._evaluate_and_log(f"[Stage 2 rnd {rnd + 1}]")
#                 self.budget.append(time.time() - s_t)

#         # ==============================================================
#         # Stage 3 – Final training (all clients, standard FedAvg)
#         # ==============================================================
#         print("\n===== Stage 3: Final Training (all clients) =====")
#         self._log("===== Stage 3: Final Training =====\n")

#         m3 = max(1, int(self.frac2 * self.num_clients))

#         for rnd in range(self.rounds2):
#             s_t = time.time()
#             chosen = list(
#                 np.random.choice(self.num_clients, m3, replace=False)
#             )
#             w_locals, w_sizes = [], []
#             for idx in chosen:
#                 client = self.clients[idx]
#                 client.set_parameters(self.global_model)
#                 client.global_model = None
#                 client.mu = 0.0
#                 client.train(beta=0.0)
#                 w_locals.append(client.model.state_dict())
#                 w_sizes.append(client.train_samples)
#             w_glob = _fedavg(w_locals, w_sizes)
#             self.global_model.load_state_dict(w_glob)

#             self._evaluate_and_log(f"[Stage 3 rnd {rnd + 1}]")
#             self.budget.append(time.time() - s_t)

#         # Final summary
#         if self.rs_test_acc:
#             print(f"\nBest acc={max(self.rs_test_acc):.4f}  "
#                   f"f1={max(self.rs_test_f1):.4f}  "
#                   f"avg time/round={np.mean(self.budget):.2f}s")
#         self.save_results()

#     # ==================================================================
#     # Label correction helpers
#     # ==================================================================

#     def _correct_labels(self, estimated_noisy_level, loss_accumulative):
#         """Stage 1 correction.

#         For each noisy client:
#           1. Rank samples by descending accumulated loss (most noisy first).
#           2. Keep only the top (n * noise_level * relabel_ratio) candidates.
#           3. Filter by global-model confidence > confidence_thres.
#           4. Relabel survivors to the global model's argmax prediction.
#         """
#         for idx in self.noisy_clients:
#             client = self.clients[idx]
#             n = len(client.train_data)
#             if n == 0:
#                 continue

#             # Global-model softmax outputs for all training samples
#             outputs, _ = client.get_outputs_and_loss(
#                 model=copy.deepcopy(self.global_model).to(self.device)
#             )

#             loss_c = loss_accumulative[idx]
#             n_relabel = int(n * estimated_noisy_level[idx] * self.relabel_ratio)
#             top_loss_idx = np.argsort(-loss_c)[:n_relabel]

#             max_conf = np.max(outputs, axis=1)
#             relabel_idx = [
#                 i for i in top_loss_idx if max_conf[i] > self.confidence_thres
#             ]
#             if relabel_idx:
#                 new_labels = np.argmax(outputs[relabel_idx], axis=1)
#                 client.relabel(relabel_idx, new_labels)
#                 self._log(
#                     f"  [Correction] client {idx}: "
#                     f"relabelled {len(relabel_idx)}/{n} samples\n"
#                 )

#     def _correct_labels_stage2(self):
#         """Stage 2 correction.

#         Relabel any sample in noisy clients where the fine-tuned global model
#         predicts with confidence above confidence_thres.
#         """
#         for idx in self.noisy_clients:
#             client = self.clients[idx]
#             if not client.train_data:
#                 continue
#             outputs, _ = client.get_outputs_and_loss(
#                 model=copy.deepcopy(self.global_model).to(self.device)
#             )
#             max_conf = np.max(outputs, axis=1)
#             relabel_idx = list(np.where(max_conf > self.confidence_thres)[0])
#             if relabel_idx:
#                 new_labels = np.argmax(outputs[relabel_idx], axis=1)
#                 client.relabel(relabel_idx, new_labels)

#     # ==================================================================
#     # Standard FL plumbing
#     # ==================================================================

#     def set_clients(self, args, clientObj):
#         for i in range(self.num_clients):
#             train_data = read_client_data(
#                 self.dataset, i, self.num_clients, data_type='train',type=self.data_type
#             )
#             test_data = read_client_data(
#                 self.dataset, i, self.num_clients, data_type='test',type=self.data_type
#             )
#             client = clientObj(
#                 args=args,
#                 id=i,
#                 poisoned_worker_ids=self.poisoned_worker_ids,
#                 train_samples=len(train_data),
#                 test_samples=len(test_data),
#             )
#             self.clients.append(client)

#     def _evaluate_and_log(self, tag=""):
#         acc_w, f1_w, pre_w, rec_w = [], [], [], []
#         n_total = 0
#         loss_all = []

#         for c in self.clients:
#             acc, pre, rec, f1, ns = c.test_metrics()
#             acc_w.append(acc * ns)
#             f1_w.append(f1 * ns)
#             pre_w.append(pre * ns)
#             rec_w.append(rec * ns)
#             n_total += ns
#             tl = c.train_metrics()
#             if tl:
#                 loss_all.extend(tl)
#                 self.train_losses[c.id].append(float(np.mean(tl)))

#         avg_acc = sum(acc_w) / n_total
#         avg_f1 = sum(f1_w) / n_total
#         avg_pre = sum(pre_w) / n_total
#         avg_rec = sum(rec_w) / n_total
#         avg_loss = float(np.mean(loss_all)) if loss_all else 0.0

#         self.rs_test_acc.append(avg_acc)
#         self.rs_test_f1.append(avg_f1)
#         self.rs_test_precision.append(avg_pre)
#         self.rs_test_recall.append(avg_rec)
#         self.rs_train_loss.append(avg_loss)

#         msg = (f"{tag}  acc={avg_acc:.4f}  f1={avg_f1:.4f}  "
#                f"prec={avg_pre:.4f}  rec={avg_rec:.4f}  loss={avg_loss:.4f}")
#         print(msg)
#         self._log(msg + "\n")

#     def _log(self, msg):
#         with open(self.log_file, 'a') as f:
#             f.write(msg)

#     def save_results(self):
#         results = {
#             'acc': self.rs_test_acc,
#             'f1': self.rs_test_f1,
#             'recall': self.rs_test_recall,
#             'precision': self.rs_test_precision,
#             'loss': self.train_losses,
#             'noisy_clients': self.noisy_clients,
#             'clean_clients': self.clean_clients,
#             'estimated_noisy_level': self.estimated_noisy_level.tolist(),
#         }
#         with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
#             pickle.dump(results, f)
import copy
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import time
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

from flcore.clients.clientFedCorr import ClientFedCorr
from utils.data_utils import read_client_data, resolve_data_type
from flcore.servers.model_saver import (round_models, track_best, save_model_artifacts,
                                        save_pickle, mean_round_time, run_suffix,
                                        normalize_noise_type, select_poisoned_workers,
                                        setup_experiment_logging, write_final_summary)


# ------------------------------------------------------------------
# LID estimator (verbatim from FedCorr util/util.py)
# ------------------------------------------------------------------

def _lid_term(outputs, k=20):
    """Local Intrinsic Dimensionality for each sample in the output matrix.

    Computes pairwise distances in the softmax-probability space, then
    applies the MLE estimator:  LID_i = -k / sum_j log(d_j / d_k)
    where d_1 ≤ … ≤ d_k are the k nearest-neighbour distances.
    Matches Algorithm 1 of the FedCorr paper (arXiv:2204.04677).
    """
    eps = 1e-6
    X = np.asarray(outputs, dtype=np.float32)
    k = min(k, len(X) - 1)
    if k <= 0:
        return np.zeros(len(X))

    # original: f = lambda v: - k / (np.sum(np.log(v / (v[-1] + eps))) + eps)
    # (a zero distance gives log(0) = -inf and hence LID = 0, as in the original)
    def f(v):
        with np.errstate(divide='ignore'):
            return -k / (np.sum(np.log(v / (v[-1] + eps))) + eps)

    # k nearest-neighbour distances, computed in row blocks so memory stays
    # O(block * N) instead of the full N x N matrix (which needs tens of GB on
    # large clients, e.g. UNSW-NB15). Same values as sorting the full row and
    # taking positions 1..k (position 0 is the point itself).
    # The block height is chosen from N so that each float64 block stays ~64 MB
    # (np.partition makes one more copy). A fixed 1024-row block needed ~6 GB per
    # job on the 358k-sample CIC-IDS exDir client and froze the machine.
    block = max(1, min(1024, 8_000_000 // max(1, len(X))))
    distances_ = np.empty((len(X), k), dtype=np.float64)
    for start in range(0, len(X), block):
        d = cdist(X[start:start + block], X)
        d = np.partition(d, k, axis=1)[:, :k + 1]
        distances_[start:start + block] = np.sort(d, axis=1)[:, 1:k + 1]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    # Guard against any residual NaN/inf
    lids = np.where(np.isfinite(lids), lids, 0.0)
    return lids


# ------------------------------------------------------------------
# FedAvg helper
# ------------------------------------------------------------------

def _fedavg(w_list, sizes):
    total = float(sum(sizes))
    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg:
        w_avg[k] = torch.zeros_like(w_avg[k], dtype=torch.float32)
    for w, s in zip(w_list, sizes):
        for k in w_avg:
            w_avg[k] += w[k].float() * (s / total)
    return w_avg


# ------------------------------------------------------------------
# FedCorr Server
# ------------------------------------------------------------------

class FedCorr(object):
    """Three-stage noise-robust federated learning.

    Stage 1 – Noise Detection  (iteration1 outer iterations):
        Every iteration all clients train locally with a proximal term
        (beta * mu * ||w - w_g||, where mu is the client's estimated noise
        level from the previous iteration).  After local training, softmax
        outputs are fed into the LID estimator; per-client mean LID is
        accumulated across iterations.  At each iteration end a 2-component
        GMM partitions clients into clean / noisy by their cumulative LID
        (lower LID ⟹ cleaner data).  A second GMM on per-sample losses
        within each noisy client estimates that client's noise level.
        Optionally, mislabelled samples are relabelled using high-confidence
        global-model predictions (correction step).

    Stage 2 – Fine-tuning  (rounds1 rounds, optional):
        Only clients whose estimated noise level ≤ clean_set_thres
        participate.  Optionally performs a second relabelling pass.

    Stage 3 – Final training  (rounds2 rounds):
        Standard FedAvg on all clients (no proximal term, no correction).

    Reference: Xu et al., "FedCorr: Multi-Stage Federated Learning for
    Label Noise Robustness", CVPR 2022.  arXiv:2204.04677.
    """

    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.global_model = copy.deepcopy(args.model)
        self.num_classes = args.num_classes
        self.algo = args.algorithm
        self.times = times
        self.seed = args.seed
        self.exp_name = args.exp_name
        self.data_type = resolve_data_type(args.data_type, self.num_clients)
        args.data_type = self.data_type

        self.noise_type = normalize_noise_type(args.noise_type)
        args.noise_type = self.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.poisoned_worker_ids = []
        self.initialize_poisoned_workers()

        # ── FedCorr hyper-parameters ──────────────────────────────────
        self.iteration1 = args.iteration1          # Stage 1 outer iterations
        self.rounds1 = args.rounds1                # Stage 2 fine-tuning rounds
        self.rounds2 = args.rounds2                # Stage 3 final rounds
        self.frac1 = args.frac1 if args.frac1 > 0 else 1.0 / self.num_clients  # Stage 1: 1 client per sub-round
        self.frac2 = args.frac2                    # client fraction (Stage 2 / 3)
        self.beta = args.beta_fedcorr              # proximal term coefficient
        self.relabel_ratio = args.relabel_ratio    # fraction of noisy samples to relabel
        self.confidence_thres = args.confidence_thres  # min confidence to accept relabelling
        self.clean_set_thres = args.clean_set_thres    # max noise level to be "clean" in Stage 2
        self.correction = args.correction          # enable label correction
        self.fine_tuning = args.fine_tuning        # enable Stage 2 fine-tuning

        # ── Metrics ───────────────────────────────────────────────────
        self.rs_train_loss = []
        # Metrics trên toàn bộ client (giữ tương thích với kết quả cũ)
        self.rs_test_acc = []
        self.rs_test_f1 = []
        self.rs_test_precision = []
        self.rs_test_recall = []

        # Metrics tách riêng client normal và client flipping/noisy
        self.rs_test_acc_normal = []
        self.rs_test_f1_normal = []
        self.rs_test_precision_normal = []
        self.rs_test_recall_normal = []
        self.rs_test_acc_flipping = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_flipping = []

        self.train_losses = {i: [] for i in range(self.num_clients)}

        # ── Client partition state ────────────────────────────────────
        self.noisy_clients = []
        self.clean_clients = list(range(self.num_clients))
        self.estimated_noisy_level = np.zeros(self.num_clients)

        # ── Create clients ────────────────────────────────────────────
        self.clients = []
        self.set_clients(args, ClientFedCorr)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Algorithm: {self.algo} | Dataset: {self.dataset} | Data type: {self.data_type}")
        print(f"FedCorr | iteration1={self.iteration1}, rounds1={self.rounds1}, "
              f"rounds2={self.rounds2}")
        print(f"         beta={self.beta}, relabel_ratio={self.relabel_ratio}, "
              f"confidence_thres={self.confidence_thres}")
        print(f"         clean_set_thres={self.clean_set_thres}, "
              f"correction={self.correction}, fine_tuning={self.fine_tuning}")
        print("Finished creating server and clients.")

        # ── Result path & Logging ─────────────────────────────────────
        setup_experiment_logging(
            self,
            args,
            extra_lines=[
                f"FedCorr params: iteration1={self.iteration1}, rounds1={self.rounds1}, rounds2={self.rounds2}, beta={self.beta}, relabel_ratio={self.relabel_ratio}, confidence_thres={self.confidence_thres}, clean_set_thres={self.clean_set_thres}, correction={self.correction}, fine_tuning={self.fine_tuning}"
            ],
        )
        self.budget = []
        self.model_list = {}
        self.selected_history = {}
        self.best_score = None
        self.best_round = None
        self._round_counter = 0

    def initialize_poisoned_workers(self):
        """
        Xác định cố định các client bị tấn công (không random) theo tỷ lệ num_of_poisoned_workers:
        list(range(int(self.num_clients * self.num_of_poisoned_workers)))
        """
        self.poisoned_worker_ids = select_poisoned_workers(
            self.num_clients, self.noise_type, self.num_of_poisoned_workers
        )
        print(f"Noise Type: {self.noise_type}")
        print(f"Poisoned Worker IDs (fixed): {self.poisoned_worker_ids}")
    
    # ==================================================================
    # Main training loop
    # ==================================================================

    def train(self):

        # ==============================================================
        # Stage 1 – Noise Detection
        # ==============================================================
        print("\n===== Stage 1: Noise Detection =====")
        self._log("===== Stage 1: Noise Detection =====\n")

        LID_accumulative = np.zeros(self.num_clients)
        # Per-client per-sample loss accumulator (list length = train_samples)
        loss_accumulative = {
            i: np.zeros(len(self.clients[i].train_data))
            for i in range(self.num_clients)
        }
        estimated_noisy_level = np.zeros(self.num_clients)

        for iteration in range(self.iteration1):
            s_t = time.time()
            print(f"\n--- Stage 1  iter {iteration + 1}/{self.iteration1} ---")
            self._log(f"--- Stage 1 iter {iteration + 1}/{self.iteration1} ---\n")

            # mu_list: 0 on first iter, estimated noise level after
            mu_list = estimated_noisy_level.copy()

            LID_client = np.zeros(self.num_clients)
            loss_iter = {
                i: np.zeros(len(self.clients[i].train_data))
                for i in range(self.num_clients)
            }

            # ── Sub-rounds: cover all clients in groups of frac1 ──────
            # Mirrors the original: for _ in range(int(1/frac1)): sample frac1 clients
            n_sub = max(1, round(1.0 / self.frac1))
            n_per_sub = max(1, int(self.num_clients * self.frac1))
            perm = np.random.permutation(self.num_clients)
            # Build non-overlapping groups; remainder goes into last group
            groups = []
            for g in range(n_sub):
                start, end = g * n_per_sub, (g + 1) * n_per_sub
                groups.append(perm[start:end].tolist())
            if n_sub * n_per_sub < self.num_clients:
                groups[-1] += perm[n_sub * n_per_sub:].tolist()

            for group in groups:
                w_locals, w_sizes = [], []
                for idx in group:
                    client = self.clients[idx]
                    # Broadcast current global model
                    client.set_parameters(self.global_model)
                    # Supply proximal reference and noise estimate
                    client.global_model = copy.deepcopy(self.global_model).to(self.device)
                    client.mu = float(mu_list[idx])
                    # Local training
                    client.train(beta=self.beta)

                    # Compute LID from LOCAL model outputs (as in original code)
                    outputs, losses = client.get_outputs_and_loss(model=client.model)
                    lids = _lid_term(outputs)
                    LID_client[idx] = float(np.mean(lids))
                    loss_iter[idx] = losses

                    w_locals.append(client.model.state_dict())
                    w_sizes.append(client.train_samples)

                # FedAvg after each sub-group → update global model
                w_glob = _fedavg(w_locals, w_sizes)
                self.global_model.load_state_dict(w_glob)

            # ── Accumulate LID and losses ─────────────────────────────
            LID_accumulative += LID_client
            # the original re-initialises loss_accumulative_whole at the start of every
            # iteration, so only the current iteration's losses are used below
            for i in range(self.num_clients):
                loss_accumulative[i] = loss_iter[i].copy()

            # ── GMM on cumulative LID to identify clean / noisy clients
            lid_for_gmm = np.nan_to_num(LID_accumulative, nan=0.0, posinf=0.0, neginf=0.0)
            gmm_lid = GaussianMixture(n_components=2, random_state=self.seed).fit(
                lid_for_gmm.reshape(-1, 1)
            )
            lid_pred = gmm_lid.predict(lid_for_gmm.reshape(-1, 1))
            # Component with LOWER mean → cleaner clients (lower LID = cleaner)
            clean_comp = int(np.argmin(gmm_lid.means_[:, 0]))
            self.clean_clients = [i for i in range(self.num_clients) if lid_pred[i] == clean_comp]
            self.noisy_clients = [i for i in range(self.num_clients) if lid_pred[i] != clean_comp]

            # ── Estimate per-client noise level via loss-based GMM ────
            estimated_noisy_level = np.zeros(self.num_clients)
            for cid in self.noisy_clients:
                loss_c = loss_accumulative[cid]
                if len(loss_c) < 2:
                    continue
                gmm_loss = GaussianMixture(n_components=2, random_state=self.seed).fit(
                    loss_c.reshape(-1, 1)
                )
                loss_pred = gmm_loss.predict(loss_c.reshape(-1, 1))
                clean_comp_loss = int(np.argmin(gmm_loss.means_[:, 0]))
                n_noisy_samples = int(np.sum(loss_pred != clean_comp_loss))
                estimated_noisy_level[cid] = n_noisy_samples / len(loss_c)

            print(f"  LID accum:     {np.round(LID_accumulative, 2).tolist()}")
            print(f"  Clean clients: {self.clean_clients}")
            print(f"  Noisy clients: {self.noisy_clients}")
            print(f"  Noise levels:  {np.round(estimated_noisy_level, 3).tolist()}")
            self._log(
                f"Clean: {self.clean_clients}\n"
                f"Noisy: {self.noisy_clients}\n"
                f"Noise levels: {np.round(estimated_noisy_level, 3).tolist()}\n"
            )

            # ── Optional label correction ─────────────────────────────
            if self.correction:
                self._correct_labels(estimated_noisy_level, loss_accumulative)

            self._evaluate_and_log(f"[Stage 1 iter {iteration + 1}]")
            self.budget.append(time.time() - s_t)

        # Save final Stage 1 estimates for Stage 2 selection
        self.estimated_noisy_level = estimated_noisy_level

        # ==============================================================
        # Stage 2 – Fine-tuning on clean clients (optional)
        # ==============================================================
        if self.fine_tuning:
            print("\n===== Stage 2: Fine-tuning (clean clients only) =====")
            self._log("===== Stage 2: Fine-tuning =====\n")

            # Clients whose estimated noise level is below threshold
            selected_clean = [
                i for i in range(self.num_clients)
                if estimated_noisy_level[i] <= self.clean_set_thres
            ]
            if not selected_clean:
                selected_clean = list(self.clean_clients)

            m2 = max(1, min(int(self.frac2 * self.num_clients), len(selected_clean)))

            for rnd in range(self.rounds1):
                s_t = time.time()
                chosen = list(np.random.choice(selected_clean, m2, replace=False))
                w_locals, w_sizes = [], []
                for idx in chosen:
                    client = self.clients[idx]
                    client.set_parameters(self.global_model)
                    client.global_model = None
                    client.mu = 0.0
                    client.train(beta=0.0)
                    w_locals.append(client.model.state_dict())
                    w_sizes.append(client.train_samples)
                w_glob = _fedavg(w_locals, w_sizes)
                self.global_model.load_state_dict(w_glob)

                self._evaluate_and_log(f"[Stage 2 rnd {rnd + 1}]")
                self.budget.append(time.time() - s_t)

            # Second correction pass (original): once, after all fine-tuning rounds,
            # relabel the noisy clients with the fine-tuned global model
            if self.correction:
                self._correct_labels_stage2()

        # ==============================================================
        # Stage 3 – Final training (all clients, standard FedAvg)
        # ==============================================================
        print("\n===== Stage 3: Final Training (all clients) =====")
        self._log("===== Stage 3: Final Training =====\n")

        m3 = max(1, int(self.frac2 * self.num_clients))

        for rnd in range(self.rounds2):
            s_t = time.time()
            chosen = list(
                np.random.choice(self.num_clients, m3, replace=False)
            )
            w_locals, w_sizes = [], []
            for idx in chosen:
                client = self.clients[idx]
                client.set_parameters(self.global_model)
                client.global_model = None
                client.mu = 0.0
                client.train(beta=0.0)
                w_locals.append(client.model.state_dict())
                w_sizes.append(client.train_samples)
            w_glob = _fedavg(w_locals, w_sizes)
            self.global_model.load_state_dict(w_glob)

            self._evaluate_and_log(f"[Stage 3 rnd {rnd + 1}]")
            self.budget.append(time.time() - s_t)

        # Final summary
        write_final_summary(self)
        self.save_results()

    # ==================================================================
    # Label correction helpers
    # ==================================================================

    def _correct_labels(self, estimated_noisy_level, loss_accumulative):
        """Stage 1 correction.

        For each noisy client:
          1. Rank samples by descending accumulated loss (most noisy first).
          2. Keep only the top (n * noise_level * relabel_ratio) candidates.
          3. Filter by global-model confidence > confidence_thres.
          4. Relabel survivors to the global model's argmax prediction.
        """
        for idx in self.noisy_clients:
            client = self.clients[idx]
            n = len(client.train_data)
            if n == 0:
                continue

            # Global-model softmax outputs for all training samples
            outputs, _ = client.get_outputs_and_loss(
                model=copy.deepcopy(self.global_model).to(self.device)
            )

            loss_c = loss_accumulative[idx]
            n_relabel = int(n * estimated_noisy_level[idx] * self.relabel_ratio)
            top_loss_idx = np.argsort(-loss_c)[:n_relabel]

            max_conf = np.max(outputs, axis=1)
            relabel_idx = [
                i for i in top_loss_idx if max_conf[i] > self.confidence_thres
            ]
            if relabel_idx:
                new_labels = np.argmax(outputs[relabel_idx], axis=1)
                client.relabel(relabel_idx, new_labels)
                self._log(
                    f"  [Correction] client {idx}: "
                    f"relabelled {len(relabel_idx)}/{n} samples\n"
                )

    def _correct_labels_stage2(self):
        """Stage 2 correction.

        Relabel any sample in noisy clients where the fine-tuned global model
        predicts with confidence above confidence_thres.
        """
        for idx in self.noisy_clients:
            client = self.clients[idx]
            if not client.train_data:
                continue
            outputs, _ = client.get_outputs_and_loss(
                model=copy.deepcopy(self.global_model).to(self.device)
            )
            max_conf = np.max(outputs, axis=1)
            relabel_idx = list(np.where(max_conf > self.confidence_thres)[0])
            if relabel_idx:
                new_labels = np.argmax(outputs[relabel_idx], axis=1)
                client.relabel(relabel_idx, new_labels)

    # ==================================================================
    # Standard FL plumbing
    # ==================================================================

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(
                self.dataset, i, self.num_clients, data_type='train',type=self.data_type
            )
            test_data = read_client_data(
                self.dataset, i, self.num_clients, data_type='test',type=self.data_type
            )
            client = clientObj(
                args=args,
                id=i,
                poisoned_worker_ids=self.poisoned_worker_ids,
                train_samples=len(train_data),
                test_samples=len(test_data),
            )
            self.clients.append(client)

    def _evaluate_and_log(self, tag=""):
        # Mỗi nhóm lưu metric đã nhân với số mẫu test của client,
        # sau đó chia tổng số mẫu để tính trung bình có trọng số.
        groups = {
            "normal": {"acc": [], "f1": [], "precision": [], "recall": [], "n": 0},
            "flipping": {"acc": [], "f1": [], "precision": [], "recall": [], "n": 0},
        }

        all_metrics = {"acc": [], "f1": [], "precision": [], "recall": [], "n": 0}
        loss_all = []

        for c in self.clients:
            acc, pre, rec, f1, ns = c.test_metrics()
            ns = int(ns)

            # Metric tổng thể trên tất cả client
            all_metrics["acc"].append(acc * ns)
            all_metrics["f1"].append(f1 * ns)
            all_metrics["precision"].append(pre * ns)
            all_metrics["recall"].append(rec * ns)
            all_metrics["n"] += ns

            # Tách nhóm theo danh sách client bị đánh dấu nhiễu/label flipping
            group_name = "flipping" if c.id in self.poisoned_worker_ids else "normal"
            groups[group_name]["acc"].append(acc * ns)
            groups[group_name]["f1"].append(f1 * ns)
            groups[group_name]["precision"].append(pre * ns)
            groups[group_name]["recall"].append(rec * ns)
            groups[group_name]["n"] += ns

            tl = c.train_metrics()
            if tl:
                loss_all.extend(tl)
                self.train_losses[c.id].append(float(np.mean(tl)))

        def weighted_average(metric_values, n_samples):
            return sum(metric_values) / n_samples if n_samples > 0 else 0.0

        # Giữ các metric tổng thể như bản cũ
        avg_acc = weighted_average(all_metrics["acc"], all_metrics["n"])
        avg_f1 = weighted_average(all_metrics["f1"], all_metrics["n"])
        avg_pre = weighted_average(all_metrics["precision"], all_metrics["n"])
        avg_rec = weighted_average(all_metrics["recall"], all_metrics["n"])
        avg_loss = float(np.mean(loss_all)) if loss_all else 0.0

        self.rs_test_acc.append(avg_acc)
        self.rs_test_f1.append(avg_f1)
        self.rs_test_precision.append(avg_pre)
        self.rs_test_recall.append(avg_rec)
        self.rs_train_loss.append(avg_loss)

        # Tính và lưu riêng từng nhóm normal / flipping
        for name, metrics in groups.items():
            n_samples = metrics["n"]
            values = {
                key: weighted_average(metrics[key], n_samples)
                for key in ("acc", "f1", "precision", "recall")
            }
            if name == "normal":
                self.rs_test_acc_normal.append(values["acc"])
                self.rs_test_f1_normal.append(values["f1"])
                self.rs_test_precision_normal.append(values["precision"])
                self.rs_test_recall_normal.append(values["recall"])
            else:
                self.rs_test_acc_flipping.append(values["acc"])
                self.rs_test_f1_flipping.append(values["f1"])
                self.rs_test_precision_flipping.append(values["precision"])
                self.rs_test_recall_flipping.append(values["recall"])

            group_msg = (
                f"{tag} [{name}] acc={values['acc']:.4f} "
                f"f1={values['f1']:.4f} "
                f"prec={values['precision']:.4f} "
                f"rec={values['recall']:.4f} n={n_samples}"
            )
            print(group_msg)
            self._log(group_msg + "\n")

        msg = (f"{tag} [all] acc={avg_acc:.4f} f1={avg_f1:.4f} "
               f"prec={avg_pre:.4f} rec={avg_rec:.4f} loss={avg_loss:.4f}")
        print(msg)
        self._log(msg + "\n")

        track_best(self, self._round_counter, self.rs_test_f1_normal[-1], self.global_model, self.clients)
        self.model_list[self._round_counter] = round_models(self.global_model, self.clients)
        self._round_counter += 1

    def _log(self, msg):
        with open(self.log_file, 'a') as f:
            f.write(msg)

    def save_results(self):
        results = {
            'dataset': self.dataset,
            'data_type': self.data_type,
            'algorithm': self.algo,
            'exp_name': self.exp_name,
            'scenario_name': getattr(self, 'scenario_name', ''),
            'seed': self.seed,
            'noise_type': self.noise_type,
            'num_of_poisoned_workers': self.num_of_poisoned_workers,
            'poisoned_sample_ratio': self.poisoned_sample_ratio,

            # Metrics tổng thể trên tất cả client
            'acc': self.rs_test_acc,
            'f1': self.rs_test_f1,
            'recall': self.rs_test_recall,
            'precision': self.rs_test_precision,

            # Metrics tách riêng client normal
            'acc-normal': self.rs_test_acc_normal,
            'f1-normal': self.rs_test_f1_normal,
            'precision-normal': self.rs_test_precision_normal,
            'recall-normal': self.rs_test_recall_normal,

            # Metrics tách riêng client flipping/noisy
            'acc-flipping': self.rs_test_acc_flipping,
            'f1-flipping': self.rs_test_f1_flipping,
            'precision-flipping': self.rs_test_precision_flipping,
            'recall-flipping': self.rs_test_recall_flipping,

            'train-loss': self.rs_train_loss,
            'loss': self.train_losses,
            'poisoned_worker_ids': self.poisoned_worker_ids,
            'noisy_clients': self.noisy_clients,
            'clean_clients': self.clean_clients,
            'estimated_noisy_level': self.estimated_noisy_level.tolist(),
            'best_round': self.best_round,
            'best_f1_normal': self.best_score,
        }
        save_pickle(results, os.path.join(self.result_path, f'metrics_results{run_suffix(self.times)}.pkl'))
        save_model_artifacts(self, self.global_model, self.model_list)
