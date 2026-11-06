import copy
import numpy as np
import torch
from flcore.clients.clientNoRo import ClientNoRo
from utils.data_utils import read_client_data, resolve_data_type
from sklearn.mixture import GaussianMixture
import os
import pickle
import time
from flcore.servers.model_saver import (round_models, track_best, save_model_artifacts,
                                        save_pickle, mean_round_time, run_suffix,
                                        normalize_noise_type, select_poisoned_workers,
                                        setup_experiment_logging, write_final_summary)


class FedNoRo(object):
    """Federated learning with noise-robust aggregation (FedNoRo).

    Two-stage algorithm from "FedNoRo: Towards Noise-Robust Federated Learning by
    Addressing Class Imbalance and Label Noise Heterogeneity"
    (arXiv:2305.05230, IJCAI 2023).

    Stage 1 – Warm-up  (rounds 0 … warmup_rounds-1):
        All clients train with standard BCE.  Server aggregates via FedAvg.

    Transition (once, at round == warmup_rounds):
        Server uses GMM ensemble voting on per-client per-class loss metrics
        to partition clients into clean and noisy sets.

    Stage 2 – Noise-robust training  (rounds warmup_rounds … global_rounds):
        · Clean clients: standard BCE training (same as Stage 1).
        · Noisy clients: knowledge-distillation training that blends a soft-BCE
          term against the teacher (global model) with the hard-BCE term on noisy
          local labels.  The blending weight w_kd is ramped up via a sigmoid
          schedule so the model first relies on local labels and gradually shifts
          to the teacher's guidance.
        · Server aggregates via DaAgg: penalises noisy client weights according
          to their L2 distance from the nearest clean client model.
    """

    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.clients = []
        self.selected_clients = []

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

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.algo = args.algorithm
        self.rs_train_loss = []
        self.rs_test_acc = []
        self.rs_test_f1 = []
        self.rs_test_precision = []
        self.rs_test_recall = []

        self.rs_test_acc_normal = []
        self.rs_test_acc_flipping = []
        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []

        self.num_classes = args.num_classes

        self.pat = args.patition
        self.alpha = args.alpha_dir
        self.poison_attack = args.poison_attack

        # FedNoRo hyperparameters
        self.warmup_rounds = args.warmup_rounds  # Stage 1 duration
        self.kd_begin = args.kd_begin            # round to start ramping up w_kd
        self.kd_end = args.kd_end                # round to finish ramping up w_kd
        self.kd_a = args.kd_a                    # maximum KD weight multiplier

        # Populated once after Stage 1
        self.noisy_clients = []
        self.clean_clients = []
        self.stage1_model = None

        self.set_clients(args, ClientNoRo)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset, "Data type: ", self.data_type)
        print(f"FedNoRo: warmup_rounds={self.warmup_rounds}, "
              f"kd_begin={self.kd_begin}, kd_end={self.kd_end}, kd_a={self.kd_a}")
        print("Finished creating server and clients.")

        self.train_losses = {i: [] for i in range(self.num_clients)}
        setup_experiment_logging(
            self,
            args,
            extra_lines=[
                f"FedNoRo params: warmup_rounds={self.warmup_rounds}, kd_begin={self.kd_begin}, kd_end={self.kd_end}, kd_a={self.kd_a}"
            ],
        )
        self.budget = []
        self.model_list = {}
        self.selected_history = {}
        self.best_score = None
        self.best_round = None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def initialize_poisoned_workers(self):
        """
        Xác định các client bị tấn công.
        """
        self.poisoned_worker_ids = select_poisoned_workers(
            self.num_clients, self.noise_type, self.num_of_poisoned_workers
        )
        print(f"Noise Type: {self.noise_type}")
        print(f"Poisoned Worker IDs: {self.poisoned_worker_ids}")

    def train(self):
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')

            if epoch < self.warmup_rounds:
                # Stage 1: all clients use standard BCE, no KD
                print(f"[Stage 1 - Warm-up] {epoch}/{self.warmup_rounds - 1}")
                self._set_clients_stage1()
            else:
                # Transition: identify noisy clients once using Stage-1 global model
                if epoch == self.warmup_rounds:
                    self.stage1_model = copy.deepcopy(self.global_model)
                    self._identify_noisy_clients()

                w_kd = self._sigmoid_rampup(epoch, self.kd_begin, self.kd_end) * self.kd_a
                print(f"[Stage 2 - Robust] round={epoch}, w_kd={w_kd:.4f}, "
                      f"noisy={self.noisy_clients}")
                with open(self.log_file, 'a') as f:
                    f.write(f"[Stage 2] w_kd={w_kd:.4f}, noisy={self.noisy_clients}\n")
                self._set_clients_stage2(w_kd)

            print("\nEvaluate global model")
            self.evaluate()
            self.selected_history[epoch] = [c.id for c in self.selected_clients]
            track_best(self, epoch, self.rs_test_f1_normal[-1], self.global_model, self.clients)

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            if epoch < self.warmup_rounds:
                self.aggregate_parameters()   # Stage 1: standard FedAvg
            else:
                self._daagg()                 # Stage 2: distance-aware aggregation

            self.model_list[epoch] = round_models(self.global_model, self.clients)

            self.budget.append(time.time() - s_t)
            print('Time:', self.budget[-1])

        # --- FINAL SUMMARY (chạy 1 lần sau khi kết thúc tất cả các round) ---
        write_final_summary(self)
        self.save_results()

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _set_clients_stage1(self):
        """Put all selected clients into standard (non-KD) mode."""
        for client in self.selected_clients:
            client.is_noisy = False
            client.teacher_model = None
            client.w_kd = 0.0

    def _set_clients_stage2(self, w_kd):
        """Configure selected clients for Stage 2.

        Noisy clients receive a copy of the current global model as teacher and
        the current KD weight.  Clean clients continue with standard BCE.
        """
        for client in self.selected_clients:
            if client.id in self.noisy_clients:
                client.is_noisy = True
                client.teacher_model = copy.deepcopy(self.global_model).to(self.device)
                client.w_kd = w_kd
            else:
                client.is_noisy = False
                client.teacher_model = None
                client.w_kd = 0.0

    # ------------------------------------------------------------------
    # Noisy-client identification (GMM ensemble, run once after Stage 1)
    # ------------------------------------------------------------------

    def _identify_noisy_clients(self):
        """Identify noisy clients via GMM ensemble voting on per-class BCE loss.

        For each client we compute a 2-element feature vector:
            [mean_BCE_loss_on_class0, mean_BCE_loss_on_class1]
        using the Stage-1 global model on each client's local training data.

        We then fit 9 independent GMMs (random_state 0..8) and take a majority
        vote across their predictions.  The component with the higher centroid
        mean is labelled 'noisy'.  Clients receiving more than 4 noisy votes
        are placed in self.noisy_clients; the rest in self.clean_clients.

        Follows Algorithm 1 of the FedNoRo paper adapted for binary classification.
        """
        print("\n[FedNoRo] Identifying noisy clients via GMM ensemble...")
        with open(self.log_file, 'a') as f:
            f.write("\n[FedNoRo] Identifying noisy clients via GMM ensemble...\n")

        n_classes = self.num_classes
        metrics_matrix = np.zeros((self.num_clients, n_classes))

        for client in self.clients:
            metrics_matrix[client.id] = client.compute_per_class_losses(
                model=self.stage1_model
            )

        # Replace NaN with column minimum
        for j in range(n_classes):
            col = metrics_matrix[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col_min = float(np.nanmin(col)) if not nan_mask.all() else 0.0
                metrics_matrix[nan_mask, j] = col_min

        # Min-max normalise each class column independently
        metrics_norm = metrics_matrix.copy()
        for j in range(n_classes):
            col_min = metrics_matrix[:, j].min()
            col_max = metrics_matrix[:, j].max()
            if col_max > col_min:
                metrics_norm[:, j] = (metrics_matrix[:, j] - col_min) / (col_max - col_min)
            else:
                metrics_norm[:, j] = 0.0

        # Ensemble of 9 GMMs; majority vote decides noisy status
        n_gmm = 9
        noisy_votes = np.zeros(self.num_clients, dtype=int)

        for seed in range(n_gmm):
            gmm = GaussianMixture(n_components=2, random_state=seed, max_iter=200)
            gmm.fit(metrics_norm)
            gmm_pred = gmm.predict(metrics_norm)

            # Component with the higher overall mean = noisy component
            comp_means = [
                metrics_norm[gmm_pred == k].mean() if (gmm_pred == k).any() else 0.0
                for k in range(2)
            ]
            noisy_component = int(np.argmax(comp_means))

            for idx in range(self.num_clients):
                if gmm_pred[idx] == noisy_component:
                    noisy_votes[idx] += 1

        majority = n_gmm // 2
        self.noisy_clients = [i for i in range(self.num_clients) if noisy_votes[i] > majority]
        self.clean_clients = [i for i in range(self.num_clients) if noisy_votes[i] <= majority]

        print(f"[FedNoRo] Noisy clients: {self.noisy_clients}")
        print(f"[FedNoRo] Clean clients: {self.clean_clients}")
        print(f"[FedNoRo] GMM votes: {noisy_votes.tolist()}")
        with open(self.log_file, 'a') as f:
            f.write(f"[FedNoRo] Noisy clients: {self.noisy_clients}\n")
            f.write(f"[FedNoRo] Clean clients: {self.clean_clients}\n")
            f.write(f"[FedNoRo] GMM votes: {noisy_votes.tolist()}\n")

    # ------------------------------------------------------------------
    # Distance-Aware Aggregation (DaAgg) – Stage 2
    # ------------------------------------------------------------------

    def _model_distance(self, model1, model2):
        """L2 distance between two models' parameter tensors."""
        total = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            total += torch.norm(p1.data.float() - p2.data.float()).item()
        return total

    def _daagg(self):
        """Distance-Aware Aggregation.

        Base weight for each client is proportional to its data size (same as
        FedAvg).  For noisy clients we additionally multiply by exp(-d), where
        d is the normalised L2 distance to the nearest clean client in model
        space.  This down-weights noisy clients whose updates diverge far from
        the clean consensus while preserving those whose updates happen to agree.

        Falls back to standard FedAvg when no clean or no noisy clients are
        present among the uploaded models.
        """
        assert len(self.uploaded_models) > 0

        weights = np.array(self.uploaded_weights, dtype=float)
        id_to_idx = {cid: idx for idx, cid in enumerate(self.uploaded_ids)}

        uploaded_noisy = [cid for cid in self.uploaded_ids if cid in self.noisy_clients]
        uploaded_clean = [cid for cid in self.uploaded_ids if cid in self.clean_clients]

        if uploaded_noisy and uploaded_clean:
            distances = np.zeros(len(self.uploaded_ids))

            for noisy_id in uploaded_noisy:
                noisy_idx = id_to_idx[noisy_id]
                noisy_model = self.uploaded_models[noisy_idx]
                min_dist = min(
                    self._model_distance(noisy_model, self.uploaded_models[id_to_idx[cid]])
                    for cid in uploaded_clean
                )
                distances[noisy_idx] = min_dist

            max_dist = distances.max()
            if max_dist > 0:
                distances = distances / max_dist

            weights = weights * np.exp(-distances)
            weights = weights / weights.sum()

        # Weighted parameter averaging
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    # ------------------------------------------------------------------
    # KD weight schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid_rampup(current, begin, end):
        """Exponential ramp-up from 0 → 1 over the interval [begin, end].

        Returns 0 before `begin`, 1 after `end`, and a smooth sigmoid-shaped
        value in between.  Matches the schedule in the FedNoRo paper (Eq. 9).
        """
        if current < begin:
            return 0.0
        if current > end:
            return 1.0
        phase = 1.0 - (current - begin) / (end - begin)
        return float(np.exp(-5.0 * phase * phase))

    # ------------------------------------------------------------------
    # Standard FL plumbing (mirrors serverAvg.py)
    # ------------------------------------------------------------------

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(
                self.dataset, i, self.num_clients, partition=self.pat, alpha=self.alpha,
                data_type='train', type=self.data_type, poison_attack=self.poison_attack
            )
            test_data = read_client_data(
                self.dataset, i, self.num_clients, partition=self.pat, alpha=self.alpha,
                data_type='test', type=self.data_type, poison_attack=self.poison_attack
            )
            client = clientObj(
                args=args,
                id=i,
                poisoned_worker_ids=self.poisoned_worker_ids,
                train_samples=len(train_data),
                test_samples=len(test_data)
            )
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(
                range(self.join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            join_clients = self.join_clients
        return list(np.random.choice(self.clients, join_clients, replace=False))

    def send_models(self):
        assert len(self.clients) > 0
        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_train_samples = sum(c.train_samples for c in self.selected_clients)

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        """Standard FedAvg aggregation used in Stage 1."""
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        #normal
        total_acc_normal = []
        total_precision_normal = []
        total_recall_normal = []
        total_f1_normal = []
        num_samples_normal = 0
        #flipping
        total_acc_flipping = []
        total_precision_flipping = []
        total_recall_flipping = []
        total_f1_flipping = []
        # total_auc = []
        num_samples_flipping = 0

        print('Performance Metrics')
        with open(self.log_file, 'a') as f:
            f.write('Performance Metrics\n')
        for c in self.selected_clients:
            """
            # acc, precision, recall, f1, auc, ns = c.test_metrics()
            # print(
            #     f'\tClient {c.id}: acc {round(acc, 2)}, auc: {round(auc, 2)}, precision: {round(precision, 2)}, '
            #     f'recall: {round(recall, 2)}, f1: {round(f1, 2)}')
            # with open(self.log_file, 'a') as f:
            #     f.write(
            #         f'\tClient {c.id}: acc {round(acc, 2)}, auc: {round(auc, 2)}, precision: {round(precision, 2)}, '
            #         f'recall: {round(recall, 2)}, f1: {round(f1, 2)}\n')
            """
            acc, precision, recall, f1, ns = c.test_metrics()
            print(
                f'\tClient {c.id}: acc {round(acc, 2)}, precision: {round(precision, 2)}, '
                f'recall: {round(recall, 2)}, f1: {round(f1, 2)}')
            with open(self.log_file, 'a') as f:
                f.write(
                    f'\tClient {c.id}: acc {round(acc, 2)}, precision: {round(precision, 2)}, '
                    f'recall: {round(recall, 2)}, f1: {round(f1, 2)}\n')

            if c.id in self.poisoned_worker_ids:
                total_acc_flipping.append(acc * ns)
                # total_auc.append(auc * ns)
                total_recall_flipping.append(recall * ns)
                total_f1_flipping.append(f1 * ns)
                total_precision_flipping.append(precision * ns)
                num_samples_flipping += ns
            else:
                total_acc_normal.append(acc * ns)
                # total_auc.append(auc * ns)
                total_recall_normal.append(recall * ns)
                total_f1_normal.append(f1 * ns)
                total_precision_normal.append(precision * ns)
                num_samples_normal += ns

        return total_acc_normal, total_f1_normal, total_recall_normal, total_precision_normal, num_samples_normal, total_acc_flipping, total_f1_flipping, total_recall_flipping, total_precision_flipping, num_samples_flipping


    def train_metrics(self):
        num_samples = []
        losses = []
        print("Train Loss")
        with open(self.log_file, 'a') as f:
            f.write('Train Loss\n')
        for c in self.selected_clients:
            list_loss = c.train_metrics()
            loss = sum(list_loss) / len(list_loss)
            self.train_losses[c.id].append(loss)
            print(f'\tClient {c.id}: Train loss: {round(loss, 2)}')
            with open(self.log_file, 'a') as f:
                f.write(f'\tClient {c.id}: Train loss: {round(loss, 2)}\n')
            num_samples.append(len(list_loss))
            losses.extend(list_loss)
        return num_samples, losses

    def evaluate(self):
        # evaluate train
        stats_train = self.train_metrics()
        train_loss = sum(stats_train[1]) * 1.0 / sum(stats_train[0])  # avg train loss for all clients
        self.rs_train_loss.append(train_loss)

        # evaluate test, avg metrics for all clients
        stats_test = self.test_metrics()
        # test_acc = sum(stats_test[0]) / stats_test[5]
        # test_auc = sum(stats_test[1]) / stats_test[5]
        # test_f1 = sum(stats_test[2]) / stats_test[5]
        # test_recall = sum(stats_test[3]) / stats_test[5]
        # test_precision = sum(stats_test[4]) / stats_test[5]

        if stats_test[4] > 0:
            test_acc_normal = sum(stats_test[0]) / stats_test[4]
            test_f1_normal = sum(stats_test[1]) / stats_test[4]
            test_recall_normal = sum(stats_test[2]) / stats_test[4]
            test_precision_normal = sum(stats_test[3]) / stats_test[4]
        else:
            test_acc_normal = 0
            test_f1_normal = 0
            test_recall_normal = 0
            test_precision_normal= 0

        if stats_test[9] > 0:
            test_acc_flipping = sum(stats_test[5]) / stats_test[9]
            test_f1_flipping = sum(stats_test[6]) / stats_test[9]
            test_recall_flipping = sum(stats_test[7]) / stats_test[9]
            test_precision_flipping = sum(stats_test[8]) / stats_test[9]
        else:
            test_acc_flipping = 0
            test_f1_flipping = 0
            test_recall_flipping = 0
            test_precision_flipping = 0

        self.rs_test_acc_normal.append(test_acc_normal)
        # self.rs_test_auc.append(test_auc)
        self.rs_test_f1_normal.append(test_f1_normal)
        self.rs_test_precision_normal.append(test_precision_normal)
        self.rs_test_recall_normal.append(test_recall_normal)

        self.rs_test_acc_flipping.append(test_acc_flipping)
        # self.rs_test_auc.append(test_auc)
        self.rs_test_f1_flipping.append(test_f1_flipping)
        self.rs_test_precision_flipping.append(test_precision_flipping)
        self.rs_test_recall_flipping.append(test_recall_flipping)

        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("Overall Performance Metrics")
        print("\ttrain loss avg: {:.4f}".format(train_loss))
        print("\ttest acc normal avg: {:.4f} | f1 normal avg: {:.4f}".format(test_acc_normal, test_f1_normal))
        print("\ttest acc flipping avg: {:.4f} | f1 flipping avg: {:.4f}".format(test_acc_flipping, test_f1_flipping))

        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))
            f.write("\ttest acc normal avg: {:.4f} | f1 normal avg: {:.4f}\n".format(test_acc_normal, test_f1_normal))
            f.write("\ttest acc flipping avg: {:.4f} | f1 flipping avg: {:.4f}\n".format(test_acc_flipping, test_f1_flipping))

    def save_results(self):
        metrics_results = {
            'dataset': self.dataset,
            'data_type': self.data_type,
            'algorithm': self.algo,
            'exp_name': self.exp_name,
            'scenario_name': getattr(self, 'scenario_name', ''),
            'seed': self.seed,
            'noise_type': self.noise_type,
            'num_of_poisoned_workers': self.num_of_poisoned_workers,
            'poisoned_sample_ratio': self.poisoned_sample_ratio,
            'acc-normal': self.rs_test_acc_normal,
            'f1-normal': self.rs_test_f1_normal,
            'precision-normal': self.rs_test_precision_normal,
            'recall-normal': self.rs_test_recall_normal,
            'acc-flipping': self.rs_test_acc_flipping,
            'f1-flipping': self.rs_test_f1_flipping,
            'precision-flipping': self.rs_test_precision_flipping,
            'recall-flipping': self.rs_test_recall_flipping,
            'f1-per-class-normal': self.rs_test_f1_normal,
            'train-loss': self.rs_train_loss,
            'loss': self.train_losses,
            'poisoned_worker_ids': self.poisoned_worker_ids,
            'noisy_clients': self.noisy_clients,
            'clean_clients': self.clean_clients,
            'selected_clients': self.selected_history,
            'best_round': self.best_round,
            'best_f1_normal': self.best_score,
        }
        save_pickle(metrics_results, os.path.join(self.result_path, f'metrics_results{run_suffix(self.times)}.pkl'))
        save_model_artifacts(self, self.global_model, self.model_list)
