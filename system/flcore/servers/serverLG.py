import copy
import numpy as np
import torch
from flcore.clients.clientLG import *
from utils.data_utils import read_client_data
import os
import pickle
import time
import random
from flcore.servers.model_saver import (round_models, track_best, save_model_artifacts,
                                        save_pickle, mean_round_time, run_suffix,
                                        normalize_noise_type, select_poisoned_workers,
                                        setup_experiment_logging, write_final_summary)
from utils.data_utils import resolve_data_type


class LG_FedAvg(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.round_to_add_clients = args.round_to_add_clients
        # self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # for visualize
        self.algo = args.algorithm
        self.rs_train_loss = []
        self.rs_test_acc_normal = []
        self.rs_test_acc_flipping = []
        # self.rs_test_auc = []
        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []

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
        #Label flipping
        self.initialize_poisoned_workers()
        # self.nc = args.num_classes
        self.set_clients(args, ClientLG)

        self.global_model = copy.deepcopy(args.model.predictor)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset, "Data type: ", self.data_type)
        print("Finished creating server and clients.")
        # save loss
        self.train_losses = {i: [] for i in range(self.num_clients)}
        setup_experiment_logging(self, args)
        self.budget = []

        # for get model in each round
        self.model_list = {}
        self.selected_history = {}
        self.best_score = None
        self.best_round = None

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
            self.selected_clients = self.select_clients(epoch)
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')
            print("\nEvaluate global model")
            self.evaluate(epoch)
            self.selected_history[epoch] = [c.id for c in self.selected_clients]
            track_best(self, epoch, self.rs_test_f1_normal[-1], self.global_model, self.clients)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            # save checkpoint: bản sao CPU (trước đây lưu tham chiếu -> mọi round giống hệt nhau)
            # [global predictor sau aggregate, client 0, client 1, ...]
            self.model_list[epoch] = round_models(self.global_model, self.clients)

            self.budget.append(time.time() - s_t)
            print('Time:', self.budget[-1])

        write_final_summary(self)
        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train', type=self.data_type)
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test', type=self.data_type)
            client = clientObj(args=args,
                               id=i,
                               poisoned_worker_ids = self.poisoned_worker_ids,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self, epoch):
        if epoch < self.round_to_add_clients:
            # Chỉ chọn trong 15 client đầu tiên (ID 0–14)
            candidate_clients = self.clients[:15]
        else:
            # Từ epoch 40 trở đi, chọn từ tất cả 20 client
            candidate_clients = self.clients

        if self.random_join_ratio:
            join_clients = int(len(candidate_clients) * self.join_ratio)
            join_clients = np.random.choice(range(join_clients, len(candidate_clients) + 1), 1, replace=False)[0]
        else:
            join_clients = int(len(candidate_clients) * self.join_ratio)

        selected_clients = list(np.random.choice(candidate_clients, join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.selected_clients:
            client.set_parameters(self.global_model)
            #client.local_initialization(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.predictor)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

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

        # return total_acc, total_auc, total_f1, total_recall, total_precision, num_samples

    def train_metrics(self):
        num_samples = []
        losses = []
        print("Train Loss")
        with open(self.log_file, 'a') as f:
            f.write('Train Loss\n')
        for c in self.selected_clients:
            list_loss = c.train_metrics()
            loss = sum(list_loss) / len(list_loss)
            self.train_losses[c.id].append(loss)  # save loss of each client after each round
            print(f'\tClient {c.id}: Train loss: {round(loss, 2)}')
            with open(self.log_file, 'a') as f:
                f.write(f'\tClient {c.id}: Train loss: {round(loss, 2)}\n')
            num_samples.append(len(list_loss))
            losses.extend(list_loss)

        return num_samples, losses

    def evaluate(self, epoch):
        # --- 1. Đánh giá Train (Giữ nguyên) ---
        stats_train = self.train_metrics()
        train_loss = sum(stats_train[1]) * 1.0 / sum(stats_train[0])
        self.rs_train_loss.append(train_loss)

        # --- 2. Đánh giá Local Test (Giữ nguyên logic của bạn) ---
        stats_test = self.test_metrics()

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
        self.rs_test_f1_normal.append(test_f1_normal)
        self.rs_test_precision_normal.append(test_precision_normal)
        self.rs_test_recall_normal.append(test_recall_normal)

        self.rs_test_acc_flipping.append(test_acc_flipping)
        self.rs_test_f1_flipping.append(test_f1_flipping)
        self.rs_test_precision_flipping.append(test_precision_flipping)
        self.rs_test_recall_flipping.append(test_recall_flipping)

        # --- 3. PHẦN THÊM MỚI: Đánh giá GLOBAL cho từng client ---
        glob_accs_per_client = []
        glob_f1s_per_client = []
        
        # Tạo danh sách lưu trữ nếu chưa có
        if not hasattr(self, 'rs_glob_acc_per_client'):
            self.rs_glob_acc_per_client = [] # Lưu list các acc của các client qua mỗi epoch

        for c in self.selected_clients:
            # Gọi hàm test_metrics_global đã viết ở bước trước trong class Client
            g_acc, g_pre, g_rec, g_f1, g_num = c.test_metrics_global()
            glob_accs_per_client.append(g_acc)
            glob_f1s_per_client.append(g_f1)
        
        # Lưu kết quả global trung bình của tất cả client trong epoch này
        avg_glob_acc = sum(glob_accs_per_client) / len(glob_accs_per_client)
        self.rs_glob_acc_per_client.append(avg_glob_acc)

        # --- 4. Hiển thị (Bổ sung phần Global) ---
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset, "Epoch: ", epoch)
        print("Overall Performance Metrics (Local)")
        print("\ttrain loss avg: {:.4f}".format(train_loss))
        print("\ttest acc normal avg: {:.4f}".format(test_acc_normal))
        print("\ttest acc flipping avg: {:.4f}".format(test_acc_flipping))
        
        print("Global Performance Metrics (Whole Test Set)")
        print("\tGlobal Acc avg: {:.4f}".format(avg_glob_acc))
        # In chi tiết một vài client tiêu biểu nếu muốn
        print(f"\tAll Clients Global Acc: {[round(a, 4) for a in glob_accs_per_client]}")
        print(f"\tAll Clients Global F1:  {[round(f, 4) for f in glob_f1s_per_client]}")

        # --- 5. Lưu File (Bổ sung phần Global) ---
        with open(self.log_file, 'a') as f:
            f.write("\nEpoch: {}, Algorithm: {}, Dataset: {}\n".format(epoch, self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))
            f.write("\ttest acc normal avg: {:.4f}\n".format(test_acc_normal))
            f.write("\ttest acc flipping avg: {:.4f}\n".format(test_acc_flipping))
            # Ghi thêm dòng Global
            f.write("\ttest acc GLOBAL avg: {:.4f}\n".format(avg_glob_acc))
            f.write("\ttest acc GLOBAL detail: {}\n".format([round(x, 4) for x in glob_accs_per_client]))
            f.write("\ttest f1 GLOBAL detail: {}\n".format([round(x, 4) for x in glob_f1s_per_client]))
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
            'acc-flipping': self.rs_test_acc_flipping,
            'acc-normal': self.rs_test_acc_normal,
            'f1-flipping': self.rs_test_f1_flipping,
            'f1-normal': self.rs_test_f1_normal,
            'recall-flipping': self.rs_test_recall_flipping,
            'precision-flipping': self.rs_test_precision_flipping,
            'precision-normal': self.rs_test_precision_normal,
            'recall-normal': self.rs_test_recall_normal,
            'train-loss': self.rs_train_loss,
            'glob-acc-per-client': getattr(self, 'rs_glob_acc_per_client', []),
            'loss': self.train_losses,
            'loss-normal': self.train_losses,
            'poisoned_worker_ids': self.poisoned_worker_ids,
            'selected_clients': self.selected_history,
            'best_round': self.best_round,
            'best_f1_normal': self.best_score,
        }

        save_pickle(metrics_results, os.path.join(self.result_path, f'metrics_results{run_suffix(self.times)}.pkl'))

        # save checkpoint of global, client 0, client 1, ... after each commnunication round
        save_model_artifacts(self, self.global_model, self.model_list)
