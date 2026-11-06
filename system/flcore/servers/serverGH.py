import copy
import numpy as np
import torch
from flcore.clients.clientGH import *
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

class FedGH(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_protos = []

        # for visualize
        self.algo = args.algorithm
        self.rs_train_loss_normal = []
        self.rs_train_loss_flipping = []
        self.rs_train_loss = []
        self.rs_test_acc_normal = []
        self.rs_test_acc_flipping = []
        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []

        # --- NEW: Per-Class F1 history & Best Snapshot ---
        self.num_classes = args.num_classes
        self.rs_test_f1_pc_normal = [] 
        self.best_f1_pc_normal = None 
        self.best_f1_normal = -1.0

        self.times = times
        self.seed = args.seed
        self.exp_name = args.exp_name
        self.data_type = resolve_data_type(args.data_type, self.num_clients)
        args.data_type = self.data_type

        self.noise_type = normalize_noise_type(args.noise_type)
        args.noise_type = self.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio  # dùng trong summary (trước đây thiếu -> AttributeError)
        self.attack_noise_std = args.attack_noise_std
        self.poisoned_worker_ids = []
        
        self.initialize_poisoned_workers()
        self.set_clients(args, ClientGH)

        self.global_model = None
        self.CEloss = torch.nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate

        # FedGH uses a shared predictor trained on prototypes
        self.predictor = copy.deepcopy(self.clients[0].model.predictor)
        self.optimizer_predictor = torch.optim.SGD(self.predictor.parameters(), lr=self.server_learning_rate)
        self.batch_size = 8

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset, "Data type: ", self.data_type)
        
        self.train_losses = {i: [] for i in range(self.num_clients)}
        setup_experiment_logging(self, args)
        self.budget = []
        self.model_list = {}
        self.selected_history = {}
        self.best_score = None
        self.best_round = None

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

    def train(self):
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')
            
            self.evaluate()
            self.selected_history[epoch] = [c.id for c in self.selected_clients]
            track_best(self, epoch, self.rs_test_f1_normal[-1], self.predictor, self.clients)

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            self.receive_protos()
            self.train_predictor()

            # Save state: bản sao CPU (predictor sau khi train + tất cả client)
            self.model_list[epoch] = round_models(self.predictor, self.clients)

            self.budget.append(time.time() - s_t)
            print('Time:', round(self.budget[-1], 2))

        # --- FINAL SUMMARY ---
        write_final_summary(self)
        self.save_results()

    def print_final_summary(self):
        write_final_summary(self)
        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train', type=self.data_type)
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test', type=self.data_type)
            client = clientObj(args=args, id=i, poisoned_worker_ids=self.poisoned_worker_ids,
                               train_samples=len(train_data), test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        return list(np.random.choice(self.clients, join_clients, replace=False))

    def send_models(self):
        for client in self.selected_clients:
            client.set_parameters(self.predictor)

    def receive_protos(self):
        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos.append((client.protos[cc], y))

    def train_predictor(self):
        if len(self.uploaded_protos) == 0: return
        proto_loader = torch.utils.data.DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        self.predictor.train()
        for p, y in proto_loader:
            out = self.predictor(p)
            loss = self.CEloss(out, y)
            self.optimizer_predictor.zero_grad()
            loss.backward()
            self.optimizer_predictor.step()

    # def test_metrics(self):
    #     # Stats containers
    #     stats = {
    #         'n': {'acc':[], 'f1':[], 'rec':[], 'pre':[], 'ns':0, 'f1_pc':[], 's_pc':[]},
    #         'f': {'acc':[], 'f1':[], 'rec':[], 'pre':[], 'ns':0, 'f1_pc':[], 's_pc':[]}
    #     }

    #     print('Performance Metrics')
    #     for c in self.selected_clients:
    #         # Client must return: acc, pre, rec, f1, ns, f1_pc, s_pc
    #         acc, pre, rec, f1, ns, f1_pc, s_pc = c.test_metrics()
            
    #         group = 'f' if c.id in self.poisoned_worker_ids else 'n'
    #         stats[group]['acc'].append(acc * ns)
    #         stats[group]['f1'].append(f1 * ns)
    #         stats[group]['rec'].append(rec * ns)
    #         stats[group]['pre'].append(pre * ns)
    #         stats[group]['ns'] += ns
    #         stats[group]['f1_pc'].append(f1_pc * s_pc)
    #         stats[group]['s_pc'].append(s_pc)

    #         log_str = f'\tClient {c.id}: acc {acc:.2f}, f1 {f1:.4f}'
    #         print(log_str)
    #         with open(self.log_file, 'a') as f: f.write(log_str + '\n')

    #     # Aggregate Per-Class with NaN handling
    #     results = []
    #     for g in ['n', 'f']:
    #         if stats[g]['ns'] > 0:
    #             with np.errstate(divide='ignore', invalid='ignore'):
    #                 avg_pc = np.true_divide(np.sum(stats[g]['f1_pc'], axis=0), np.sum(stats[g]['s_pc'], axis=0))
    #                 avg_pc[np.sum(stats[g]['s_pc'], axis=0) == 0] = np.nan
    #             results.extend([stats[g]['acc'], stats[g]['f1'], stats[g]['rec'], stats[g]['pre'], stats[g]['ns'], avg_pc])
    #         else:
    #             results.extend([[], [], [], [], 0, np.full(self.num_classes, np.nan)])
        
    #     return results # Returns 12 elements


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
    def evaluate(self):
        # evaluate train
        stats_train = self.train_metrics()
        # train_metrics trả về (num_normal, loss_normal, num_flip, loss_flip):
        # gộp cả 2 nhóm để tránh ZeroDivisionError khi mọi client được chọn đều bị poison
        n_total = sum(stats_train[0]) + sum(stats_train[2])
        train_loss = (sum(stats_train[1]) + sum(stats_train[3])) * 1.0 / n_total if n_total > 0 else 0.0
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

    def train_metrics(self):
        num_s_n, loss_n, num_s_f, loss_f = [], [], [], []
        for c in self.selected_clients:
            list_loss = c.train_metrics()
            avg_l = sum(list_loss)/len(list_loss)
            self.train_losses[c.id].append(avg_l)
            if c.id in self.poisoned_worker_ids:
                num_s_f.append(len(list_loss)); loss_f.extend(list_loss)
            else:
                num_s_n.append(len(list_loss)); loss_n.extend(list_loss)
        return num_s_n, loss_n, num_s_f, loss_f

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
            'f1-pc-normal': self.rs_test_f1_pc_normal,
            'train-loss': self.rs_train_loss,
            'loss': self.train_losses,
            'poisoned_worker_ids': self.poisoned_worker_ids,
            'selected_clients': self.selected_history,
            'best_round': self.best_round,
            'best_f1_normal': self.best_score,
        }
        save_pickle(metrics_results, os.path.join(self.result_path, f'metrics_results{run_suffix(self.times)}.pkl'))
        # FedGH: "global" là predictor dùng chung
        save_model_artifacts(self, self.predictor, self.model_list)
