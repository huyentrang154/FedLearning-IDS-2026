import copy
import numpy as np
import torch
import os
import pickle
import time
from flcore.clients.clientRep import ClientRep
from utils.data_utils import read_client_data, resolve_data_type
from flcore.servers.model_saver import (round_models, track_best, save_model_artifacts,
                                        save_pickle, mean_round_time, run_suffix,
                                        normalize_noise_type, select_poisoned_workers,
                                        setup_experiment_logging, write_final_summary)

class FedRep(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model.encoder) # FedRep aggregates Encoder only
        self.num_clients = args.num_clients
        self.num_classes = args.num_classes
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.clients = []
        self.selected_clients = []
        self.algo = args.algorithm
        
        # Metrics storage
        self.rs_train_loss = []
        self.rs_test_acc_normal = []
        self.rs_test_acc_flipping = []
        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []
        self.rs_test_f1_pc_normal = []

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
        
        self.set_clients(args, ClientRep)
        
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

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            # Lấy size thực tế để tránh ZeroDivisionError
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train', type=self.data_type)
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test', type=self.data_type)
            
            client = clientObj(args, i, self.poisoned_worker_ids, len(train_data), len(test_data))
            self.clients.append(client)

    def train(self):
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = np.random.choice(self.clients, self.join_clients, replace=False)
            
            for client in self.selected_clients:
                client.set_parameters(self.global_model)

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')
            self.evaluate()
            self.selected_history[epoch] = [c.id for c in self.selected_clients]
            track_best(self, epoch, self.rs_test_f1_normal[-1], self.global_model, self.clients)

            for client in self.selected_clients:
                client.train()

            self.aggregate_parameters()
            self.model_list[epoch] = round_models(self.global_model, self.clients)
             # 5. Dọn dẹp bộ nhớ quan trọng
            self.budget.append(time.time() - s_t)
            print('Time:', self.budget[-1])

        write_final_summary(self)
        self.save_results()

    def aggregate_parameters(self):
        active_samples = sum([c.train_samples for c in self.selected_clients])
        if active_samples == 0: return # Phòng ngừa

        new_params = copy.deepcopy(self.global_model.state_dict())
        for key in new_params.keys():
            new_params[key] = torch.zeros_like(new_params[key])
            for client in self.selected_clients:
                w = client.train_samples / active_samples
                new_params[key] += client.model.encoder.state_dict()[key] * w
        self.global_model.load_state_dict(new_params)

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
        num_samples, losses = [], []
        print("Train Loss")
        for c in self.selected_clients:
            list_loss = c.train_metrics()
            loss = sum(list_loss) / len(list_loss)
            self.train_losses[c.id].append(loss)
            print(f'\tClient {c.id}: Train loss: {round(loss, 2)}')
            num_samples.append(len(list_loss))
            losses.extend(list_loss)
        return num_samples, losses
    # def evaluate(self, epoch):
    #     # Train Loss
    #     losses = []
    #     for c in self.selected_clients:
    #         losses.extend(c.train_metrics())
    #     avg_loss = np.mean(losses) if losses else 0
    #     self.rs_train_loss.append(avg_loss)

    #     # Test
    #     acc, f1, f1_pc = self.test_metrics()
    #     self.rs_test_acc_normal.append(acc)
    #     self.rs_test_f1_normal.append(f1)
    #     self.rs_test_f1_pc_normal.append(f1_pc)

    #     print(f"Metrics: Acc={acc:.4f}, F1={f1:.4f}")
    #     print(f"Per-Class F1: {np.round(f1_pc, 3).tolist()}")

    #     with open(self.log_file, 'a') as f:
    #         f.write(f"Round {epoch}: Acc={acc:.4f}, F1={f1:.4f}, PC-F1={f1_pc.tolist()}\n")
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
            'train_loss': self.rs_train_loss,
            'train-loss': self.rs_train_loss,
            'acc': self.rs_test_acc_normal,
            'f1': self.rs_test_f1_normal,
            'f1_pc': self.rs_test_f1_pc_normal,
            'acc-normal': self.rs_test_acc_normal,
            'f1-normal': self.rs_test_f1_normal,
            'precision-normal': self.rs_test_precision_normal,
            'recall-normal': self.rs_test_recall_normal,
            'acc-flipping': self.rs_test_acc_flipping,
            'f1-flipping': self.rs_test_f1_flipping,
            'precision-flipping': self.rs_test_precision_flipping,
            'recall-flipping': self.rs_test_recall_flipping,
            'loss': self.train_losses,
            'poisoned_worker_ids': self.poisoned_worker_ids,
            'selected_clients': self.selected_history,
            'best_round': self.best_round,
            'best_f1_normal': self.best_score,
        }
        save_pickle(metrics_results, os.path.join(self.result_path, f'metrics_results{run_suffix(self.times)}.pkl'))
        save_model_artifacts(self, self.global_model, self.model_list)
