import copy
import numpy as np
import torch
from flcore.clients.clientALA import *
from utils.data_utils import read_client_data
import os
import pickle
import time
import random


class FedALA(object):
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

        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.poisoned_worker_ids = []
        # self.nc = args.num_classes
        self.set_clients(args, ClientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("Finished creating server and clients.")

        # save loss
        self.train_losses = {i: [] for i in range(self.num_clients)}
        # set path to save result
        self.data_type = args.data_type
        self.result_path = os.path.join('..', 'results', self.dataset, self.data_type, self.algo, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.result_path, exist_ok=True)

        self.log_file = f"{self.result_path}/tracking.log"
        with open(self.log_file, 'a') as f:
            f.write(f"\nAlogrithm: {self.algo} \nDataset: {self.dataset}\n")
        self.budget = []

        # for get model in each round
        self.model_list = {}

    def initialize_poisoned_workers(self):
        """Select poisoned workers based on the given parameters."""
        if self.noise_type is not None and self.num_of_poisoned_workers > 0:
            self.poisoned_worker_ids = np.random.choice(
                range(self.num_clients),
                size=int(self.num_of_poisoned_workers*self.num_clients),
                replace=False
            ).tolist()
            print(f"Poisoned Worker IDs: {self.poisoned_worker_ids}")
    
    def train(self):
        #Label flipping
        self.initialize_poisoned_workers()
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')
            print("\nEvaluate global model")
            self.evaluate()

            for client in self.selected_clients:
                client.train()
            # Log poisoned clients and poisoned samples
            poisoned_clients = [c.id for c in self.selected_clients if c.id in self.poisoned_worker_ids]
            print(f"Poisoned clients in this round: {poisoned_clients}")
            with open(self.log_file, 'a') as f:
                f.write(f"Poisoned clients in this round: {poisoned_clients}\n")
            # save state_dict of model of each clients
            models = [self.global_model]
            for c in self.clients:
                models.append(c.model)
            self.model_list.update({epoch: models})

            self.receive_models()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('Time:', self.budget[-1])

        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("\nBest performance metrics")
        print("\tacc-flipping:", max(self.rs_test_acc_flipping))
        print("\tacc-normal:", max(self.rs_test_acc_normal))
        # print("\tauc:", max(self.rs_test_auc))
        print("\tf1-flipping:", max(self.rs_test_f1_flipping))
        print("\tf1-normal:", max(self.rs_test_f1_normal))
        print("\trecall-flipping:", max(self.rs_test_recall_flipping))
        print("\trecall-normal:", max(self.rs_test_recall_normal))
        print("\tprecision-flipping:", max(self.rs_test_precision_flipping))
        print("\tprecision-normal:", max(self.rs_test_precision_normal))
        print("\tTime:", sum(self.budget[1:]) / len(self.budget[1:]))
        
        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
            f.write("\nBest performance metrics")
            f.write("\tacc-flipping:", max(self.rs_test_acc_flipping))
            f.write("\tacc-normal:", max(self.rs_test_acc_normal))
            # print("\tauc:", max(self.rs_test_auc))
            f.write("\tf1-flipping:", max(self.rs_test_f1_flipping))
            f.write("\tf1-normal:", max(self.rs_test_f1_normal))
            f.write("\trecall-flipping:", max(self.rs_test_recall_flipping))
            f.write("\trecall-normal:", max(self.rs_test_recall_normal))
            f.write("\tprecision-flipping:", max(self.rs_test_precision_flipping))
            f.write("\tprecision-normal:", max(self.rs_test_precision_normal))
        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i,self.num_clients,  data_type='train')
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test')
            client = clientObj(args=args,
                               id=i,
                               poisoned_worker_ids = self.poisoned_worker_ids,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.selected_clients:
            # client.set_parameters(self.global_model)
            client.local_initialization(self.global_model)

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
            self.uploaded_models.append(client.model)

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

        test_acc_normal = sum(stats_test[0]) / stats_test[4]
        test_f1_normal = sum(stats_test[1]) / stats_test[4]
        test_recall_normal = sum(stats_test[2]) / stats_test[4]
        test_precision_normal = sum(stats_test[3]) / stats_test[4]

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
        print("\ttest acc normal avg: {:.4f}".format(test_acc_normal))
        print("\ttest acc flipping avg: {:.4f}".format(test_acc_flipping))
        # # print("\ttest auc avg: {:.4f}".format(test_auc))
        # print("\ttest f1 avg: {:.4f}".format(test_f1))
        # print("\ttest recall avg: {:.4f}".format(test_recall))
        # print("\ttest precision avg: {:.4f}".format(test_precision))

        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))
            # f.write("\ttest acc avg: {:.4f}\n".format(test_acc))
            # # f.write("\ttest auc avg: {:.4f}\n".format(test_auc))
            # f.write("\ttest f1 avg: {:.4f}\n".format(test_f1))
            # f.write("\ttest recall avg: {:.4f}\n".format(test_recall))
            # f.write("\ttest precision avg: {:.4f}\n".format(test_precision))

    def save_results(self):
        metrics_results = {
            'acc-flipping': self.rs_test_acc_flipping,
            'acc-normal': self.rs_test_acc_normal,
            'f1-flipping': self.rs_test_f1_flipping,
            'f1-normal': self.rs_test_f1_normal,
            'recall-flipping': self.rs_test_recall_flipping,
            'precision-flipping': self.rs_test_precision_flipping,
            'precision-normal': self.rs_test_precision_normal,
        }

        with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)

        # save checkpoint of global, client 0, client 1, ... after each commnunication round
        with open(f'{self.result_path}/models.pkl', 'wb') as f:
            pickle.dump(self.model_list, f)
