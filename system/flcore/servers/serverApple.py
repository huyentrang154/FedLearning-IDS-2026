import copy
import numpy as np
import torch
from flcore.clients.clientApple import *
from utils.data_utils import read_client_data
import os
import pickle
import time


class APPLE(object):
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
        self.rs_test_acc = []
        # self.rs_test_auc = []
        self.rs_test_f1 = []
        self.rs_test_precision = []
        self.rs_test_recall = []

        self.times = times
        # self.nc = args.num_classes
        self.set_clients(args, ClientApple)

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

        self.client_models = [c.model_c for c in self.clients]

        train_samples = 0
        for client in self.clients:
            train_samples += client.train_samples
        p0 = [client.train_samples / train_samples for client in self.clients]

        for c in self.clients:
            c.p0 = p0

    def train(self):
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
                client.train(epoch)

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
        print("\tacc:", max(self.rs_test_acc))
        # print("\tauc:", max(self.rs_test_auc))
        print("\tf1:", max(self.rs_test_f1))
        print("\trecall:", max(self.rs_test_recall))
        print("\tprecision:", max(self.rs_test_precision))
        print("\tTime:", sum(self.budget[1:]) / len(self.budget[1:]))

        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, data_type='train')
            test_data = read_client_data(self.dataset, i, data_type='test')
            client = clientObj(args=args,
                               id=i,
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
        self.client_models = [c.model_c for c in self.clients]
        for client in self.selected_clients:
            # client.set_parameters(self.global_model)
            client.set_models(self.client_models)

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
        total_acc = []
        total_precision = []
        total_recall = []
        total_f1 = []
        # total_auc = []
        num_samples = 0

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

            total_acc.append(acc * ns)
            # total_auc.append(auc * ns)
            total_recall.append(recall * ns)
            total_f1.append(f1 * ns)
            total_precision.append(precision * ns)
            num_samples += ns

        return total_acc, total_f1, total_recall, total_precision, num_samples

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

        test_acc = sum(stats_test[0]) / stats_test[4]
        test_f1 = sum(stats_test[1]) / stats_test[4]
        test_recall = sum(stats_test[2]) / stats_test[4]
        test_precision = sum(stats_test[3]) / stats_test[4]

        self.rs_test_acc.append(test_acc)
        # self.rs_test_auc.append(test_auc)
        self.rs_test_f1.append(test_f1)
        self.rs_test_precision.append(test_precision)
        self.rs_test_recall.append(test_recall)

        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("Overall Performance Metrics")
        print("\ttrain loss avg: {:.4f}".format(train_loss))
        print("\ttest acc avg: {:.4f}".format(test_acc))
        # print("\ttest auc avg: {:.4f}".format(test_auc))
        print("\ttest f1 avg: {:.4f}".format(test_f1))
        print("\ttest recall avg: {:.4f}".format(test_recall))
        print("\ttest precision avg: {:.4f}".format(test_precision))

        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))
            f.write("\ttest acc avg: {:.4f}\n".format(test_acc))
            # f.write("\ttest auc avg: {:.4f}\n".format(test_auc))
            f.write("\ttest f1 avg: {:.4f}\n".format(test_f1))
            f.write("\ttest recall avg: {:.4f}\n".format(test_recall))
            f.write("\ttest precision avg: {:.4f}\n".format(test_precision))

    def save_results(self):
        metrics_results = {
            'acc': self.rs_test_acc,
            # 'auc': self.rs_test_auc,
            'f1': self.rs_test_f1,
            'recall': self.rs_test_recall,
            'precision': self.rs_test_precision,
            'loss': self.train_losses
        }

        with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)

        # save checkpoint of global, client 0, client 1, ... after each commnunication round
        with open(f'{self.result_path}/models.pkl', 'wb') as f:
            pickle.dump(self.model_list, f)
