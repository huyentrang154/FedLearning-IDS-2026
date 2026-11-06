# flcore/servers/serverFLTrust.py

import copy
import numpy as np
import torch
from flcore.clients.clientFLTrust import *
from utils.data_utils import read_client_data
import os
import pickle
import time
import gc


class FLTrust(object):

    def __init__(self, args, times):

        self.device = args.device
        self.dataset = args.dataset

        self.global_rounds = args.global_rounds

        self.global_model = copy.deepcopy(args.model)

        self.num_clients = args.num_clients

        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio

        self.join_clients = int(
            self.num_clients * self.join_ratio
        )

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.algo = args.algorithm

        self.rs_train_loss = []

        self.rs_test_acc_normal = []
        self.rs_test_acc_flipping = []

        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []

        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []

        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []

        self.times = times

        self.noise_type = args.noise_type

        self.num_of_poisoned_workers = (
            args.num_of_poisoned_workers
        )

        self.poisoned_sample_ratio = (
            args.poisoned_sample_ratio
        )

        self.attack_noise_std = args.attack_noise_std

        self.poisoned_worker_ids = []

        self.initialize_poisoned_workers()

        self.set_clients(args, ClientFLTrust)

        print(
            f"\nJoin ratio / total clients: "
            f"{self.join_ratio} / {self.num_clients}"
        )

        print(
            "\nAlgorithm: ",
            self.algo,
            "Dataset: ",
            self.dataset
        )

        print("Finished creating server and clients.")

        self.train_losses = {
            i: [] for i in range(self.num_clients)
        }

        self.data_type = args.data_type

        self.result_path = os.path.join(
            '..',
            'results',
            self.dataset,
            self.data_type,
            self.algo,
            time.strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(self.result_path, exist_ok=True)

        self.log_file = f"{self.result_path}/tracking.log"

        with open(self.log_file, 'a') as f:

            f.write(
                f"\nAlgorithm: {self.algo}"
                f"\nDataset: {self.dataset}\n"
            )

        self.budget = []

        self.model_list = {}

    def initialize_poisoned_workers(self):

        if (
            self.noise_type is not None
            and self.num_of_poisoned_workers > 0
        ):

            self.poisoned_worker_ids = np.random.choice(
                range(self.num_clients),
                size=int(
                    self.num_of_poisoned_workers *
                    self.num_clients
                ),
                replace=False
            ).tolist()

            print(
                f"Poisoned Worker IDs: "
                f"{self.poisoned_worker_ids}"
            )

    def train(self):

        for epoch in range(self.global_rounds):

            start = time.time()

            self.selected_clients = self.select_clients()

            self.send_models()

            print(
                f"\n-------------Round number: "
                f"{epoch}-------------"
            )

            with open(self.log_file, 'a') as f:

                f.write(
                    f'-------------Round number: '
                    f'{epoch}-------------\n'
                )

            print("\nEvaluate global model")

            self.evaluate()

            # local training
            for client in self.selected_clients:

                client.train()

            # collect models
            self.uploaded_models = []

            for c in self.selected_clients:

                state_dict = {
                    k: v.detach().cpu()
                    for k, v in c.model.state_dict().items()
                }

                self.uploaded_models.append(state_dict)

            # aggregation
            self.aggregate_parameters()

            self.budget.append(time.time() - start)

            print("Time cost:", self.budget[-1])

            gc.collect()

            torch.cuda.empty_cache()

        print("\nAlgorithm: ", self.algo,
              "Dataset: ", self.dataset)

        print("\nBest performance metrics")

        print(
            "\tacc-flipping:",
            max(self.rs_test_acc_flipping)
        )

        print(
            "\tacc-normal:",
            max(self.rs_test_acc_normal)
        )

        print(
            "\tf1-flipping:",
            max(self.rs_test_f1_flipping)
        )

        print(
            "\tf1-normal:",
            max(self.rs_test_f1_normal)
        )

        print(
            "\trecall-flipping:",
            max(self.rs_test_recall_flipping)
        )

        print(
            "\trecall-normal:",
            max(self.rs_test_recall_normal)
        )

        print(
            "\tprecision-flipping:",
            max(self.rs_test_precision_flipping)
        )

        print(
            "\tprecision-normal:",
            max(self.rs_test_precision_normal)
        )

        print(
            "\tTime:",
            sum(self.budget[1:]) / len(self.budget[1:])
        )

        summary_lines = [

            f"\nAlgorithm: {self.algo} | "
            f"Dataset: {self.dataset}\n",

            "\nBest performance metrics\n",

            f"\tacc-flipping: "
            f"{max(self.rs_test_acc_flipping):.4f}\n",

            f"\tacc-normal: "
            f"{max(self.rs_test_acc_normal):.4f}\n",

            f"\tf1-flipping: "
            f"{max(self.rs_test_f1_flipping):.4f}\n",

            f"\tf1-normal: "
            f"{max(self.rs_test_f1_normal):.4f}\n",

            f"\trecall-flipping: "
            f"{max(self.rs_test_recall_flipping):.4f}\n",

            f"\trecall-normal: "
            f"{max(self.rs_test_recall_normal):.4f}\n",

            f"\tprecision-flipping: "
            f"{max(self.rs_test_precision_flipping):.4f}\n",

            f"\tprecision-normal: "
            f"{max(self.rs_test_precision_normal):.4f}\n",

            f"\tAvg Time/Round: "
            f"{sum(self.budget[1:]) / len(self.budget[1:]):.4f} sec\n"
        ]

        for line in summary_lines:

            print(line.strip())

            with open(self.log_file, 'a') as f:
                f.write(line)

        self.save_results()

    def set_clients(self, args, clientObj):

        for i in range(self.num_clients):

            train_data = read_client_data(
                self.dataset,
                i,
                self.num_clients,
                data_type='train'
            )

            test_data = read_client_data(
                self.dataset,
                i,
                self.num_clients,
                data_type='test'
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
                range(
                    self.join_clients,
                    self.num_clients + 1
                ),
                1,
                replace=False
            )[0]

        else:

            join_clients = self.join_clients

        selected_clients = list(
            np.random.choice(
                self.clients,
                join_clients,
                replace=False
            )
        )

        return selected_clients

    def send_models(self):

        assert len(self.clients) > 0

        for client in self.selected_clients:

            client.set_parameters(self.global_model)

    def receive_models(self):

        assert len(self.selected_clients) > 0

        self.uploaded_models = []

        for client in self.selected_clients:

            state_dict = {
                k: v.detach().cpu()
                for k, v in client.model.state_dict().items()
            }

            self.uploaded_models.append(state_dict)

    def cosine_similarity(self, x, y):

        x = x.flatten().cpu()
        y = y.flatten().cpu()

        return torch.dot(x, y) / (
            torch.norm(x) *
            torch.norm(y) + 1e-12
        )

    def aggregate_parameters(self):

        assert len(self.uploaded_models) > 0

        global_dict = self.global_model.state_dict()

        trust_scores = []

        for client_dict in self.uploaded_models:

            score = 0.0

            for key in global_dict.keys():

                if not torch.is_floating_point(
                        global_dict[key]):
                    continue

                global_param = global_dict[key].float().cpu()

                client_param = client_dict[key].float().cpu()

                sim = self.cosine_similarity(
                    global_param,
                    client_param
                )

                sim = torch.clamp(sim, min=0)

                score += sim.item()

            trust_scores.append(score)

        trust_scores = np.array(trust_scores)

        if trust_scores.sum() == 0:
            trust_scores = np.ones_like(trust_scores)

        trust_scores = trust_scores / trust_scores.sum()

        # aggregate
        for key in global_dict.keys():

            if not torch.is_floating_point(global_dict[key]):
                continue

            agg = torch.zeros_like(
                global_dict[key],
                dtype=torch.float32
            )

            for idx, client_dict in enumerate(
                    self.uploaded_models):

                agg += (
                    client_dict[key]
                    .float()
                    .to(agg.device)
                    * trust_scores[idx]
                )

            global_dict[key] = agg

        self.global_model.load_state_dict(global_dict)

    def test_metrics(self):

        # normal
        total_acc_normal = []
        total_precision_normal = []
        total_recall_normal = []
        total_f1_normal = []

        num_samples_normal = 0

        # flipping
        total_acc_flipping = []
        total_precision_flipping = []
        total_recall_flipping = []
        total_f1_flipping = []

        num_samples_flipping = 0

        print('Performance Metrics')

        with open(self.log_file, 'a') as f:
            f.write('Performance Metrics\n')

        for c in self.selected_clients:

            acc, precision, recall, f1, ns = (
                c.test_metrics()
            )

            print(
                f'\tClient {c.id}: '
                f'acc {round(acc, 2)}, '
                f'precision: {round(precision, 2)}, '
                f'recall: {round(recall, 2)}, '
                f'f1: {round(f1, 2)}'
            )

            with open(self.log_file, 'a') as f:

                f.write(
                    f'\tClient {c.id}: '
                    f'acc {round(acc, 2)}, '
                    f'precision: {round(precision, 2)}, '
                    f'recall: {round(recall, 2)}, '
                    f'f1: {round(f1, 2)}\n'
                )

            if c.id in self.poisoned_worker_ids:

                total_acc_flipping.append(acc * ns)

                total_recall_flipping.append(recall * ns)

                total_f1_flipping.append(f1 * ns)

                total_precision_flipping.append(
                    precision * ns
                )

                num_samples_flipping += ns

            else:

                total_acc_normal.append(acc * ns)

                total_recall_normal.append(recall * ns)

                total_f1_normal.append(f1 * ns)

                total_precision_normal.append(
                    precision * ns
                )

                num_samples_normal += ns

        return (
            total_acc_normal,
            total_f1_normal,
            total_recall_normal,
            total_precision_normal,
            num_samples_normal,

            total_acc_flipping,
            total_f1_flipping,
            total_recall_flipping,
            total_precision_flipping,
            num_samples_flipping
        )

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

            print(
                f'\tClient {c.id}: '
                f'Train loss: {round(loss, 2)}'
            )

            with open(self.log_file, 'a') as f:

                f.write(
                    f'\tClient {c.id}: '
                    f'Train loss: {round(loss, 2)}\n'
                )

            num_samples.append(len(list_loss))

            losses.extend(list_loss)

        return num_samples, losses

    def evaluate(self):

        stats_train = self.train_metrics()

        train_loss = (
            sum(stats_train[1]) * 1.0 /
            sum(stats_train[0])
        )

        self.rs_train_loss.append(train_loss)

        stats_test = self.test_metrics()

        test_acc_normal = (
            sum(stats_test[0]) / stats_test[4]
        )

        test_f1_normal = (
            sum(stats_test[1]) / stats_test[4]
        )

        test_recall_normal = (
            sum(stats_test[2]) / stats_test[4]
        )

        test_precision_normal = (
            sum(stats_test[3]) / stats_test[4]
        )

        if stats_test[9] > 0:

            test_acc_flipping = (
                sum(stats_test[5]) / stats_test[9]
            )

            test_f1_flipping = (
                sum(stats_test[6]) / stats_test[9]
            )

            test_recall_flipping = (
                sum(stats_test[7]) / stats_test[9]
            )

            test_precision_flipping = (
                sum(stats_test[8]) / stats_test[9]
            )

        else:

            test_acc_flipping = 0
            test_f1_flipping = 0
            test_recall_flipping = 0
            test_precision_flipping = 0

        self.rs_test_acc_normal.append(
            test_acc_normal
        )

        self.rs_test_f1_normal.append(
            test_f1_normal
        )

        self.rs_test_precision_normal.append(
            test_precision_normal
        )

        self.rs_test_recall_normal.append(
            test_recall_normal
        )

        self.rs_test_acc_flipping.append(
            test_acc_flipping
        )

        self.rs_test_f1_flipping.append(
            test_f1_flipping
        )

        self.rs_test_precision_flipping.append(
            test_precision_flipping
        )

        self.rs_test_recall_flipping.append(
            test_recall_flipping
        )

        print(
            "\nAlgorithm: ",
            self.algo,
            "Dataset: ",
            self.dataset
        )

        print("Overall Performance Metrics")

        print(
            "\ttrain loss avg: {:.4f}".format(
                train_loss
            )
        )

        print(
            "\ttest acc normal avg: {:.4f}".format(
                test_acc_normal
            )
        )

        print(
            "\ttest acc flipping avg: {:.4f}".format(
                test_acc_flipping
            )
        )

        with open(self.log_file, 'a') as f:

            f.write(
                "\nAlgorithm: {}, Dataset: {}\n".format(
                    self.algo,
                    self.dataset
                )
            )

            f.write(
                "Overall Performance Metrics\n"
            )

            f.write(
                "\ttrain loss avg: {:.4f}\n".format(
                    train_loss
                )
            )

    def save_results(self):

        metrics_results = {

            'acc-flipping':
                self.rs_test_acc_flipping,

            'acc-normal':
                self.rs_test_acc_normal,

            'f1-flipping':
                self.rs_test_f1_flipping,

            'f1-normal':
                self.rs_test_f1_normal,

            'recall-flipping':
                self.rs_test_recall_flipping,

            'recall-normal':
                self.rs_test_recall_normal,

            'precision-flipping':
                self.rs_test_precision_flipping,

            'precision-normal':
                self.rs_test_precision_normal,

            'train-loss':
                self.rs_train_loss
        }

        with open(
            f'{self.result_path}/metrics_results.pkl',
            'wb'
        ) as f:

            pickle.dump(metrics_results, f)

        with open(
            f'{self.result_path}/models.pkl',
            'wb'
        ) as f:

            pickle.dump(self.model_list, f)