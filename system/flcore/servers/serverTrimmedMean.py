import copy
import numpy as np
import torch

from flcore.clients.clientTrimmedMean import *
from utils.data_utils import read_client_data

import os
import pickle
import time
import gc


class FedTrimmedMean(object):

    def __init__(self, args, times):

        self.device = args.device
        self.dataset = args.dataset

        self.global_rounds = args.global_rounds

        self.global_model = copy.deepcopy(
            args.model
        )

        self.num_clients = args.num_clients

        self.join_ratio = args.join_ratio

        self.random_join_ratio = (
            args.random_join_ratio
        )

        self.join_clients = int(
            self.num_clients * self.join_ratio
        )

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # =====================================
        # VISUALIZATION
        # =====================================
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

        # =====================================
        # TRIMMED MEAN
        # =====================================
        self.trim_ratio = getattr(
            args,
            "trim_ratio",
            0.2
        )

        # =====================================
        # POISONING
        # =====================================
        self.times = times

        self.noise_type = args.noise_type

        self.num_of_poisoned_workers = (
            args.num_of_poisoned_workers
        )

        self.poisoned_sample_ratio = (
            args.poisoned_sample_ratio
        )

        self.attack_noise_std = (
            args.attack_noise_std
        )

        self.poisoned_worker_ids = []

        # =====================================
        # LABEL FLIPPING
        # =====================================
        self.initialize_poisoned_workers()

        # =====================================
        # CREATE CLIENTS
        # =====================================
        self.set_clients(
            args,
            ClientTrimmedMean
        )

        print(
            f"\nJoin ratio / total clients: "
            f"{self.join_ratio} / "
            f"{self.num_clients}"
        )

        print(
            "\nAlgorithm: ",
            self.algo,
            "Dataset: ",
            self.dataset
        )

        print(
            "Finished creating server and clients."
        )

        # =====================================
        # SAVE LOSS
        # =====================================
        self.train_losses = {
            i: []
            for i in range(self.num_clients)
        }

        # =====================================
        # RESULT PATH
        # =====================================
        self.data_type = args.data_type

        self.result_path = os.path.join(
            '..',
            'results',
            self.dataset,
            self.data_type,
            self.algo,
            time.strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(
            self.result_path,
            exist_ok=True
        )

        self.log_file = (
            f"{self.result_path}/tracking.log"
        )

        with open(self.log_file, 'a') as f:

            f.write(
                f"\nAlogrithm: {self.algo}"
                f"\nDataset: {self.dataset}\n"
            )

        self.budget = []

        # =====================================
        # SAVE MODEL
        # =====================================
        self.model_list = {}

    # =====================================================
    # INITIALIZE POISONED WORKERS
    # =====================================================
    def initialize_poisoned_workers(self):

        if (
            self.noise_type is not None
            and self.num_of_poisoned_workers > 0
        ):

            self.poisoned_worker_ids = (
                np.random.choice(
                    range(self.num_clients),
                    size=int(
                        self.num_of_poisoned_workers
                        * self.num_clients
                    ),
                    replace=False
                ).tolist()
            )

            print(
                f"Poisoned Worker IDs: "
                f"{self.poisoned_worker_ids}"
            )

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self):

        for epoch in range(self.global_rounds):

            start = time.time()

            self.selected_clients = (
                self.select_clients()
            )

            self.send_models()

            print(
                f"\n-------------"
                f"Round number: {epoch}"
                f"-------------"
            )

            with open(self.log_file, 'a') as f:

                f.write(
                    f'-------------'
                    f'Round number: {epoch}'
                    f'-------------\n'
                )

            print("\nEvaluate global model")

            self.evaluate()

            # =================================
            # LOCAL TRAINING
            # =================================
            for client in self.selected_clients:
                client.train()

            # =================================
            # COLLECT MODELS
            # =================================
            self.uploaded_models = []

            for c in self.selected_clients:

                state_dict = {

                    k: v.detach().cpu()

                    for k, v
                    in c.model.state_dict().items()
                }

                self.uploaded_models.append(
                    state_dict
                )

            # =================================
            # AGGREGATION
            # =================================
            self.aggregate_parameters()

            # =================================
            # SAVE MODEL
            # =================================
            self.model_list.update({
                epoch: self.uploaded_models
            })

            # =================================
            # CLEANUP
            # =================================
            gc.collect()

            torch.cuda.empty_cache()

            self.budget.append(
                time.time() - start
            )

            print(
                'Time:',
                self.budget[-1]
            )

        # =================================================
        # FINAL RESULTS
        # =================================================
        print(
            "\nAlgorithm: ",
            self.algo,
            "Dataset: ",
            self.dataset
        )

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
            sum(self.budget[1:])
            / len(self.budget[1:])
        )

        summary_lines = [

            f"\nAlgorithm: "
            f"{self.algo} | "
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

    # =====================================================
    # SET CLIENTS
    # =====================================================
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

    # =====================================================
    # SELECT CLIENTS
    # =====================================================
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

    # =====================================================
    # SEND MODEL
    # =====================================================
    def send_models(self):

        assert (len(self.clients) > 0)

        for client in self.selected_clients:

            client.set_parameters(
                self.global_model
            )

    # =====================================================
    # TRIMMED MEAN AGGREGATION
    # =====================================================
    def aggregate_parameters(self):

        assert (
            len(self.uploaded_models) > 0
        )

        global_dict = (
            self.global_model.state_dict()
        )

        num_clients = len(
            self.uploaded_models
        )

        trim_num = int(
            self.trim_ratio * num_clients
        )

        for key in global_dict.keys():

            if not torch.is_floating_point(
                global_dict[key]
            ):
                continue

            # ================================
            # STACK PARAMETERS
            # ================================
            stacked = torch.stack([

                client[key].float()

                for client
                in self.uploaded_models

            ], dim=0)

            # ================================
            # SORT
            # ================================
            sorted_tensor, _ = torch.sort(
                stacked,
                dim=0
            )

            # ================================
            # TRIM EXTREME VALUES
            # ================================
            if (
                2 * trim_num < num_clients
            ):

                trimmed_tensor = sorted_tensor[
                    trim_num:
                    num_clients - trim_num
                ]

            else:

                trimmed_tensor = sorted_tensor

            # ================================
            # COMPUTE MEAN
            # ================================
            trimmed_mean = torch.mean(
                trimmed_tensor,
                dim=0
            )

            global_dict[key] = trimmed_mean

        self.global_model.load_state_dict(
            global_dict
        )

    # =====================================================
    # TEST METRICS
    # =====================================================
    def test_metrics(self):

        # ===== NORMAL =====
        total_acc_normal = []
        total_precision_normal = []
        total_recall_normal = []
        total_f1_normal = []

        num_samples_normal = 0

        # ===== FLIPPING =====
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

            # =================================
            # FLIPPING
            # =================================
            if c.id in self.poisoned_worker_ids:

                total_acc_flipping.append(
                    acc * ns
                )

                total_recall_flipping.append(
                    recall * ns
                )

                total_f1_flipping.append(
                    f1 * ns
                )

                total_precision_flipping.append(
                    precision * ns
                )

                num_samples_flipping += ns

            # =================================
            # NORMAL
            # =================================
            else:

                total_acc_normal.append(
                    acc * ns
                )

                total_recall_normal.append(
                    recall * ns
                )

                total_f1_normal.append(
                    f1 * ns
                )

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

    # =====================================================
    # TRAIN METRICS
    # =====================================================
    def train_metrics(self):

        num_samples = []
        losses = []

        print("Train Loss")

        with open(self.log_file, 'a') as f:
            f.write('Train Loss\n')

        for c in self.selected_clients:

            list_loss = c.train_metrics()

            loss = (
                sum(list_loss)
                / len(list_loss)
            )

            self.train_losses[c.id].append(
                loss
            )

            print(
                f'\tClient {c.id}: '
                f'Train loss: {round(loss, 2)}'
            )

            with open(self.log_file, 'a') as f:

                f.write(
                    f'\tClient {c.id}: '
                    f'Train loss: '
                    f'{round(loss, 2)}\n'
                )

            num_samples.append(
                len(list_loss)
            )

            losses.extend(list_loss)

        return num_samples, losses

    # =====================================================
    # EVALUATE
    # =====================================================
    def evaluate(self):

        # =====================================
        # TRAIN
        # =====================================
        stats_train = self.train_metrics()

        train_loss = (
            sum(stats_train[1])
            * 1.0
            / sum(stats_train[0])
        )

        self.rs_train_loss.append(
            train_loss
        )

        # =====================================
        # TEST
        # =====================================
        stats_test = self.test_metrics()

        # ===== NORMAL =====
        test_acc_normal = (
            sum(stats_test[0])
            / stats_test[4]
        )

        test_f1_normal = (
            sum(stats_test[1])
            / stats_test[4]
        )

        test_recall_normal = (
            sum(stats_test[2])
            / stats_test[4]
        )

        test_precision_normal = (
            sum(stats_test[3])
            / stats_test[4]
        )

        # ===== FLIPPING =====
        if stats_test[9] > 0:

            test_acc_flipping = (
                sum(stats_test[5])
                / stats_test[9]
            )

            test_f1_flipping = (
                sum(stats_test[6])
                / stats_test[9]
            )

            test_recall_flipping = (
                sum(stats_test[7])
                / stats_test[9]
            )

            test_precision_flipping = (
                sum(stats_test[8])
                / stats_test[9]
            )

        else:

            test_acc_flipping = 0
            test_f1_flipping = 0
            test_recall_flipping = 0
            test_precision_flipping = 0

        # =====================================
        # SAVE NORMAL
        # =====================================
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

        # =====================================
        # SAVE FLIPPING
        # =====================================
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

        print(
            "Overall Performance Metrics"
        )

        print(
            "\ttrain loss avg: "
            "{:.4f}".format(train_loss)
        )

        print(
            "\ttest acc normal avg: "
            "{:.4f}".format(test_acc_normal)
        )

        print(
            "\ttest acc flipping avg: "
            "{:.4f}".format(test_acc_flipping)
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

    # =====================================================
    # SAVE RESULTS
    # =====================================================
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

            pickle.dump(
                metrics_results,
                f
            )

        # =====================================
        # SAVE MODELS
        # =====================================
        with open(
            f'{self.result_path}/models.pkl',
            'wb'
        ) as f:

            pickle.dump(
                self.model_list,
                f
            )