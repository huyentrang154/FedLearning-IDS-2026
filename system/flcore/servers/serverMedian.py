import copy
import numpy as np
import torch
import time
import os
import pickle
import gc

from flcore.clients.clientMedian import ClientMedian
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader


class FedMedian(object):
    def __init__(self, args, times):
        # ===== BASIC =====
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

        # ===== STORAGE =====
        self.uploaded_models = []
        self.uploaded_ids = []

        # ===== METRICS =====
        self.rs_train_loss = []
        self.rs_test_acc = []
        self.rs_test_f1 = []
        self.rs_test_precision = []
        self.rs_test_recall = []
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

        # ===== POISON =====
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_worker_ids = []

        # ===== INIT =====
        self.times = times
        self.initialize_poisoned_workers()
        self.set_clients(args, ClientMedian)

        self.full_test_data = read_client_data(self.dataset, -1, self.num_clients, data_type='test')
        self.full_test_loader = DataLoader(self.full_test_data, batch_size=args.batch_size, shuffle=False)

        # ===== LOG =====
        self.result_path = os.path.join('..', 'results', 'Median', self.dataset,
                                        time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.result_path, exist_ok=True)

        self.log_file = f"{self.result_path}/log.txt"

        print("\nAlgorithm: Median Aggregation")
        print("Clients:", self.num_clients)

    # ================= CLIENT =================
    def initialize_poisoned_workers(self):
        if self.noise_type is not None and self.num_of_poisoned_workers > 0:
            self.poisoned_worker_ids = np.random.choice(
                range(self.num_clients),
                size=int(self.num_of_poisoned_workers*self.num_clients),
                replace=False
            ).tolist()
            print(f"Poisoned Worker IDs: {self.poisoned_worker_ids}")
    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train')
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test')

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
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1)[0]
        else:
            join_clients = self.join_clients

        return list(np.random.choice(self.clients, join_clients, replace=False))

    def send_models(self):
        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    # ================= CORE: MEDIAN =================
    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():

            # chỉ xử lý weight float
            if not torch.is_floating_point(global_dict[key]):
                continue

            stacked = torch.stack(
                [client[key].float() for client in self.uploaded_models],
                dim=0
            )

            median_param = torch.median(stacked, dim=0).values
            global_dict[key] = median_param

        self.global_model.load_state_dict(global_dict)

    # ================= TRAIN =================
    def train(self):
        self.budget = []

        for round in range(self.global_rounds):
            start_time = time.time()

            print(f"\n------------- Round {round} -------------")

            self.selected_clients = self.select_clients()
            self.send_models()

            # ===== LOCAL TRAIN =====
            for client in self.selected_clients:
                client.train()
                torch.cuda.empty_cache()

            # ===== COLLECT MODELS =====
            self.uploaded_models = []
            self.uploaded_ids = []

            for c in self.selected_clients:
                state_dict = {
                    k: v.detach().cpu() for k, v in c.model.state_dict().items()
                }
                self.uploaded_models.append(state_dict)
                self.uploaded_ids.append(c.id)

            # ===== AGGREGATE =====
            self.aggregate_parameters()

            # ===== EVALUATE =====
            self.evaluate()

            # ===== CLEAN MEMORY =====
            del self.uploaded_models
            gc.collect()
            torch.cuda.empty_cache()

            self.budget.append(time.time() - start_time)
            print("Time:", self.budget[-1])

# --- FINAL SUMMARY ---
        # self.print_final_summary()
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

        summary_lines = [
            f"\nAlgorithm: {self.algo} | Dataset: {self.dataset}\n",
            "\nBest performance metrics\n",
            f"\tacc-flipping: {max(self.rs_test_acc_flipping):.4f}\n",
            f"\tacc-normal: {max(self.rs_test_acc_normal):.4f}\n",
            f"\tf1-flipping: {max(self.rs_test_f1_flipping):.4f}\n",
            f"\tf1-normal: {max(self.rs_test_f1_normal):.4f}\n",
            f"\trecall-flipping: {max(self.rs_test_recall_flipping):.4f}\n",
            f"\trecall-normal: {max(self.rs_test_recall_normal):.4f}\n",
            f"\tprecision-flipping: {max(self.rs_test_precision_flipping):.4f}\n",
            f"\tprecision-normal: {max(self.rs_test_precision_normal):.4f}\n",
            f"\tAvg Time/Round: {sum(self.budget[1:]) / len(self.budget[1:]):.4f} sec\n"
        ]

        for line in summary_lines:
            print(line.strip())
            with open(self.log_file, 'a') as f:
                f.write(line)
        self.save_results()

    # ================= METRICS =================
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


        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))

    # def evaluate(self, epoch):
    #     stats_train = self.train_metrics()
    #     train_loss = sum(stats_train[1]) / sum(stats_train[0]) if sum(stats_train[0]) > 0 else 0
    #     self.rs_train_loss.append(train_loss)

    #     res = self.test_metrics()
        
    #     # Normal calculation
    #     acc_n = sum(res[0]) / res[4] if res[4] > 0 else 0
    #     f1_n = sum(res[1]) / res[4] if res[4] > 0 else 0
    #     self.rs_test_acc_normal.append(acc_n)
    #     self.rs_test_f1_normal.append(f1_n)
    #     self.rs_test_f1_pc_normal.append(res[10])

    #     # Update best F1 PC
    #     if f1_n >= max(self.rs_test_f1_normal):
    #         self.best_f1_pc_normal = res[10]

    #     # Flipping calculation
    #     acc_f, f1_f = 0, 0
    #     if res[9] > 0:
    #         acc_f = sum(res[5]) / res[9]
    #         f1_f = sum(res[6]) / res[9]
    #         self.rs_test_acc_flipping.append(acc_f)
    #         self.rs_test_f1_flipping.append(f1_f)
    #         self.rs_test_f1_pc_flipping.append(res[11])

    #     # Display
    #     def fmt(v): return "[" + ", ".join([f"{x:.4f}" for x in v]) + "]"
    #     f1_pc_str = fmt(res[10])
        
    #     print(f"\nAlgorithm: {self.algo} Dataset: {self.dataset} Epoch: {epoch}")
    #     print(f"\ttrain loss avg: {train_loss:.4f}")
    #     print(f"\ttest acc normal avg: {acc_n:.4f} | f1 normal avg: {f1_n:.4f}")
    #     print(f"\tNormal F1 Per-Class: {f1_pc_str}")

    #     with open(self.log_file, 'a') as f:
    #         f.write(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc_N={acc_n:.4f}, F1_N={f1_n:.4f}\n")
    #         f.write(f"F1_PerClass_N: {f1_pc_str}\n")

    def train_metrics(self):
        num_samples, losses = [], []
        print("Train Loss")
        for c in self.selected_clients:
            list_loss = c.train_metrics()
            loss = sum(list_loss) / len(list_loss)
            # self.train_losses[c.id].append(loss)
            print(f'\tClient {c.id}: Train loss: {round(loss, 2)}')
            num_samples.append(len(list_loss))
            losses.extend(list_loss)
        return num_samples, losses

    def save_results(self):
        # Đã cập nhật để lưu thêm f1-per-class vào file pkl
        metrics_results = {
            'acc-normal': self.rs_test_acc_normal,
            'f1-normal': self.rs_test_f1_normal,
            'f1-per-class-normal': self.rs_test_f1_pc_normal,
            'train-loss': self.rs_train_loss
        }
        with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)
        with open(f'{self.result_path}/models.pkl', 'wb') as f:
            pickle.dump(self.model_list, f)
