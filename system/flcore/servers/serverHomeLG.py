import copy
import numpy as np
import torch
from flcore.clients.clientHomeLG import *
from utils.data_utils import read_client_data
import os
import pickle
import time
import matplotlib.pyplot as plt

class FedHomeLG(object):
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
        self.uploaded_models = []

        # Tracking metrics
        self.algo = args.algorithm
        self.rs_train_loss = []
        self.ce_curve = [] 
        self.re_curve = [] 
        
        self.rs_test_acc_normal = []
        self.rs_test_acc_flipping = []
        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []

        # --- Per-Class F1 history ---
        self.num_classes = args.num_classes
        self.rs_test_f1_pc_normal = [] 
        self.rs_test_f1_pc_flipping = []
        self.best_f1_pc_normal = None 

        self.times = times
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_worker_ids = []
        
        self.initialize_poisoned_workers()
        self.set_clients(args, ClientHomeLG)

        # Predictor model aggregation
        # self.global_model = copy.deepcopy(args.model.predictor)
        self.global_encoder = copy.deepcopy(args.model.encoder)
        self.global_decoder = copy.deepcopy(args.model.decoder)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Algorithm: ", self.algo, " | Dataset: ", self.dataset)

        self.train_losses = {i: [] for i in range(self.num_clients)}
        self.data_type = args.data_type
        self.result_path = os.path.join('..', 'results', self.dataset, self.data_type, self.algo, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.result_path, exist_ok=True)

        self.log_file = f"{self.result_path}/tracking.log"
        with open(self.log_file, 'a') as f:
            f.write(f"\nAlgorithm: {self.algo} \nDataset: {self.dataset}\n")
        
        self.budget = []
        self.model_list = {}

    def initialize_poisoned_workers(self):
        if self.noise_type is not None and self.num_of_poisoned_workers > 0:
            self.poisoned_worker_ids = np.random.choice(
                range(self.num_clients),
                size=int(self.num_of_poisoned_workers*self.num_clients),
                replace=False
            ).tolist()
            print(f"Poisoned Worker IDs: {self.poisoned_worker_ids}")

    def train(self):
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')
            
            self.evaluate()

            for client in self.selected_clients:
                client.train()

            # Save models state
            # models = [copy.deepcopy(self.global_model)]
            models = [
                copy.deepcopy(self.global_encoder), 
                copy.deepcopy(self.global_decoder)
            ]
            for c in self.clients:
                models.append(copy.deepcopy(c.model))
            self.model_list.update({epoch: models})

            self.receive_models()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('Time:', round(self.budget[-1], 2))

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

    def print_final_summary(self):
        def fmt(v): 
            if v is None: return "N/A"
            return "[" + ", ".join([f"{x:.4f}" if not np.isnan(x) else "NaN" for x in v]) + "]"
        
        summary_lines = [
            f"\n{'='*20} FINAL SUMMARY {'='*20}\n",
            f"Algorithm: {self.algo} | Dataset: {self.dataset}\n",
            f"\tBest Acc-normal: {max(self.rs_test_acc_normal):.4f}\n",
            f"\tBest F1-normal (Macro): {max(self.rs_test_f1_normal):.4f}\n",
            f"\tBest F1 Per-Class (at Best Round): {fmt(self.best_f1_pc_normal)}\n",
        ]
        
        if len(self.rs_test_acc_flipping) > 0 and max(self.rs_test_acc_flipping) > 0:
            summary_lines.extend([
                f"\tBest Acc-flipping: {max(self.rs_test_acc_flipping):.4f}\n",
                f"\tBest F1-flipping (Macro): {max(self.rs_test_f1_flipping):.4f}\n"
            ])

        for line in summary_lines:
            print(line.strip())
            with open(self.log_file, 'a') as f:
                f.write(line)
        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train')
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test')
            client = clientObj(args=args, id=i, poisoned_worker_ids=self.poisoned_worker_ids,
                               train_samples=len(train_data), test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        return list(np.random.choice(self.clients, join_clients, replace=False))

    # def send_models(self):
    #     for client in self.selected_clients:
    #         client.set_parameters(self.global_model)
    def send_models(self):
        for client in self.selected_clients:
            # Gửi cả encoder và decoder (có thể gói vào một object tạm)
            # Giả sử Client.set_parameters nhận vào một object có thuộc tính .encoder và .decoder
            shared_model = type('SharedModel', (), {
                'encoder': self.global_encoder,
                'decoder': self.global_decoder
            })
            client.set_parameters(shared_model)

    # def receive_models(self):
    #     active_train_samples = sum([c.train_samples for c in self.selected_clients])
    #     self.uploaded_weights = [c.train_samples / active_train_samples for c in self.selected_clients]
    #     self.uploaded_ids = [c.id for c in self.selected_clients]
    #     self.uploaded_models = [c.model.predictor for c in self.selected_clients]
    def receive_models(self):
        active_train_samples = sum([c.train_samples for c in self.selected_clients])
        self.uploaded_weights = [c.train_samples / active_train_samples for c in self.selected_clients]
        self.uploaded_ids = [c.id for c in self.selected_clients]
        
        # Lấy cặp (encoder, decoder) từ mỗi client
        # Lưu ý: Bạn cần thêm hàm get_parameters() bên phía Client như đã thảo luận trước đó
        self.uploaded_models = [c.get_parameters() for c in self.selected_clients]

    # def aggregate_parameters(self):
    #     self.global_model = copy.deepcopy(self.uploaded_models[0])
    #     for param in self.global_model.parameters():
    #         param.data = torch.zeros_like(param.data)
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
    #             server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        # 1. Reset tham số global về 0
        for param in self.global_encoder.parameters():
            param.data = torch.zeros_like(param.data)
        for param in self.global_decoder.parameters():
            param.data = torch.zeros_like(param.data)

        # 2. Thực hiện cộng dồn có trọng số (Weighted Aggregation)
        # uploaded_models bây giờ là danh sách các tuple: [(enc1, dec1), (enc2, dec2), ...]
        for w, (client_enc, client_dec) in zip(self.uploaded_weights, self.uploaded_models):
            
            # Tổng hợp Encoder
            for server_param, client_param in zip(self.global_encoder.parameters(), client_enc.parameters()):
                server_param.data += client_param.data.clone() * w
            
            # Tổng hợp Decoder
            for server_param, client_param in zip(self.global_decoder.parameters(), client_dec.parameters()):
                server_param.data += client_param.data.clone() * w
    # def test_metrics(self):
    #     # TỔNG HỢP PER-CLASS THEO TRỌNG SỐ (WEIGHTED AVERAGE)
    #     results_n = {'acc': [], 'f1': [], 'ns': 0, 'f1_pc': np.zeros(self.num_classes), 's_pc': np.zeros(self.num_classes)}
    #     results_f = {'acc': [], 'f1': [], 'ns': 0, 'f1_pc': np.zeros(self.num_classes), 's_pc': np.zeros(self.num_classes)}

    #     print('Performance Metrics Evaluation...')
    #     for c in self.selected_clients:
    #         acc, pre, rec, f1, ns, f1_pc, s_pc = c.test_metrics()
            
    #         # Chọn nhóm để cộng dồn
    #         group = results_f if c.id in self.poisoned_worker_ids else results_n
            
    #         group['acc'].append(acc * ns)
    #         group['f1'].append(f1 * ns)
    #         group['ns'] += ns
    #         group['f1_pc'] += (f1_pc * s_pc) # Cộng dồn tử số: F1_class * support_class
    #         group['s_pc'] += s_pc           # Cộng dồn mẫu số: support_class

    #     def compute_final(g):
    #         if g['ns'] == 0: return 0, 0, np.zeros(self.num_classes)
    #         avg_acc = sum(g['acc']) / g['ns']
    #         avg_f1 = sum(g['f1']) / g['ns']
            
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             avg_f1_pc = np.true_divide(g['f1_pc'], g['s_pc'])
    #             avg_f1_pc[g['s_pc'] == 0] = np.nan # Đánh dấu NaN nếu lớp đó không có mẫu nào
    #         return avg_acc, avg_f1, avg_f1_pc

    #     acc_n, f1_n, pc_n = compute_final(results_n)
    #     acc_f, f1_f, pc_f = compute_final(results_f)

    #     return acc_n, f1_n, pc_n, acc_f, f1_f, pc_f
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


        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))

    def plot_convergence(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rs_train_loss, label='Total Loss')
        plt.plot(self.ce_curve, label='CE Loss')
        plt.plot(self.re_curve, label='RE Loss')
        plt.title(f"{self.algo} Convergence")
        plt.xlabel("Round"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.result_path, "loss_plot.png"))
        plt.close()

    def save_results(self):
        metrics_results = {
            'acc-normal': self.rs_test_acc_normal,
            'f1-normal': self.rs_test_f1_normal,
            'f1-pc-normal': self.rs_test_f1_pc_normal,
            'ce-loss': self.ce_curve,
            're-loss': self.re_curve,
            'train-loss': self.rs_train_loss
        }
        with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)
        with open(f'{self.result_path}/models.pkl', 'wb') as f:
            pickle.dump(self.model_list, f)