import copy
import numpy as np
import torch
from flcore.clients.clientAvg import *
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
import os
import pickle
import time
import random
import gc
from flcore.servers.model_saver import (round_models, track_best, save_model_artifacts,
                                        save_pickle, mean_round_time, run_suffix,
                                        normalize_noise_type, select_poisoned_workers,
                                        setup_experiment_logging, write_final_summary)
from utils.data_utils import resolve_data_type
class FedAvg(object):
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
        self.rs_test_f1_normal = []
        self.rs_test_f1_flipping = []
        self.rs_test_precision_normal = []
        self.rs_test_precision_flipping = []
        self.rs_test_recall_normal = []
        self.rs_test_recall_flipping = []
        
        # --- NEW: Store Per-Class F1 history ---
        self.num_classes = args.num_classes
        self.rs_test_f1_pc_normal = [] 
        self.rs_test_f1_pc_flipping = []
        self.best_f1_pc_normal = None # Lưu mảng F1 tốt nhất

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
        self.set_clients(args, ClientAvg)
        
        self.full_test_data = read_client_data(self.dataset, -1, self.num_clients, data_type='test', type=self.data_type)
        self.full_test_loader = DataLoader(self.full_test_data, batch_size=args.batch_size, drop_last=False, shuffle=False)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset, "Data type: ", self.data_type)
        print("Finished creating server and clients.")

        self.train_losses = {i: [] for i in range(self.num_clients)}
        setup_experiment_logging(self, args)
        self.budget = []
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

    # def train(self):
    #     for epoch in range(self.global_rounds + 1):
    #         s_t = time.time()
    #         self.selected_clients = self.select_clients()
    #         self.send_models()

    #         print(f"\n-------------Round number: {epoch}-------------")
    #         with open(self.log_file, 'a') as f:
    #             f.write(f'-------------Round number: {epoch}-------------\n')
            
    #         self.evaluate()

    #         for client in self.selected_clients:
    #             client.train()
    #             torch.cuda.empty_cache()

    #         models = [copy.deepcopy(self.global_model)]
    #         for c in self.clients:
    #             state_dict = c.model.state_dict()
    #             for key in state_dict.keys():
    #                 state_dict[key] = state_dict[key].to('cpu') # Đẩy dữ liệu sang RAM máy
    #             models.append(state_dict)
                
    #         self.model_list.update({epoch: models})

    #         self.receive_models()
    #         self.aggregate_parameters()

    #         self.budget.append(time.time() - s_t)
    #         print('Time:', self.budget[-1])
    #         torch.cuda.empty_cache()
    def train(self):
        for epoch in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {epoch}-------------")
            with open(self.log_file, 'a') as f:
                f.write(f'-------------Round number: {epoch}-------------\n')
            
            # 1. Đánh giá mô hình
            self.evaluate()
            self.selected_history[epoch] = [c.id for c in self.selected_clients]
            track_best(self, epoch, self.rs_test_f1_normal[-1], self.global_model, self.clients)

            # 2. Huấn luyện Local
            for client in self.selected_clients:
                client.train()
                # Giải phóng VRAM ngay sau khi mỗi client tập xong
                torch.cuda.empty_cache()

            # 3. Tổng hợp mô hình (Thay thế cách lưu list bằng tính toán trực tiếp)
            # CHÚ Ý: Nếu receive_models và aggregate_parameters của bạn 
            # phụ thuộc vào self.model_list, hãy sửa lại như sau:
            
            # Chỉ giữ lại models của round HIỆN TẠI, không lưu lại lịch sử các round trước
            current_models = []
            for c in self.selected_clients: # Chỉ lấy những client tham gia huấn luyện
                # Chuyển state_dict sang CPU để tiết kiệm VRAM
                state_dict = {k: v.to('cpu') for k, v in c.model.state_dict().items()}
                current_models.append(state_dict)
            
            # Cập nhật danh sách model để hàm aggregate_parameters sử dụng
            self.uploaded_weights = [1.0 / len(self.selected_clients)] * len(self.selected_clients)
            self.uploaded_ids = [c.id for c in self.selected_clients]
            self.uploaded_models = current_models

            # 4. Thực hiện gộp mô hình
            # self.receive_models() # Nếu hàm này dùng để nhận model, hãy kiểm tra nó
            self.aggregate_parameters() # Đảm bảo hàm này lấy từ self.uploaded_models

            # Lưu checkpoint của round (bản sao trên CPU: global sau aggregate + tất cả client)
            self.model_list[epoch] = round_models(self.global_model, self.clients)

            # 5. Dọn dẹp bộ nhớ quan trọng
            self.budget.append(time.time() - s_t)
            print('Time:', self.budget[-1])
            
            # Xóa các biến tạm và dọn rác
            del current_models
            gc.collect() 
            torch.cuda.empty_cache()

            # (Đã bỏ self.model_list.clear(): trước đây khiến models.pkl luôn rỗng)

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
            client.set_parameters(self.global_model)

    def receive_models(self):
        active_train_samples = sum([c.train_samples for c in self.selected_clients])
        self.uploaded_weights = [c.train_samples / active_train_samples for c in self.selected_clients]
        self.uploaded_ids = [c.id for c in self.selected_clients]
        self.uploaded_models = [c.model for c in self.selected_clients]

    # def aggregate_parameters(self):
    #     self.global_model = copy.deepcopy(self.uploaded_models[0])
    #     for param in self.global_model.parameters():
    #         param.data = torch.zeros_like(param.data)
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
    #             server_param.data += client_param.data.clone() * w
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        global_dict = self.global_model.state_dict()
        
        # Tạo dict chứa tổng
        for key in global_dict.keys():
            # Chỉ xóa về 0 nếu là kiểu Float để chuẩn bị cộng dồn
            if torch.is_floating_point(global_dict[key]):
                global_dict[key] = torch.zeros_like(global_dict[key])
            # Nếu là Long (như num_batches_tracked), giữ nguyên hoặc xử lý riêng

        for client_dict in self.uploaded_models:
            for key in global_dict.keys():
                device = global_dict[key].device
                
                # Chỉ cộng dồn nếu là kiểu số thực (weights, biases)
                if torch.is_floating_point(global_dict[key]):
                    global_dict[key] += client_dict[key].to(device) / len(self.uploaded_models)
                else:
                    # Với các biến Long, chúng ta thường lấy giá trị của client cuối cùng 
                    # hoặc giữ nguyên bản của server
                    global_dict[key] = client_dict[key].to(device)

        # Nạp lại vào mô hình
        self.global_model.load_state_dict(global_dict)

    # def test_metrics(self):
    #     # normal
    #     t_acc_n, t_pre_n, t_rec_n, t_f1_n = [], [], [], []
    #     num_s_n = 0
    #     sum_f1_pc_n = np.zeros(self.num_classes)
    #     count_s_pc_n = np.zeros(self.num_classes)

    #     # flipping
    #     t_acc_f, t_pre_f, t_rec_f, t_f1_f = [], [], [], []
    #     num_s_f = 0
    #     sum_f1_pc_f = np.zeros(self.num_classes)
    #     count_s_pc_f = np.zeros(self.num_classes)

    #     print('Performance Metrics')
    #     with open(self.log_file, 'a') as f:
    #         f.write('Performance Metrics\n')

    #     for c in self.selected_clients:
    #         # Client must return: acc, pre, rec, f1, ns, f1_pc, s_pc
    #         acc, pre, rec, f1, ns, f1_pc, s_pc = c.test_metrics()
            
    #         log_str = f'\tClient {c.id}: acc {round(acc, 2)}, precision: {round(pre, 2)}, recall: {round(rec, 2)}, f1: {round(f1, 2)}'
    #         print(log_str)
    #         with open(self.log_file, 'a') as f:
    #             f.write(log_str + '\n')

    #         if c.id in self.poisoned_worker_ids:
    #             t_acc_f.append(acc * ns); t_f1_f.append(f1 * ns)
    #             t_rec_f.append(rec * ns); t_pre_f.append(pre * ns)
    #             num_s_f += ns
    #             sum_f1_pc_f += (f1_pc * s_pc)
    #             count_s_pc_f += s_pc
    #         else:
    #             t_acc_n.append(acc * ns); t_f1_n.append(f1 * ns)
    #             t_rec_n.append(rec * ns); t_pre_n.append(pre * ns)
    #             num_s_n += ns
    #             sum_f1_pc_n += (f1_pc * s_pc)
    #             count_s_pc_n += s_pc

    #     avg_f1_pc_n = np.divide(sum_f1_pc_n, count_s_pc_n, out=np.zeros_like(sum_f1_pc_n), where=count_s_pc_n!=0)
    #     avg_f1_pc_f = np.divide(sum_f1_pc_f, count_s_pc_f, out=np.zeros_like(sum_f1_pc_f), where=count_s_pc_f!=0)

    #     return (t_acc_n, t_f1_n, t_rec_n, t_pre_n, num_s_n, 
    #             t_acc_f, t_f1_f, t_rec_f, t_pre_f, num_s_f, 
    #             avg_f1_pc_n, avg_f1_pc_f)
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
            self.train_losses[c.id].append(loss)
            print(f'\tClient {c.id}: Train loss: {round(loss, 2)}')
            num_samples.append(len(list_loss))
            losses.extend(list_loss)
        return num_samples, losses

    def save_results(self):
        # Đã cập nhật để lưu thêm f1-per-class và metadata kịch bản vào file pkl
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
            'f1-per-class-normal': self.rs_test_f1_pc_normal,
            'train-loss': self.rs_train_loss,
            'loss': self.train_losses,
            'poisoned_worker_ids': self.poisoned_worker_ids,
            'selected_clients': self.selected_history,
            'best_round': self.best_round,
            'best_f1_normal': self.best_score,
        }
        save_pickle(metrics_results, os.path.join(self.result_path, f'metrics_results{run_suffix(self.times)}.pkl'))
        save_model_artifacts(self, self.global_model, self.model_list)

