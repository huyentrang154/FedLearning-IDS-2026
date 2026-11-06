import copy
import numpy as np
import torch
import os
import pickle
import time
import random
from flcore.clients.clientAE import ClientAE      # <-- đảm bảo đường dẫn này đúng
from utils.data_utils import read_client_data

class FedAE(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        # We'll keep global_models as dict of 3 models
        self.global_models = None  # will init after clients created
        self.num_clients = args.num_clients
        self.round_to_add_clients = args.round_to_add_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio

        self.clients = []
        self.selected_clients = []

        # uploaded data from clients for aggregation
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []  # list of dicts per-client: {"normal":m,"abnormal":m,"hybrid":m}

        # for visualize / logging
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
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.poisoned_worker_ids = []
        # initialize poisoned workers
        self.initialize_poisoned_workers()

        # create clients (ClientAE)
        self.set_clients(args, ClientAE)

        # initialize global_models by copying models from client 0 (if exists)
        if len(self.clients) > 0:
            # clients[0] has attribute `models` which is a dict of 3 models
            # make deepcopy to avoid sharing references
            client0_models = self.clients[0].models
            self.global_models = {
                k: copy.deepcopy(v).to(self.device) for k, v in client0_models.items()
            }
        else:
            raise ValueError("No clients created; cannot initialize global models.")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("Finished creating server and clients.")

        # save loss per client
        self.train_losses = {i: [] for i in range(self.num_clients)}
        # set path to save result
        self.data_type = args.data_type
        self.result_path = os.path.join('..', 'results', self.dataset, self.data_type, self.algo, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.result_path, exist_ok=True)

        self.log_file = f"{self.result_path}/tracking.log"
        with open(self.log_file, 'a') as f:
            f.write(f"\nAlgorithm: {self.algo} \nDataset: {self.dataset}\n")
        self.budget = []

        # for get model snapshots in each round
        self.model_list = {}

    def initialize_poisoned_workers(self):
        """Select poisoned workers based on the given parameters."""
        if self.noise_type is not None and self.num_of_poisoned_workers > 0:
            # choose indexes among num_clients
            n_poison = int(self.num_of_poisoned_workers * self.num_clients)
            self.poisoned_worker_ids = np.random.choice(
                range(self.num_clients),
                size=n_poison,
                replace=False
            ).tolist()
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
            self.evaluate()

            # local training on selected clients
            for client in self.selected_clients:
                client.train()

            # save snapshot: first element = global_models, then each client models
            models_snapshot = {"global": {k: copy.deepcopy(v) for k, v in self.global_models.items()}}
            for c in self.clients:
                # assume each client has attribute `models` (dict)
                models_snapshot[f"client_{c.id}"] = {k: copy.deepcopy(v) for k, v in c.models.items()}
            self.model_list.update({epoch: models_snapshot})

            # receive client updates and aggregate
            self.receive_models()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('Time:', self.budget[-1])

        # print best results summary
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("\nBest performance metrics")
        if len(self.rs_test_acc_flipping) > 0:
            print("\tacc-flipping:", max(self.rs_test_acc_flipping))
        if len(self.rs_test_acc_normal) > 0:
            print("\tacc-normal:", max(self.rs_test_acc_normal))
        if len(self.rs_test_f1_flipping) > 0:
            print("\tf1-flipping:", max(self.rs_test_f1_flipping))
        if len(self.rs_test_f1_normal) > 0:
            print("\tf1-normal:", max(self.rs_test_f1_normal))
        print("\tTime avg per round:", sum(self.budget[1:]) / len(self.budget[1:]) if len(self.budget) > 1 else sum(self.budget))

        self.save_results()

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, self.num_clients, data_type='train')
            test_data = read_client_data(self.dataset, i, self.num_clients, data_type='test')
            client = clientObj(
                args=args,
                id=i,
                poisoned_worker_ids = self.poisoned_worker_ids,
                train_samples=len(train_data),
                test_samples=len(test_data)
            )
            # if client needs to know poisoned ids (optional), ClientAE constructor can accept
            if hasattr(client, 'poisoned_worker_ids'):
                client.poisoned_worker_ids = self.poisoned_worker_ids
            self.clients.append(client)

    def select_clients(self, epoch):
        if epoch < self.round_to_add_clients:
            candidate_clients = self.clients[:15]
        else:
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
        # send each of the 3 global models to selected clients
        for client in self.selected_clients:
            # ClientAE.set_parameters(model, ae_type) -- call 3 times
            client.set_parameters(self.global_models["normal"], ae_type="normal")
            client.set_parameters(self.global_models["abnormal"], ae_type="abnormal")
            client.set_parameters(self.global_models["hybrid"], ae_type="hybrid")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # Tổng số mẫu cho từng loại model
        total_normal = sum(c.class_counts.get(0, 0) for c in self.selected_clients)   # y=0
        total_abnormal = sum(c.class_counts.get(1, 0) for c in self.selected_clients) # y=1
        total_hybrid = sum(c.train_samples for c in self.selected_clients)            # all

        self.uploaded_weights = []  # list các dict trọng số {"normal":..,"abnormal":..,"hybrid":..}
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.selected_clients:
            # Lấy số mẫu theo từng loại
            n_normal = client.class_counts.get(0, 0)
            n_abnormal = client.class_counts.get(1, 0)
            n_hybrid = client.train_samples

            # Tính trọng số từng loại model
            w_normal = n_normal / total_normal if total_normal > 0 else 0
            w_abnormal = n_abnormal / total_abnormal if total_abnormal > 0 else 0
            w_hybrid = n_hybrid / total_hybrid if total_hybrid > 0 else 0

            self.uploaded_weights.append({
                "normal": w_normal,
                "abnormal": w_abnormal,
                "hybrid": w_hybrid
            })
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append({k: copy.deepcopy(v) for k, v in client.models.items()})


    def add_parameters_to_model(self, server_model, client_model, w):
        """Add weighted client parameters into server_model (in-place)."""
        for server_param, client_param in zip(server_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # initialize global models with zeros of same shape
        for k in self.global_models.keys():
            # zero all params
            for param in self.global_models[k].parameters():
                param.data = torch.zeros_like(param.data)

        # sum weighted parameters from clients per model type
        for w_dict, client_models in zip(self.uploaded_weights, self.uploaded_models):
            for k in self.global_models.keys():
                self.add_parameters_to_model(self.global_models[k], client_models[k], w_dict[k])

    def test_metrics(self):
    # This function aggregates metrics from clients
        metrics_normal = {"normal": [], "abnormal": [], "hybrid": [], "ensemble": []}
        metrics_flipping = {"normal": [], "abnormal": [], "hybrid": [], "ensemble": []}

        num_samples_normal = 0
        num_samples_flipping = 0

        print('Performance Metrics')
        with open(self.log_file, 'a') as f:
            f.write('Performance Metrics\n')

        for c in self.selected_clients:
            results = c.test_metrics()  # client trả về dict {"normal": {...}, "abnormal": {...}, "hybrid": {...}, "ensemble": {...}, "n_samples": N}
            ns = results["n_samples"]

            # Ghi log từng client
            print(f"\tClient {c.id} (samples={ns}):")
            with open(self.log_file, 'a') as f:
                f.write(f"\tClient {c.id} (samples={ns}):\n")

            for model_type in ["normal", "abnormal", "hybrid", "ensemble"]:
                acc = results[model_type]["acc"]
                precision = results[model_type]["precision"]
                recall = results[model_type]["recall"]
                f1 = results[model_type]["f1"]

                print(f"\t\t{model_type}: acc {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}")
                with open(self.log_file, 'a') as f:
                    f.write(f"\t\t{model_type}: acc {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}\n")

                if c.id in self.poisoned_worker_ids:
                    metrics_flipping[model_type].append({
                        "acc": acc * ns,
                        "precision": precision * ns,
                        "recall": recall * ns,
                        "f1": f1 * ns
                    })
                else:
                    metrics_normal[model_type].append({
                        "acc": acc * ns,
                        "precision": precision * ns,
                        "recall": recall * ns,
                        "f1": f1 * ns
                    })

            if c.id in self.poisoned_worker_ids:
                num_samples_flipping += ns
            else:
                num_samples_normal += ns

        return metrics_normal, num_samples_normal, metrics_flipping, num_samples_flipping

    # def test_metrics(self):
    #     # This function may not be used directly; server uses clients' test_metrics to compute aggregated stats
    #     total_acc_normal = []
    #     total_precision_normal = []
    #     total_recall_normal = []
    #     total_f1_normal = []
    #     num_samples_normal = 0

    #     total_acc_flipping = []
    #     total_precision_flipping = []
    #     total_recall_flipping = []
    #     total_f1_flipping = []
    #     num_samples_flipping = 0

    #     print('Performance Metrics')
    #     with open(self.log_file, 'a') as f:
    #         f.write('Performance Metrics\n')

    #     for c in self.selected_clients:
    #         acc, precision, recall, f1, ns = c.test_metrics()
    #         print(f'\tClient {c.id}: acc {round(acc, 4)}, precision: {round(precision, 4)}, recall: {round(recall, 4)}, f1: {round(f1, 4)}')
    #         with open(self.log_file, 'a') as f:
    #             f.write(f'\tClient {c.id}: acc {round(acc, 4)}, precision: {round(precision, 4)}, recall: {round(recall, 4)}, f1: {round(f1, 4)}\n')

    #         if c.id in self.poisoned_worker_ids:
    #             total_acc_flipping.append(acc * ns)
    #             total_recall_flipping.append(recall * ns)
    #             total_f1_flipping.append(f1 * ns)
    #             total_precision_flipping.append(precision * ns)
    #             num_samples_flipping += ns
    #         else:
    #             total_acc_normal.append(acc * ns)
    #             total_recall_normal.append(recall * ns)
    #             total_f1_normal.append(f1 * ns)
    #             total_precision_normal.append(precision * ns)
    #             num_samples_normal += ns

    #     return (total_acc_normal, total_f1_normal, total_recall_normal, total_precision_normal, num_samples_normal,
    #             total_acc_flipping, total_f1_flipping, total_recall_flipping, total_precision_flipping, num_samples_flipping)

    def train_metrics(self):
        print("Train Loss")
        with open(self.log_file, 'a') as f:
            f.write('Train Loss\n')

        # Lưu theo dạng dict để theo dõi từng AE
        metrics_all = { "normal": [], "abnormal": [], "hybrid": [] }

        for c in self.selected_clients:
            metrics_result = c.train_metrics()   # dict {"normal":..., "abnormal":..., "hybrid":...}

            for ae_type, loss in metrics_result.items():
                if loss is None:
                    continue

                # Nếu key chưa tồn tại thì khởi tạo list
                if (c.id, ae_type) not in self.train_losses:
                    self.train_losses[(c.id, ae_type)] = []

                self.train_losses[(c.id, ae_type)].append(loss)
                metrics_all[ae_type].append(loss)

                print(f'\tClient {c.id} [{ae_type}]: Train loss: {round(loss, 4)}')
                with open(self.log_file, 'a') as f:
                    f.write(f'\tClient {c.id} [{ae_type}]: Train loss: {round(loss, 4)}\n')

        return metrics_all




    # def evaluate(self):
    #     # evaluate train
    #     # stats_train = self.train_metrics()
    #     # train_loss = sum(stats_train[1]) * 1.0 / sum(stats_train[0]) if sum(stats_train[0]) > 0 else 0.0
    #     # self.rs_train_loss.append(train_loss)

    #     # evaluate test, avg metrics for all clients
    #     stats_test = self.test_metrics()
    #     test_acc_normal = sum(stats_test[0]) / stats_test[4] if stats_test[4] > 0 else 0
    #     test_f1_normal = sum(stats_test[1]) / stats_test[4] if stats_test[4] > 0 else 0
    #     test_recall_normal = sum(stats_test[2]) / stats_test[4] if stats_test[4] > 0 else 0
    #     test_precision_normal = sum(stats_test[3]) / stats_test[4] if stats_test[4] > 0 else 0

    #     if stats_test[9] > 0:
    #         test_acc_flipping = sum(stats_test[5]) / stats_test[9]
    #         test_f1_flipping = sum(stats_test[6]) / stats_test[9]
    #         test_recall_flipping = sum(stats_test[7]) / stats_test[9]
    #         test_precision_flipping = sum(stats_test[8]) / stats_test[9]
    #     else:
    #         test_acc_flipping = test_f1_flipping = test_recall_flipping = test_precision_flipping = 0

    #     self.rs_test_acc_normal.append(test_acc_normal)
    #     self.rs_test_f1_normal.append(test_f1_normal)
    #     self.rs_test_precision_normal.append(test_precision_normal)
    #     self.rs_test_recall_normal.append(test_recall_normal)

    #     self.rs_test_acc_flipping.append(test_acc_flipping)
    #     self.rs_test_f1_flipping.append(test_f1_flipping)
    #     self.rs_test_precision_flipping.append(test_precision_flipping)
    #     self.rs_test_recall_flipping.append(test_recall_flipping)

    #     print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
    #     print("Overall Performance Metrics")
    #     #print("\ttrain loss avg: {:.4f}".format(train_loss))
    #     print("\ttest acc normal avg: {:.4f}".format(test_acc_normal))
    #     print("\ttest acc flipping avg: {:.4f}".format(test_acc_flipping))

    #     with open(self.log_file, 'a') as f:
    #         f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
    #         f.write("Overall Performance Metrics\n")
    #         #f.write("\ttrain loss avg: {:.4f}\n".format(train_loss))
    # def evaluate(self):
    # # === Evaluate train loss trung bình cho từng AE ===
    #     stats_train = self.train_metrics()   # dict {"normal": [...], "abnormal": [...], "hybrid": [...]}

    #     avg_train_losses = {}
    #     for ae_type, losses in stats_train.items():
    #         if len(losses) > 0:
    #             avg_train_losses[ae_type] = sum(losses) / len(losses)
    #         else:
    #             avg_train_losses[ae_type] = None

    #     # === Evaluate test ===
    #     stats_test = self.test_metrics()
    #     test_acc_normal = sum(stats_test[0]) / stats_test[4] if stats_test[4] > 0 else 0
    #     test_f1_normal = sum(stats_test[1]) / stats_test[4] if stats_test[4] > 0 else 0
    #     test_recall_normal = sum(stats_test[2]) / stats_test[4] if stats_test[4] > 0 else 0
    #     test_precision_normal = sum(stats_test[3]) / stats_test[4] if stats_test[4] > 0 else 0

    #     if stats_test[9] > 0:
    #         test_acc_flipping = sum(stats_test[5]) / stats_test[9]
    #         test_f1_flipping = sum(stats_test[6]) / stats_test[9]
    #         test_recall_flipping = sum(stats_test[7]) / stats_test[9]
    #         test_precision_flipping = sum(stats_test[8]) / stats_test[9]
    #     else:
    #         test_acc_flipping = test_f1_flipping = test_recall_flipping = test_precision_flipping = 0

    #     self.rs_test_acc_normal.append(test_acc_normal)
    #     self.rs_test_f1_normal.append(test_f1_normal)
    #     self.rs_test_precision_normal.append(test_precision_normal)
    #     self.rs_test_recall_normal.append(test_recall_normal)

    #     self.rs_test_acc_flipping.append(test_acc_flipping)
    #     self.rs_test_f1_flipping.append(test_f1_flipping)
    #     self.rs_test_precision_flipping.append(test_precision_flipping)
    #     self.rs_test_recall_flipping.append(test_recall_flipping)

    #     # === Print results ===
    #     print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
    #     print("Overall Performance Metrics")
    #     for ae_type, avg_loss in avg_train_losses.items():
    #         if avg_loss is not None:
    #             print(f"\ttrain loss avg [{ae_type}]: {avg_loss:.4f}")
    #         else:
    #             print(f"\ttrain loss avg [{ae_type}]: N/A")

    #     print("\ttest acc normal avg: {:.4f}".format(test_acc_normal))
    #     print("\ttest acc flipping avg: {:.4f}".format(test_acc_flipping))

    #     # === Save to log file ===
    #     with open(self.log_file, 'a') as f:
    #         f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
    #         f.write("Overall Performance Metrics\n")
    #         for ae_type, avg_loss in avg_train_losses.items():
    #             if avg_loss is not None:
    #                 f.write(f"\ttrain loss avg [{ae_type}]: {avg_loss:.4f}\n")
    #             else:
    #                 f.write(f"\ttrain loss avg [{ae_type}]: N/A\n")

    #         f.write("\ttest acc normal avg: {:.4f}\n".format(test_acc_normal))
    #         f.write("\ttest acc flipping avg: {:.4f}\n".format(test_acc_flipping))

    def evaluate(self):
    # === Evaluate train loss trung bình cho từng AE ===
        stats_train = self.train_metrics()   # dict {"normal": [...], "abnormal": [...], "hybrid": [...]}

        avg_train_losses = {}
        for ae_type, losses in stats_train.items():
            if len(losses) > 0:
                avg_train_losses[ae_type] = sum(losses) / len(losses)
            else:
                avg_train_losses[ae_type] = None

        # === Evaluate test ===
        metrics_normal, num_samples_normal, metrics_flipping, num_samples_flipping = self.test_metrics()

        def aggregate(metrics_list, num_samples):
            """Tính trung bình có trọng số cho acc, precision, recall, f1."""
            if num_samples == 0:
                return {"acc": 0, "precision": 0, "recall": 0, "f1": 0}
            total = {"acc": 0, "precision": 0, "recall": 0, "f1": 0}
            for m in metrics_list:
                for k in total.keys():
                    total[k] += m[k]
            return {k: total[k] / num_samples for k in total}

        avg_test_normal = {m: aggregate(metrics_normal[m], num_samples_normal) for m in metrics_normal}
        avg_test_flipping = {m: aggregate(metrics_flipping[m], num_samples_flipping) for m in metrics_flipping}

        # === Save results for ensemble (ví dụ bạn muốn plot sau này) ===
        self.rs_test_acc_normal.append(avg_test_normal["ensemble"]["acc"])
        self.rs_test_f1_normal.append(avg_test_normal["ensemble"]["f1"])
        self.rs_test_precision_normal.append(avg_test_normal["ensemble"]["precision"])
        self.rs_test_recall_normal.append(avg_test_normal["ensemble"]["recall"])

        self.rs_test_acc_flipping.append(avg_test_flipping["ensemble"]["acc"])
        self.rs_test_f1_flipping.append(avg_test_flipping["ensemble"]["f1"])
        self.rs_test_precision_flipping.append(avg_test_flipping["ensemble"]["precision"])
        self.rs_test_recall_flipping.append(avg_test_flipping["ensemble"]["recall"])

        # === Print results ===
        print("\nAlgorithm: ", self.algo, "Dataset: ", self.dataset)
        print("Overall Performance Metrics")
        for ae_type, avg_loss in avg_train_losses.items():
            if avg_loss is not None:
                print(f"\ttrain loss avg [{ae_type}]: {avg_loss:.4f}")
            else:
                print(f"\ttrain loss avg [{ae_type}]: N/A")

        print("\n\tTest (normal clients):")
        for model_type, vals in avg_test_normal.items():
            print(f"\t\t{model_type}: acc {vals['acc']:.4f}, precision {vals['precision']:.4f}, "
                f"recall {vals['recall']:.4f}, f1 {vals['f1']:.4f}")

        print("\n\tTest (poisoned clients):")
        for model_type, vals in avg_test_flipping.items():
            print(f"\t\t{model_type}: acc {vals['acc']:.4f}, precision {vals['precision']:.4f}, "
                f"recall {vals['recall']:.4f}, f1 {vals['f1']:.4f}")

        # === Save to log file ===
        with open(self.log_file, 'a') as f:
            f.write("\nAlgorithm: {}, Dataset: {}\n".format(self.algo, self.dataset))
            f.write("Overall Performance Metrics\n")
            for ae_type, avg_loss in avg_train_losses.items():
                if avg_loss is not None:
                    f.write(f"\ttrain loss avg [{ae_type}]: {avg_loss:.4f}\n")
                else:
                    f.write(f"\ttrain loss avg [{ae_type}]: N/A\n")

            f.write("\n\tTest (normal clients):\n")
            for model_type, vals in avg_test_normal.items():
                f.write(f"\t\t{model_type}: acc {vals['acc']:.4f}, precision {vals['precision']:.4f}, "
                        f"recall {vals['recall']:.4f}, f1 {vals['f1']:.4f}\n")

            f.write("\n\tTest (poisoned clients):\n")
            for model_type, vals in avg_test_flipping.items():
                f.write(f"\t\t{model_type}: acc {vals['acc']:.4f}, precision {vals['precision']:.4f}, "
                        f"recall {vals['recall']:.4f}, f1 {vals['f1']:.4f}\n")

    def save_results(self):
        metrics_results = {
            'acc-flipping': self.rs_test_acc_flipping,
            'acc-normal': self.rs_test_acc_normal,
            'f1-flipping': self.rs_test_f1_flipping,
            'f1-normal': self.rs_test_f1_normal,
            'recall-flipping': self.rs_test_recall_flipping,
            'precision-flipping': self.rs_test_precision_flipping,
            'precision-normal': self.rs_test_precision_normal,
            'loss-per-client': self.train_losses
        }

        with open(f'{self.result_path}/metrics_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)

        # save checkpoint of global and client models after each communication round
        with open(f'{self.result_path}/models.pkl', 'wb') as f:
            pickle.dump(self.model_list, f)
