# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# from utils.data_utils import read_client_data
# from torch.utils.data import DataLoader
# from sklearn import metrics


# class ClientHomeLG(object):
#     def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
#         self.model = copy.deepcopy(args.model)
#         self.dataset = args.dataset
#         self.device = args.device
#         self.num_clients = args.num_clients
#         self.id = id

#         self.poisoned_worker_ids = poisoned_worker_ids
#         self.train_samples = train_samples
#         self.test_samples = test_samples
#         self.batch_size = args.batch_size
#         self.learning_rate = args.local_learning_rate
#         self.local_steps = args.local_steps

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.re_loss = nn.MSELoss()
#         self.global_model = copy.deepcopy(self.model)
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
#         # self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)

#         self.noise_type = args.noise_type
#         self.num_of_poisoned_workers = args.num_of_poisoned_workers
#         self.poisoned_sample_ratio = args.poisoned_sample_ratio
#         self.attack_noise_std = args.attack_noise_std
#         self.num_classes = args.num_classes
#         self.data_type = args.data_type
#         self.train_loader = self.load_train_data()
#         self.test_loader = self.load_test_data()
#         # self.num_classes = args.num_classes


#     def apply_poisoning(self, train_data, noise_type, poisoned_sample_ratio, attack_noise_std):
#         """Apply data poisoning to the client's training data."""
#         X = np.array([x.numpy() for x, _ in train_data])
#         Y = np.array([y.item() for _, y in train_data])

#         if noise_type == "label_flipping":
#             possible_labels = np.arange(self.num_classes)
#             flip_indices = np.random.choice(
#                 len(Y), size=int(len(Y) * poisoned_sample_ratio), replace=False
#             )
#             for idx in flip_indices:
#                 Y[idx] = np.random.choice(
#                     possible_labels[possible_labels != Y[idx]]
#                 )
#         elif noise_type == "gaussian_noise":
#             for idx in range(int(len(X) * poisoned_sample_ratio)):
#                 X[idx] += attack_noise_std * np.random.randn(*X[idx].shape)

#         # Update poisoned data
#         train_data = [(torch.tensor(x), torch.tensor(y)) for x, y in zip(X, Y)]
#         return train_data

#     def train(self):
#         self.model.train()
#         # self.global_model = copy.deepcopy(self.model)

#         for epoch in range(self.local_steps):
#             for i, (x, y) in enumerate(self.train_loader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 output_l, x_reconstruction_l = self.model(x)
#                 # output_g, x_reconstruction_g = self.global_model(x)
#                 ce_loss = self.ce_loss(output_l, y)
#                 re_loss = self.re_loss(x_reconstruction_l, x)
#                 loss = ce_loss + re_loss
#                 self.optimizer.zero_grad()
#                 # self.optimizer_g.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 # self.optimizer_g.step()

#     def load_train_data(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#         train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
#         if self.id in self.poisoned_worker_ids:
#             train_data = self.apply_poisoning(train_data,
#                         noise_type=self.noise_type,
#                         poisoned_sample_ratio=self.poisoned_sample_ratio,
#                         attack_noise_std=self.attack_noise_std)
#         return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

#     def load_test_data(self, batch_size=1):
#         if batch_size is None:
#             batch_size = self.batch_size
#         test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type)
#         return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

#     def test_metrics(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()

#         y_true = []
#         y_predict = []

#         with torch.no_grad():
#             for x, y in self.test_loader:
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 output, x_reconstruction = self.model(x)
#                 y_predict.append(torch.argmax(output, dim=1).detach().cpu().numpy())
#                 y_true.append(y.detach().cpu().numpy())

#         y_predict = np.concatenate(y_predict, axis=0)
#         y_true = np.concatenate(y_true, axis=0)

#         acc = metrics.accuracy_score(y_true, y_predict)
#         precision = metrics.precision_score(y_true, y_predict, average='macro')
#         recall = metrics.recall_score(y_true, y_predict, average='macro')
#         f1 = metrics.f1_score(y_true, y_predict, average='macro')
#         # auc = metrics.roc_auc_score(y_true, y_predict, average='micro')
#         test_num = len(y_predict)
#         # import pdb
#         # pdb.set_trace()

#         # return acc, precision, recall, f1, auc, test_num
#         return acc, precision, recall, f1, test_num

#     def train_metrics(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()
#         list_loss = []
#         # for calculate threshold
#         with torch.no_grad():
#             for x, y in self.train_loader:
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 output, x_reconstruction = self.model(x)
#                 ce_loss = self.ce_loss(output, y)
#                 re_loss = self.re_loss(x_reconstruction, x)
#                 loss = ce_loss + re_loss
#                 list_loss.append(loss.item())
#         return list_loss

#     def set_parameters(self, model):
#         for new_param, old_param in zip(model.parameters(), self.model.predictor.parameters()):
#             old_param.data = new_param.data.clone()
import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, random_split
from sklearn import metrics


class ClientHomeLG(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model).to(args.device)
        self.dataset = args.dataset
        self.device = args.device
        self.num_clients = args.num_clients
        self.id = id

        self.poisoned_worker_ids = poisoned_worker_ids
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.re_loss = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Poisoning
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.num_classes = args.num_classes

        # Data
        self.data_type = args.data_type
        full_train_data = self.load_train_data(return_raw=True)

        # Apply poisoning
        if self.id in self.poisoned_worker_ids:
            full_train_data = self.apply_poisoning(
                full_train_data,
                noise_type=self.noise_type,
                poisoned_sample_ratio=self.poisoned_sample_ratio,
                attack_noise_std=self.attack_noise_std,
            )

        # --- Split 90/10 train-val giống FedAvg ---
        self.train_data, self.val_data = self._split_train_val(full_train_data)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False)

        self.test_loader = self.load_test_data()

        # --- Early Stopping params ---
        self.patience = getattr(args, "patience", 5)
        self.min_delta = getattr(args, "min_delta", 1e-4)
        self.early_stop_metric = getattr(args, "early_stop_metric", "f1")  # 'f1' or 'loss'


    # ====== DATA =========
    def load_train_data(self, return_raw=False):
        train_data = read_client_data(
            self.dataset, self.id, self.num_clients,
            data_type='train', type=self.data_type
        )
        return train_data if return_raw else DataLoader(train_data, self.batch_size, drop_last=False, shuffle=True)


    def load_test_data(self, batch_size=1):
        test_data = read_client_data(
            self.dataset, self.id, self.num_clients,
            data_type='test', type=self.data_type
        )
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


    # ====== 90/10 split =========
    def _split_train_val(self, data_list, val_ratio=0.1, seed=42):
        n = len(data_list)
        if n <= 1:
            return data_list, data_list
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val
        train_set, val_set = random_split(
            data_list,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        return list(train_set), list(val_set)


    # ======= Poisoning =======
    def apply_poisoning(self, train_data, noise_type, poisoned_sample_ratio, attack_noise_std):
        X = [x.clone() for x, _ in train_data]
        Y = [int(y) for _, y in train_data]
        n = len(X)
        k = int(n * poisoned_sample_ratio)

        poisoned_idx = np.random.choice(n, k, replace=False)

        if noise_type == "label_flipping":
            for idx in poisoned_idx:
                old = Y[idx]
                choices = [c for c in range(self.num_classes) if c != old]
                Y[idx] = np.random.choice(choices)

        elif noise_type == "gaussian_noise":
            for idx in poisoned_idx:
                X[idx] = X[idx] + attack_noise_std * torch.randn_like(X[idx])

        return [(X[i], torch.tensor(Y[i])) for i in range(n)]


    # ======== Early Stopping Training ========
    def train(self):
        self.model.train()

        best_state = None
        no_improve = 0

        best_score = -float("inf") if self.early_stop_metric == "f1" else float("inf")

        for epoch in range(self.local_steps):
            # ---- TRAIN ONE EPOCH ----
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).long()

                logits, x_rec = self.model(x)

                loss_ce = self.ce_loss(logits, y)
                loss_re = self.re_loss(x_rec, x)
                loss = loss_ce + loss_re

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # ---- VALIDATE ----
            val_stats = self.evaluate_on_val()
            val_loss, val_f1 = val_stats["loss"], val_stats["f1"]

            print(f"[Client {self.id}] Epoch {epoch+1}/{self.local_steps} | "
                  f"Val F1={val_f1:.4f} | Val Loss={val_loss:.6f}")

            # ---- CHECK IMPROVEMENT ----
            improved = False

            if self.early_stop_metric == "f1":
                if val_f1 > best_score + self.min_delta:
                    improved = True
                    best_score = val_f1
            else:
                if val_loss < best_score - self.min_delta:
                    improved = True
                    best_score = val_loss

            if improved:
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"[Client {self.id}] Early stopping at epoch {epoch+1} "
                          f"(best {self.early_stop_metric}={best_score:.6f})")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)


    # ======== Validation metric ========
    def evaluate_on_val(self):
        self.model.eval()

        y_true = []
        y_pred = []
        losses = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device).long()

                logits, x_rec = self.model(x)

                loss = self.ce_loss(logits, y) + self.re_loss(x_rec, x)
                losses.append(loss.item())

                y_pred.append(logits.argmax(dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        loss_avg = float(np.mean(losses))

        return {"loss": loss_avg, "f1": f1}


    # ========= test metrics =========
    # def test_metrics(self, model=None):
    #     if model is None:
    #         model = self.model
    #     model.eval()
    #     model.to(self.device)

    #     y_true = []
    #     y_predict = []

    #     with torch.no_grad():
    #         for x, y in self.test_loader:
    #             # Xử lý input nếu là list (thường gặp trong dataset NSL-KDD tùy biến)
    #             x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
    #             y = y.to(self.device).long()

    #             # FedHomeLG model thường trả về tuple (logits, x_reconstruction)
    #             outputs = model(x)
    #             if isinstance(outputs, (list, tuple)):
    #                 logits = outputs[0]
    #             else:
    #                 logits = outputs

    #             y_predict.append(torch.argmax(logits, dim=1).cpu().numpy())
    #             y_true.append(y.cpu().numpy())

    #     y_true = np.concatenate(y_true)
    #     y_predict = np.concatenate(y_predict)

    #     # 1. Đếm số mẫu mỗi class tại client này (shape: [num_classes])
    #     s_pc = np.array([np.sum(y_true == i) for i in range(self.num_classes)])
        
    #     # 2. F1 từng class (labels đảm bảo shape luôn cố định là num_classes)
    #     f1_pc = metrics.f1_score(y_true, y_predict, labels=np.arange(self.num_classes), average=None, zero_division=0)

    #     # 3. Các chỉ số macro
    #     acc = metrics.accuracy_score(y_true, y_predict)
    #     pre = metrics.precision_score(y_true, y_predict, average='macro', zero_division=0)
    #     rec = metrics.recall_score(y_true, y_predict, average='macro', zero_division=0)
    #     f1 = metrics.f1_score(y_true, y_predict, average='macro', zero_division=0)

    #     # Trả về ĐÚNG 7 giá trị cho Server
    #     return acc, pre, rec, f1, len(y_true), f1_pc, s_pc
    def test_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        y_true = []
        y_predict = []

        with torch.no_grad():
            for x, y in self.test_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, x_reconstruction = self.model(x)
                y_predict.append(torch.argmax(output, dim=1).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_predict = np.concatenate(y_predict, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        acc = metrics.accuracy_score(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict, average='macro')
        recall = metrics.recall_score(y_true, y_predict, average='macro')
        f1 = metrics.f1_score(y_true, y_predict, average='macro')
        # auc = metrics.roc_auc_score(y_true, y_predict, average='micro')
        test_num = len(y_predict)
        # import pdb
        # pdb.set_trace()

        # return acc, precision, recall, f1, auc, test_num
        return acc, precision, recall, f1, test_num

    def train_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        list_loss = []
        # for calculate threshold
        with torch.no_grad():
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, x_reconstruction = self.model(x)
                ce_loss = self.ce_loss(output, y)
                re_loss = self.re_loss(x_reconstruction, x)
                loss = ce_loss + re_loss
                list_loss.append(loss.item())
        return list_loss
    # re_list = []
        # ========= metrics for convergence plotting =========
    # if model is None:
    #     model = self.model
    # model.eval()

    # ce_list = []
    # re_list = []
    # total_list = []

    # with torch.no_grad():
    #     for x, y in self.train_loader:
    #         x = x.to(self.device)
    #         y = y.to(self.device).long()

    #         logits, x_rec = self.model(x)

    #         loss_ce = self.ce_loss(logits, y)
    #         loss_re = self.re_loss(x_rec, x)
    #         loss_total = loss_ce + loss_re

    #         ce_list.append(loss_ce.item())
    #         re_list.append(loss_re.item())
    #         total_list.append(loss_total.item())

    # return ce_list, re_list, total_list


    # ========= Only update predictor params ========= code goocs
    # def set_parameters(self, model):
    #     for new, old in zip(model.parameters(), self.model.predictor.parameters()):
    #         old.data = new.data.clone()
    # ========= Gửi Encoder và Decoder lên Server =========
    def get_parameters(self):
        """
        Trả về phần Encoder và Decoder của Client để Server thực hiện trung bình cộng.
        """
        return self.model.encoder, self.model.decoder

# ========= Chỉ cập nhật Encoder và Decoder từ Server =========
    def set_parameters(self, server_model):
        """
        server_model: Model chứa tham số đã trung bình cộng từ server gửi xuống.
        self.model: Model hiện tại của Client.
        """
        # 1. Cập nhật Encoder
        for new_param, old_param in zip(server_model.encoder.parameters(), self.model.encoder.parameters()):
            old_param.data = new_param.data.clone()
            
        # 2. Cập nhật Decoder
        for new_param, old_param in zip(server_model.decoder.parameters(), self.model.decoder.parameters()):
            old_param.data = new_param.data.clone()

        # Không cập nhật self.model.predictor -> nó sẽ giữ nguyên những gì vừa train ở local