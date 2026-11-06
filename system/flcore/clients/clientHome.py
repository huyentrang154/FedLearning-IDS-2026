import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics


class ClientHome(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.num_clients = args.num_clients
        self.poisoned_worker_ids = poisoned_worker_ids
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.num_classes = args.num_classes

        self.ce_loss = nn.CrossEntropyLoss()
        self.re_loss = nn.MSELoss()
        self.global_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.data_type = args.data_type
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

    def apply_poisoning(self, train_data, noise_type, poisoned_sample_ratio, attack_noise_std):
        X = np.array([x.detach().cpu().numpy() for x, _ in train_data])
        Y = np.array([y.detach().cpu().item() for _, y in train_data])

        if noise_type == "label_flipping":
            possible_labels = np.arange(self.num_classes)
            flip_indices = np.random.choice(len(Y), size=int(len(Y) * poisoned_sample_ratio), replace=False)
            for idx in flip_indices:
                Y[idx] = np.random.choice(possible_labels[possible_labels != Y[idx]])
        elif noise_type == "Asymmetric":
            print("Poisoned Asymmetric")
            for idx in range(len(Y)):
                if np.random.rand() < poisoned_sample_ratio:
                    if Y[idx] == 0:
                        Y[idx] = np.random.randint(1, self.num_classes)
                    else:
                        Y[idx] = 0
        elif noise_type == "Symmetric":
            print("Poisoned Symmetric")
            possible_labels = np.arange(self.num_classes)
            for idx in range(len(Y)):
                if np.random.rand() < poisoned_sample_ratio:
                    Y[idx] = np.random.choice(possible_labels[possible_labels != Y[idx]])
        elif noise_type == "gaussian_noise":
            for idx in range(int(len(X) * poisoned_sample_ratio)):
                X[idx] += attack_noise_std * np.random.randn(*X[idx].shape)

        train_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)) for x, y in zip(X, Y)]
        return train_data

    def train(self):
        self.model.train()
        self.global_model = copy.deepcopy(self.model)

        for epoch in range(self.local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, x_reconstruction = self.model(x)
                ce_loss = self.ce_loss(output, y)
                re_loss = self.re_loss(x_reconstruction, x)
                loss = ce_loss + re_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def load_train_data(self, batch_size=None):
        train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
        batch_size = min(self.batch_size, len(train_data))
        if self.id in self.poisoned_worker_ids:
            train_data = self.apply_poisoning(train_data,
                        noise_type=self.noise_type,
                        poisoned_sample_ratio=self.poisoned_sample_ratio,
                        attack_noise_std=self.attack_noise_std)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

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

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# from utils.data_utils import read_client_data
# from torch.utils.data import DataLoader, random_split
# from sklearn import metrics

# # ALA gốc
# from utils.ALA import ALA


# class ClientHome(object):
#     def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
#         # ---- core ----
#         self.model = copy.deepcopy(args.model).to(args.device)
#         self.dataset = args.dataset
#         self.device = args.device
#         self.id = id
#         self.num_clients = args.num_clients

#         self.poisoned_worker_ids = poisoned_worker_ids
#         self.train_samples = train_samples
#         self.test_samples = test_samples

#         self.batch_size = args.batch_size
#         self.learning_rate = args.local_learning_rate
#         self.local_steps = args.local_steps

#         # ---- poisoning ----
#         self.noise_type = args.noise_type
#         self.num_of_poisoned_workers = args.num_of_poisoned_workers
#         self.poisoned_sample_ratio = args.poisoned_sample_ratio
#         self.attack_noise_std = args.attack_noise_std
#         self.num_classes = args.num_classes

#         # ---- losses & optim ----
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.re_loss = nn.MSELoss()
#         self.lambda_ce = getattr(args, "lambda_ce", 1.0)
#         self.lambda_re = getattr(args, "lambda_re", 1.0)

#         momentum = getattr(args, "momentum", 0.9)
#         weight_decay = getattr(args, "weight_decay", 0.0)
#         self.optimizer = torch.optim.SGD(
#             self.model.parameters(),
#             lr=self.learning_rate,
#             momentum=momentum,
#             weight_decay=weight_decay
#         )

#         self.global_model = copy.deepcopy(self.model).to(self.device)

#         # ---- data ----
#         self.data_type = args.data_type
#         self.raw_train_data = read_client_data(
#             self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type
#         )

#         # Apply poisoning nếu client thuộc nhóm poison
#         if self.id in self.poisoned_worker_ids:
#             print(f"[Client {self.id}] Apply poisoning: {self.noise_type}")
#             self.raw_train_data = self.apply_poisoning(
#                 self.raw_train_data,
#                 noise_type=self.noise_type,
#                 poisoned_sample_ratio=self.poisoned_sample_ratio,
#                 attack_noise_std=self.attack_noise_std
#             )

#         # Split train/val: 90% / 10%
#         train_split, val_split = self._split_train_val(self.raw_train_data, val_ratio=0.1)

#         # Loaders
#         self.train_loader = self._make_loader(train_split, self.batch_size, drop_last=True, shuffle=True)
#         self.val_loader   = self._make_loader(
#             val_split, max(1, min(self.batch_size, len(val_split))), drop_last=False, shuffle=False
#         )
#         self.test_loader  = self.load_test_data()

#         # ---- Early Stopping config ----
#         self.early_stop_metric = getattr(args, "early_stop_metric", "f1")  # 'f1' hoặc 'loss'
#         self.patience = getattr(args, "patience", 5)
#         self.min_delta = getattr(args, "min_delta", 1e-4)

#         # ---- ALA hyper-params ----
#         self.eta = getattr(args, "eta", 1.0)
#         self.rand_percent = getattr(args, "rand_percent", 20)     # % dữ liệu local để học weight
#         self.layer_idx = getattr(args, "layer_idx", 0)            # 0 = toàn bộ, >0: số lớp trên cùng
#         self.threshold = getattr(args, "ala_threshold", 0.1)
#         self.num_pre_loss = getattr(args, "ala_num_pre_loss", 10)

#         # ---- khởi tạo ALA với dữ liệu thô (list[(x,y)]) ----
#         self.ALA = ALA(
#             cid=self.id,
#             loss=self.ce_loss,                 # ALA tối ưu theo CE (logits)
#             train_data=self.raw_train_data,    # list of tuples, không phải DataLoader
#             batch_size=self.batch_size,
#             rand_percent=self.rand_percent,
#             layer_idx=self.layer_idx,
#             eta=self.eta,
#             device=self.device,
#             threshold=self.threshold,
#             num_pre_loss=self.num_pre_loss
#         )

#     # ---------------- Poisoning ----------------
#     def apply_poisoning(self, train_data, noise_type, poisoned_sample_ratio, attack_noise_std):
#         """Apply data poisoning to the client's training data (label_flipping | gaussian_noise)."""
#         # Đảm bảo dữ liệu ở CPU trước khi numpy()
#         X = [x.detach().cpu().numpy().astype(np.float32) for x, _ in train_data]
#         Y = [int(y.detach().cpu().item()) for _, y in train_data]

#         n = len(Y)
#         n_poison = int(n * poisoned_sample_ratio)
#         if n_poison <= 0:
#             return train_data

#         if noise_type == "label_flipping":
#             possible = np.arange(self.num_classes)
#             flip_idx = np.random.choice(n, size=n_poison, replace=False)
#             for idx in flip_idx:
#                 cur = Y[idx]
#                 Y[idx] = int(np.random.choice(possible[possible != cur]))

#         elif noise_type == "gaussian_noise":
#             poison_idx = np.random.choice(n, size=n_poison, replace=False)
#             for idx in poison_idx:
#                 X[idx] = (X[idx] + (attack_noise_std * np.random.randn(*X[idx].shape)).astype(np.float32))

#         # Trả về tensor CPU; sẽ .to(device) khi train
#         poisoned = [(torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)) for x, y in zip(X, Y)]
#         return poisoned

#     # ---------------- Data helpers ----------------
#     def _make_loader(self, data_list, batch_size, drop_last, shuffle):
#         batch_size = min(batch_size, max(1, len(data_list)))
#         return DataLoader(data_list, batch_size, drop_last=drop_last, shuffle=shuffle)

#     def _split_train_val(self, data_list, val_ratio=0.1, seed=42):
#         n_total = len(data_list)
#         if n_total <= 1:
#             # không đủ để tách, cho toàn bộ làm train, copy 1 mẫu làm val
#             return data_list, data_list

#         n_val = max(1, int(n_total * val_ratio))
#         n_train = n_total - n_val
#         if n_train <= 0:
#             n_train = n_total - 1
#             n_val = 1
#         train_set, val_set = random_split(
#             data_list,
#             [n_train, n_val],
#             generator=torch.Generator().manual_seed(seed)
#         )
#         return list(train_set), list(val_set)

#     def load_test_data(self, batch_size=1):
#         if batch_size is None:
#             batch_size = self.batch_size
#         test_data = read_client_data(self.dataset, self.id, self.num_clients,
#                                      data_type='test', type=self.data_type)
#         # test không bỏ mẫu cuối
#         return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

#     # ---------------- Validation eval ----------------
#     def evaluate_on_val(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()

#         y_true, y_pred = [], []
#         losses = []
#         with torch.no_grad():
#             for x, y in self.val_loader:
#                 x_t = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y_t = y.to(self.device).long()

#                 out = model(x_t)
#                 if isinstance(out, tuple):
#                     logits, x_recon = out[0], out[1]
#                 else:
#                     logits, x_recon = out, x_t

#                 ce = self.ce_loss(logits, y_t)
#                 re = self.re_loss(x_recon.float(), x_t.float())
#                 losses.append((self.lambda_ce * ce + self.lambda_re * re).item())

#                 y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
#                 y_true.append(y_t.detach().cpu().numpy())

#         if len(losses) == 0:
#             return {"loss": float("inf"), "f1": 0.0}

#         y_true = np.concatenate(y_true)
#         y_pred = np.concatenate(y_pred)
#         f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
#         return {"loss": float(np.mean(losses)), "f1": float(f1)}

#     # ---------------- Local Train (with Early Stopping) ----------------
#     def train(self):
#         self.model.train()
#         self.global_model = copy.deepcopy(self.model).to(self.device)

#         best_score = -float("inf") if self.early_stop_metric == "f1" else float("inf")
#         best_state = None
#         no_improve = 0

#         for epoch in range(self.local_steps):
#             # Train one epoch
#             for x, y in self.train_loader:
#                 x_t = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y_t = y.to(self.device).long()

#                 out = self.model(x_t)
#                 if isinstance(out, tuple):
#                     logits, x_recon = out[0], out[1]
#                 else:
#                     logits, x_recon = out, x_t  # fallback nếu không có recon

#                 ce = self.ce_loss(logits, y_t)
#                 re = self.re_loss(x_recon.float(), x_t.float())
#                 loss = self.lambda_ce * ce + self.lambda_re * re

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#             # Validate
#             val_stats = self.evaluate_on_val(self.model)
#             val_f1, val_loss = val_stats["f1"], val_stats["loss"]
#             print(f"[Client {self.id}] Epoch {epoch+1}/{self.local_steps} | "
#                   f"Val F1={val_f1:.4f} | Val Loss={val_loss:.6f}")

#             # Early stopping logic
#             improved = False
#             if self.early_stop_metric == "f1":
#                 if val_f1 > best_score + self.min_delta:
#                     improved = True
#                     best_score = val_f1
#             else:  # 'loss'
#                 if val_loss < best_score - self.min_delta:
#                     improved = True
#                     best_score = val_loss

#             if improved:
#                 best_state = copy.deepcopy(self.model.state_dict())
#                 no_improve = 0
#             else:
#                 no_improve += 1
#                 if no_improve >= self.patience:
#                     print(f"[Client {self.id}] Early stopping at epoch {epoch+1} "
#                           f"(best {self.early_stop_metric}={best_score:.6f})")
#                     break

#         # Khôi phục best model nếu có
#         if best_state is not None:
#             self.model.load_state_dict(best_state)

#     # ---------------- Metrics ----------------
#     def test_metrics(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()

#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for x, y in self.test_loader:
#                 x_t = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y_t = y.to(self.device).long()

#                 out = model(x_t)
#                 logits = out[0] if isinstance(out, tuple) else out
#                 y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
#                 y_true.append(y_t.cpu().numpy())

#         if len(y_pred) == 0:
#             return 0.0, 0.0, 0.0, 0.0, 0

#         y_pred = np.concatenate(y_pred, axis=0)
#         y_true = np.concatenate(y_true, axis=0)

#         acc = metrics.accuracy_score(y_true, y_pred)
#         precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
#         recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
#         f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
#         test_num = len(y_pred)
#         return acc, precision, recall, f1, test_num

#     def train_metrics(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()
#         losses = []
#         with torch.no_grad():
#             for x, y in self.train_loader:
#                 x_t = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y_t = y.to(self.device).long()

#                 out = model(x_t)
#                 if isinstance(out, tuple):
#                     logits, x_recon = out[0], out[1]
#                 else:
#                     logits, x_recon = out, x_t
#                 ce = self.ce_loss(logits, y_t)
#                 re = self.re_loss(x_recon.float(), x_t.float())
#                 losses.append((self.lambda_ce * ce + self.lambda_re * re).item())
#         return losses

#     # ---------------- Server -> Client ----------------
#     def set_parameters(self, received_global_model):
#         """
#         Nhận global model từ server và khởi tạo local bằng ALA gốc:
#           - Nếu layer_idx > 0: ALA sẽ giữ nguyên lower layers theo global, chỉ “pha trộn” các lớp trên cùng.
#           - Nếu layer_idx == 0: xem toàn bộ là higher và đều được ALA pha trộn.
#         ALA chỉ dùng CrossEntropy trên logits để học trọng số.
#         """
#         # đảm bảo cùng device
#         global_copy = copy.deepcopy(received_global_model).to(self.device)

#         # bật train mode & grad cho local model để ALA có thể backward
#         self.model.train()
#         for p in self.model.parameters():
#             p.requires_grad_(True)

#         if hasattr(self, "ALA") and self.ALA is not None:
#             # Chạy ALA trong bối cảnh có grad
#             with torch.enable_grad():
#                 self.ALA.adaptive_local_aggregation(global_copy, self.model)
#         else:
#             # fallback: copy nguyên xi
#             for new_param, old_param in zip(global_copy.parameters(), self.model.parameters()):
#                 old_param.data.copy_(new_param.data)


