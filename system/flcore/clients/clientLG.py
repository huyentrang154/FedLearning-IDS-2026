# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# from utils.data_utils import read_client_data
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from sklearn.preprocessing import label_binarize
# from sklearn import metrics


# class ClientLG(object):
#     def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
#         self.model = copy.deepcopy(args.model)
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

#         self.loss = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
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

#         for epoch in range(self.local_steps):
#             for i, (x, y) in enumerate(self.train_loader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 output = self.model(x)
#                 if isinstance(output, tuple):
#                     output = output[0]
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#     def load_train_data(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#         train_data = read_client_data(self.dataset, self.id,self.num_clients, data_type='train', type=self.data_type)
#         if self.id in self.poisoned_worker_ids:
#             train_data = self.apply_poisoning(train_data,
#                         noise_type=self.noise_type,
#                         poisoned_sample_ratio=self.poisoned_sample_ratio,
#                         attack_noise_std=self.attack_noise_std)
#         return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
#     def load_test_data(self, batch_size=1):
#         if batch_size is None:
#             batch_size = self.batch_size
#         test_data = read_client_data(self.dataset, self.id, self.num_clients,data_type='test', type=self.data_type)
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
#                 output = self.model(x)
#                 if isinstance(output, tuple):
#                     output = output[0]
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
#                 output = self.model(x)
#                 if isinstance(output, tuple):
#                     output = output[0]
#                 loss = self.loss(output, y)
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


class ClientLG(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        # ---- Core ----
        self.model = copy.deepcopy(args.model).to(args.device)
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

        # ---- Poisoning ----
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.num_classes = args.num_classes
        self.data_type = args.data_type

        # ---- Loss & optimizer ----
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # ---- Load & split data ----
        full_train_data = read_client_data(
            self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type
        )
        if self.id in self.poisoned_worker_ids:
            full_train_data = self.apply_poisoning(
                full_train_data,
                noise_type=self.noise_type,
                poisoned_sample_ratio=self.poisoned_sample_ratio,
                attack_noise_std=self.attack_noise_std,
            )

        # Chia train/val (90/10)
        self.train_data, self.val_data = self._split_train_val(full_train_data, val_ratio=0.1)
        self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False)
        self.test_loader = self.load_test_data()

        # ---- Early Stopping ----
        self.patience = getattr(args, "patience", 5)
        self.min_delta = getattr(args, "min_delta", 1e-4)
        self.early_stop_metric = getattr(args, "early_stop_metric", "f1")  # "f1" hoặc "loss"

    # ---------------- Poisoning ----------------
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

    # ---------------- Split train/val ----------------
    def _split_train_val(self, data_list, val_ratio=0.1, seed=42):
        n_total = len(data_list)
        if n_total <= 1:
            return data_list, data_list
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        train_set, val_set = random_split(
            data_list, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        return list(train_set), list(val_set)

    # ---------------- Train with Early Stopping ----------------
    def train(self):
        self.model.train()
        best_state = None
        no_improve = 0
        best_score = -float("inf") if self.early_stop_metric == "f1" else float("inf")

        for epoch in range(self.local_steps):
            epoch_losses = []

            # ---- Train one epoch ----
            for x, y in self.train_loader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device).long()
                out = self.model(x)
                if isinstance(out, tuple):
                    out = out[0]

                loss = self.loss(out, y)
                epoch_losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = np.mean(epoch_losses)

            # ---- Validation ----
            val_stats = self.evaluate_on_val()
            val_loss, val_f1 = val_stats["loss"], val_stats["f1"]

            print(f"[Client {self.id}] Epoch {epoch+1}/{self.local_steps} | "
                  f"TrainLoss={avg_train_loss:.4f} | ValLoss={val_loss:.6f} | ValF1={val_f1:.4f}")

            # ---- Early Stopping logic ----
            improved = False
            if self.early_stop_metric == "f1":
                if val_f1 > best_score + self.min_delta:
                    improved = True
                    best_score = val_f1
            else:  # loss
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

        # ---- Restore best model ----
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ---------------- Validation ----------------
    def evaluate_on_val(self):
        self.model.eval()
        y_true, y_pred = [], []
        losses = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device).long()
                out = self.model(x)
                if isinstance(out, tuple):
                    out = out[0]

                loss = self.loss(out, y)
                losses.append(loss.item())
                y_pred.append(torch.argmax(out, dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        avg_loss = float(np.mean(losses))
        return {"loss": avg_loss, "f1": f1}

    # ---------------- Test ----------------
    def test_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        y_true, y_predict = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                y_predict.append(torch.argmax(output, dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_predict = np.concatenate(y_predict, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        acc = metrics.accuracy_score(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict, average='macro', zero_division=0)
        recall = metrics.recall_score(y_true, y_predict, average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true, y_predict, average='macro', zero_division=0)
        test_num = len(y_predict)
        return acc, precision, recall, f1, test_num
    
    def test_metrics_global(self, model=None):
        """
        Kiểm tra độ chính xác trên TOÀN BỘ tập dữ liệu test (không theo ID của client).
        """
        if model is None:
            model = self.model
        model.eval()

        # Load toàn bộ dữ liệu test bằng cách truyền id=-1 hoặc sửa logic read_client_data
        # Giả định read_client_data hỗ trợ lấy full data khi truyền tham số thích hợp
        # Ở đây ta sẽ load lại toàn bộ thay vì dùng self.test_loader của riêng client
        full_test_data = read_client_data(self.dataset, -1, self.num_clients, data_type='test', type=self.data_type)
        full_test_loader = DataLoader(full_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)

        y_true = []
        y_predict = []

        with torch.no_grad():
            for x, y in full_test_loader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output, _ = model(x)
                y_predict.append(torch.argmax(output, dim=1).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_predict = np.concatenate(y_predict, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # Tính toán các chỉ số
        acc = metrics.accuracy_score(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict, average='macro', zero_division=0)
        recall = metrics.recall_score(y_true, y_predict, average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true, y_predict, average='macro', zero_division=0)
        
        return acc, precision, recall, f1, len(y_true)

    def train_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.loss(output, y)
                losses.append(loss.item())
        return losses

    # ---------------- FedLG parameter update ----------------
    def set_parameters(self, model):
        # chỉ đồng bộ predictor (encoder giữ local)
        for new_param, old_param in zip(model.parameters(), self.model.predictor.parameters()):
            old_param.data = new_param.data.clone()

    # ---------------- Data Loaders ----------------
    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, self.num_clients,
                                      data_type='train', type=self.data_type)
        if self.id in self.poisoned_worker_ids:
            train_data = self.apply_poisoning(train_data,
                                              noise_type=self.noise_type,
                                              poisoned_sample_ratio=self.poisoned_sample_ratio,
                                              attack_noise_std=self.attack_noise_std)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=1):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.num_clients,
                                     data_type='test', type=self.data_type)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
