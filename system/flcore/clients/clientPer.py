# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# from utils.data_utils import read_client_data
# from torch.utils.data import DataLoader
# from sklearn import metrics


# class ClientPer(object):
#     def __init__(self, args, id, train_samples, test_samples):
#         self.model = copy.deepcopy(args.model)
#         self.dataset = args.dataset
#         self.device = args.device
#         self.id = id

#         self.train_samples = train_samples
#         self.test_samples = test_samples
#         self.batch_size = args.batch_size
#         self.learning_rate = args.local_learning_rate
#         self.local_steps = args.local_steps

#         self.loss = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

#         self.train_loader = self.load_train_data()
#         self.test_loader = self.load_test_data()

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
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#     def load_train_data(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#         train_data = read_client_data(self.dataset, self.id, data_type='train')
#         return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

#     def load_test_data(self, batch_size=1):
#         if batch_size is None:
#             batch_size = self.batch_size
#         test_data = read_client_data(self.dataset, self.id, data_type='test')
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
#                 loss = self.loss(output, y)
#                 list_loss.append(loss.item())
#         return list_loss

#     def set_parameters(self, model):
#         for new_param, old_param in zip(model.parameters(), self.model.encoder.parameters()):
#             old_param.data = new_param.data.clone()
import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, random_split
from sklearn import metrics


class ClientPer(object):
    def __init__(self, args, id, train_samples, test_samples):
        # ----- Cấu hình cơ bản -----
        self.model = copy.deepcopy(args.model).to(args.device)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.num_clients = args.num_clients
        self.data_type = args.data_type
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # Loss & optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # ----- Early stopping config (giống FedAvg) -----
        self.patience = getattr(args, "patience", 5)
        self.min_delta = getattr(args, "min_delta", 1e-4)
        self.early_stop_metric = getattr(args, "early_stop_metric", "f1")  # 'f1' hoặc 'loss'

        # ----- Load full train data, rồi tách train / val -----
        full_train_data = self.load_train_data(return_raw=True)
        self.train_data, self.val_data = self._split_train_val(full_train_data, val_ratio=0.1)

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=len(self.val_data) if len(self.val_data) > 0 else 1,
            shuffle=False
        )

        self.test_loader = self.load_test_data()

    # =========================================================
    #                 TRAIN + EARLY STOPPING
    # =========================================================
    def train(self):
        self.model.train()

        best_state = None
        no_improve = 0
        best_score = -float("inf") if self.early_stop_metric == "f1" else float("inf")

        for epoch in range(self.local_steps):
            # --------- TRAIN ONE EPOCH ----------
            for x, y in self.train_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # --------- VALIDATION ----------
            val_stats = self.evaluate_on_val()
            val_loss, val_f1 = val_stats["loss"], val_stats["f1"]

            print(f"[Client {self.id}] Epoch {epoch+1}/{self.local_steps} | "
                  f"Val F1={val_f1:.4f} | Val Loss={val_loss:.6f}")

            # --------- EARLY STOPPING LOGIC ----------
            improved = False
            if self.early_stop_metric == "f1":
                if val_f1 > best_score + self.min_delta:
                    improved = True
                    best_score = val_f1
            else:  # 'loss'
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

    # =========================================================
    #                       DATA LOADING
    # =========================================================
    def load_train_data(self, batch_size=None, return_raw=False):
        """
        Nếu return_raw=True → trả về list dataset để chia train/val.
        Ngược lại, trả về DataLoader.
        """
        train_data = read_client_data(self.dataset, self.id,self.num_clients, data_type='train',type=self.data_type)

        if return_raw:
            return train_data

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=1):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test',type=self.data_type)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    # =========================================================
    #                    TRAIN / VAL SPLIT
    # =========================================================
    def _split_train_val(self, data_list, val_ratio=0.1, seed=42):
        n_total = len(data_list)
        if n_total <= 1:
            return data_list, data_list
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        train_set, val_set = random_split(
            data_list,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        return list(train_set), list(val_set)

    # =========================================================
    #                 VALIDATION EVALUATION
    # =========================================================
    def evaluate_on_val(self):
        self.model.eval()
        y_true, y_pred, losses = [], [], []

        # Nếu vì lí do nào đó val rỗng → trả giá trị xấu để không bị crash
        if len(self.val_data) == 0:
            return {"loss": 9999.0, "f1": 0.0}

        with torch.no_grad():
            for x, y in self.val_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss(output, y)
                losses.append(loss.item())

                y_pred.append(output.argmax(dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        avg_loss = float(np.mean(losses))

        return {"loss": avg_loss, "f1": f1}

    # =========================================================
    #                       TEST METRICS
    # =========================================================
    def test_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        y_true = []
        y_predict = []

        with torch.no_grad():
            for x, y in self.test_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]

                y_predict.append(output.argmax(dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_predict = np.concatenate(y_predict, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        acc = metrics.accuracy_score(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict, average='macro', zero_division=0)
        recall = metrics.recall_score(y_true, y_predict, average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true, y_predict, average='macro', zero_division=0)
        test_num = len(y_predict)

        return acc, precision, recall, f1, test_num

    # =========================================================
    #                  TRAIN LOSS (FOR LOGGING)
    # =========================================================
    def train_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        list_loss = []

        with torch.no_grad():
            for x, y in self.train_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss(output, y)
                list_loss.append(loss.item())
        return list_loss

    # =========================================================
    #         ĐỒNG BỘ PARAM TOÀN CỤC → ENCODER LOCAL
    # =========================================================
    def set_parameters(self, model):
        """
        FedPer: server chỉ gửi encoder, phần head/predictor là cá nhân hóa.
        """
        for new_param, old_param in zip(model.parameters(), self.model.encoder.parameters()):
            old_param.data = new_param.data.clone()
