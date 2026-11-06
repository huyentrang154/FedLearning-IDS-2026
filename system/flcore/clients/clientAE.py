# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn import metrics
# from utils.data_utils import read_client_data
# import copy

# class ClientAE(object):
#     def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
#         self.device = args.device
#         self.id = id
#         self.dataset = args.dataset
#         self.num_clients = args.num_clients
#         self.train_samples = train_samples
#         self.poisoned_worker_ids = poisoned_worker_ids
#         self.test_samples = test_samples
#         self.batch_size = args.batch_size
#         self.learning_rate = args.local_learning_rate
#         self.local_steps = args.local_steps
#         self.data_type = args.data_type
#         self.noise_type = args.noise_type
#         self.num_of_poisoned_workers = args.num_of_poisoned_workers
#         self.poisoned_sample_ratio = args.poisoned_sample_ratio
#         self.attack_noise_std = args.attack_noise_std
#         # Define 3 models
#         self.models = {
#             "normal": copy.deepcopy(args.model),
#             "abnormal": copy.deepcopy(args.model),
#             "hybrid": copy.deepcopy(args.model),
#         }

#         # Loss functions
#         self.recon_loss = nn.MSELoss()
#         self.cls_loss = nn.CrossEntropyLoss()

#         # Optimizers
#         self.optimizers = {
#             name: torch.optim.Adam(model.parameters(), lr=self.learning_rate)
#             for name, model in self.models.items()
#         }

#         self.train_loader = self.load_train_data()
#         self.test_loader = self.load_test_data()

#     # ---------------------------
#     # Training
#     # ---------------------------
#     # def train(self):
#     #     for ae_type, model in self.models.items():
#     #         model.train()
#     #         optimizer = self.optimizers[ae_type]

#     #         for epoch in range(self.local_steps):
#     #             for x, y in self.train_loader:
#     #                 x, y = x.to(self.device), y.to(self.device)
#     #                 optimizer.zero_grad()

#     #                 logits, recon = model(x)

#     #                 if ae_type == "hybrid":
#     #                     loss_recon = self.recon_loss(recon, x)
#     #                     loss_cls = self.cls_loss(logits, y)
#     #                     loss = loss_recon + loss_cls

#     #                 elif ae_type == "normal":
#     #                     idx = (y == 0)
#     #                     if idx.sum() == 0:
#     #                         continue
#     #                     _, recon = model(x[idx])
#     #                     loss = self.recon_loss(recon, x[idx])

#     #                 elif ae_type == "abnormal":
#     #                     idx = (y == 1)
#     #                     if idx.sum() == 0:
#     #                         continue
#     #                     _, recon = model(x[idx])
#     #                     loss = self.recon_loss(recon, x[idx])

#     #                 loss.backward()
#     #                 optimizer.step()
#     def train(self, max_epochs=50, tol=1e-4, patience=3):
#         for ae_type, model in self.models.items():
#             model.train()
#             optimizer = self.optimizers[ae_type]

#             if ae_type == "hybrid":
#                 # --- Pha 1: train reconstruction cho đến khi hội tụ ---
#                 best_loss = float("inf")
#                 patience_counter = 0

#                 for epoch in range(max_epochs):
#                     epoch_losses = []
#                     for x, y in self.train_loader:
#                         x, y = x.to(self.device), y.to(self.device)
#                         optimizer.zero_grad()

#                         _, recon = model(x)  # chỉ dùng reconstruction
#                         loss_recon = self.recon_loss(recon, x)

#                         loss_recon.backward()
#                         optimizer.step()
#                         epoch_losses.append(loss_recon.item())

#                     if len(epoch_losses) > 0:
#                         avg_loss = sum(epoch_losses) / len(epoch_losses)

#                         if abs(best_loss - avg_loss) < tol:
#                             patience_counter += 1
#                             if patience_counter >= patience:
#                                 print(f"Client {self.id}, {ae_type}: Recon early stopping at epoch {epoch}, loss={avg_loss:.4f}")
#                                 break
#                         else:
#                             patience_counter = 0
#                             best_loss = avg_loss

#                 # --- Pha 2: train classification (CE loss) cố định số vòng ---
#                 for epoch in range(self.local_steps):
#                     for x, y in self.train_loader:
#                         x, y = x.to(self.device), y.to(self.device)
#                         optimizer.zero_grad()

#                         logits, _ = model(x)
#                         loss_cls = self.cls_loss(logits, y)

#                         loss_cls.backward()
#                         optimizer.step()

#             else:
#                 # AE-normal / AE-abnormal: train như cũ
#                 best_loss = float("inf")
#                 patience_counter = 0

#                 for epoch in range(max_epochs):
#                     epoch_losses = []
#                     for x, y in self.train_loader:
#                         x, y = x.to(self.device), y.to(self.device)
#                         optimizer.zero_grad()

#                         if ae_type == "normal":
#                             idx = (y == 0)
#                             if idx.sum() == 0:
#                                 continue
#                             _, recon = model(x[idx])
#                             loss = self.recon_loss(recon, x[idx])

#                         elif ae_type == "abnormal":
#                             idx = (y == 1)
#                             if idx.sum() == 0:
#                                 continue
#                             _, recon = model(x[idx])
#                             loss = self.recon_loss(recon, x[idx])

#                         loss.backward()
#                         optimizer.step()
#                         epoch_losses.append(loss.item())

#                     if len(epoch_losses) > 0:
#                         avg_loss = sum(epoch_losses) / len(epoch_losses)
#                         if abs(best_loss - avg_loss) < tol:
#                             patience_counter += 1
#                             if patience_counter >= patience:
#                                 print(f"Client {self.id}, {ae_type}: Early stopping at epoch {epoch}, loss={avg_loss:.4f}")
#                                 break
#                         else:
#                             patience_counter = 0
#                             best_loss = avg_loss

#     # ---------------------------
#     # Testing with ensemble voting
#     # ---------------------------
#     def train_metrics(self, model_type=None):
#         results = {}

#         # Nếu chỉ muốn 1 model thì lấy ra
#         models_to_eval = {model_type: self.models[model_type]} if model_type else self.models

#         for ae_type, model in models_to_eval.items():
#             model.eval()
#             losses = []

#             with torch.no_grad():
#                 for x, y in self.train_loader:
#                     x, y = x.to(self.device), y.to(self.device)

#                     logits, recon = model(x)

#                     if ae_type == "hybrid":
#                         loss = self.recon_loss(recon, x)

#                     elif ae_type == "normal":
#                         idx = (y == 0)
#                         if idx.sum() == 0:
#                             continue
#                         _, recon = model(x[idx])
#                         loss = self.recon_loss(recon, x[idx])

#                     elif ae_type == "abnormal":
#                         idx = (y == 1)
#                         if idx.sum() == 0:
#                             continue
#                         _, recon = model(x[idx])
#                         loss = self.recon_loss(recon, x[idx])

#                     losses.append(loss.item())

#             results[ae_type] = sum(losses) / len(losses) if losses else None

#         return results
#     def test_metrics(self, threshold=0.05):
#         for model in self.models.values():
#             model.eval()

#         y_true = []
#         preds_normal, preds_abnormal, preds_hybrid, preds_ensemble = [], [], [], []

#         with torch.no_grad():
#             for x, y in self.test_loader:
#                 x, y = x.to(self.device), y.to(self.device)

#                 # -------------------------
#                 # 1) AE-Normal
#                 _, recon = self.models["normal"](x)
#                 loss = F.mse_loss(recon, x, reduction="mean").item()
#                 pred_normal = 0 if loss < threshold else 1

#                 # 2) AE-Abnormal
#                 _, recon = self.models["abnormal"](x)
#                 loss = F.mse_loss(recon, x, reduction="mean").item()
#                 pred_abnormal = 1 if loss < threshold else 0

#                 # 3) AE-Hybrid (classifier)
#                 logits, _ = self.models["hybrid"](x)
#                 pred_hybrid = torch.argmax(logits, dim=1).item()

#                 # Ensemble (majority voting)
#                 votes = [pred_normal, pred_abnormal, pred_hybrid]
#                 pred_final = 1 if votes.count(1) >= 2 else 0

#                 # -------------------------
#                 y_true.append(y.cpu().numpy())
#                 preds_normal.append(pred_normal)
#                 preds_abnormal.append(pred_abnormal)
#                 preds_hybrid.append(pred_hybrid)
#                 preds_ensemble.append(pred_final)

#         y_true = np.concatenate(y_true, axis=0)

#         def compute_all_metrics(y_true, y_pred):
#             acc = metrics.accuracy_score(y_true, y_pred)
#             precision = metrics.precision_score(y_true, y_pred, average="macro")
#             recall = metrics.recall_score(y_true, y_pred, average="macro")
#             f1 = metrics.f1_score(y_true, y_pred, average="macro")
#             return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

#         results = {
#             "normal": compute_all_metrics(y_true, preds_normal),
#             "abnormal": compute_all_metrics(y_true, preds_abnormal),
#             "hybrid": compute_all_metrics(y_true, preds_hybrid),
#             "ensemble": compute_all_metrics(y_true, preds_ensemble),
#             "n_samples": len(y_true)
#         }

#         return results


#     # def test_metrics(self, threshold=0.05):
#     #     for model in self.models.values():
#     #         model.eval()

#     #     y_true, y_pred = [], []

#     #     with torch.no_grad():
#     #         for x, y in self.test_loader:
#     #             x, y = x.to(self.device), y.to(self.device)

#     #             votes = []

#     #             # 1) AE-Normal
#     #             _, recon = self.models["normal"](x)
#     #             loss = F.mse_loss(recon, x, reduction="mean").item()
#     #             votes.append(0 if loss < threshold else 1)

#     #             # 2) AE-Abnormal
#     #             _, recon = self.models["abnormal"](x)
#     #             loss = F.mse_loss(recon, x, reduction="mean").item()
#     #             votes.append(1 if loss < threshold else 0)

#     #             # 3) AE-Hybrid
#     #             logits, _ = self.models["hybrid"](x)
#     #             pred = torch.argmax(logits, dim=1).item()
#     #             votes.append(pred)

#     #             # Majority voting
#     #             pred_final = 1 if votes.count(1) >= 2 else 0

#     #             y_true.append(y.cpu().numpy())
#     #             y_pred.append(pred_final)

#     #     y_true = np.concatenate(y_true, axis=0)
#     #     acc = metrics.accuracy_score(y_true, y_pred)
#     #     precision = metrics.precision_score(y_true, y_pred, average="macro")
#     #     recall = metrics.recall_score(y_true, y_pred, average="macro")
#     #     f1 = metrics.f1_score(y_true, y_pred, average="macro")
#     #     return acc, precision, recall, f1, len(y_true)

    
#     # ---------------------------
#     # Data loaders
#     # ---------------------------
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

#     # ---------------------------
#     # Federated parameters exchange
#     # ---------------------------
#     def set_parameters(self, model, ae_type="normal"):
#         for new_param, old_param in zip(model.parameters(), self.models[ae_type].parameters()):
#             old_param.data = new_param.data.clone()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.data_utils import read_client_data
import copy
import collections

class ClientAE(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.device = args.device
        self.id = id
        self.dataset = args.dataset
        self.num_clients = args.num_clients
        self.train_samples = train_samples
        self.poisoned_worker_ids = poisoned_worker_ids
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.data_type = args.data_type
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std

        # Define 3 models
        self.models = {
            "normal": copy.deepcopy(args.model),
            "abnormal": copy.deepcopy(args.model),
            "hybrid": copy.deepcopy(args.model),
        }

        # Loss functions
        self.recon_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()

        # Optimizers
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for name, model in self.models.items()
        }

        # Load data
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

        # === Count samples per class (0,1,...) from training set ===
        y_all = []
        for _, y in self.train_loader:
            y_all.extend(y.cpu().numpy().tolist())
        self.class_counts = dict(collections.Counter(y_all))
        # ví dụ: {0: 150, 1: 80}

    # ---------------------------
    # Training
    # ---------------------------
    def train(self, max_epochs=50, tol=1e-4, patience=3):
        for ae_type, model in self.models.items():
            model.train()
            optimizer = self.optimizers[ae_type]

            if ae_type == "hybrid":
                # --- Phase 1: reconstruction ---
                best_loss = float("inf")
                patience_counter = 0

                for epoch in range(max_epochs):
                    epoch_losses = []
                    for x, y in self.train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()
                        _, recon = model(x)
                        loss_recon = self.recon_loss(recon, x)
                        loss_recon.backward()
                        optimizer.step()
                        epoch_losses.append(loss_recon.item())

                    if epoch_losses:
                        avg_loss = sum(epoch_losses) / len(epoch_losses)
                        if abs(best_loss - avg_loss) < tol:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"Client {self.id}, {ae_type}: Recon early stop at epoch {epoch}, loss={avg_loss:.4f}")
                                break
                        else:
                            patience_counter = 0
                            best_loss = avg_loss

                # --- Phase 2: classification ---
                for epoch in range(self.local_steps):
                    for x, y in self.train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()
                        logits, _ = model(x)
                        loss_cls = self.cls_loss(logits, y)
                        loss_cls.backward()
                        optimizer.step()

            else:
                # AE-normal / AE-abnormal
                best_loss = float("inf")
                patience_counter = 0
                for epoch in range(max_epochs):
                    epoch_losses = []
                    for x, y in self.train_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        optimizer.zero_grad()

                        if ae_type == "normal":
                            idx = (y == 0)
                            if idx.sum() == 0:
                                continue
                            _, recon = model(x[idx])
                            loss = self.recon_loss(recon, x[idx])

                        elif ae_type == "abnormal":
                            idx = (y == 1)
                            if idx.sum() == 0:
                                continue
                            _, recon = model(x[idx])
                            loss = self.recon_loss(recon, x[idx])

                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())

                    if epoch_losses:
                        avg_loss = sum(epoch_losses) / len(epoch_losses)
                        if abs(best_loss - avg_loss) < tol:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"Client {self.id}, {ae_type}: Early stop at epoch {epoch}, loss={avg_loss:.4f}")
                                break
                        else:
                            patience_counter = 0
                            best_loss = avg_loss

    # ---------------------------
    # Train metrics
    # ---------------------------
    def train_metrics(self, model_type=None):
        results = {}
        models_to_eval = {model_type: self.models[model_type]} if model_type else self.models

        for ae_type, model in models_to_eval.items():
            model.eval()
            losses = []
            with torch.no_grad():
                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits, recon = model(x)

                    if ae_type == "hybrid":
                        loss = self.recon_loss(recon, x)
                    elif ae_type == "normal":
                        idx = (y == 0)
                        if idx.sum() == 0: continue
                        _, recon = model(x[idx])
                        loss = self.recon_loss(recon, x[idx])
                    elif ae_type == "abnormal":
                        idx = (y == 1)
                        if idx.sum() == 0: continue
                        _, recon = model(x[idx])
                        loss = self.recon_loss(recon, x[idx])
                    losses.append(loss.item())

            results[ae_type] = sum(losses)/len(losses) if losses else None
        return results

    # ---------------------------
    # Test metrics
    # ---------------------------
    def test_metrics(self, threshold=0.05):
        for model in self.models.values():
            model.eval()

        y_true = []
        preds_normal, preds_abnormal, preds_hybrid, preds_ensemble = [], [], [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # AE-Normal
                _, recon = self.models["normal"](x)
                loss = F.mse_loss(recon, x, reduction="mean").item()
                pred_normal = 0 if loss < threshold else 1

                # AE-Abnormal
                _, recon = self.models["abnormal"](x)
                loss = F.mse_loss(recon, x, reduction="mean").item()
                pred_abnormal = 1 if loss < threshold else 0

                # AE-Hybrid
                logits, _ = self.models["hybrid"](x)
                pred_hybrid = torch.argmax(logits, dim=1).item()

                # Ensemble
                votes = [pred_normal, pred_abnormal, pred_hybrid]
                pred_final = 1 if votes.count(1) >= 2 else 0

                y_true.append(y.cpu().numpy())
                preds_normal.append(pred_normal)
                preds_abnormal.append(pred_abnormal)
                preds_hybrid.append(pred_hybrid)
                preds_ensemble.append(pred_final)

        y_true = np.concatenate(y_true, axis=0)

        def compute_all(y_true, y_pred):
            return {
                "acc": metrics.accuracy_score(y_true, y_pred),
                "precision": metrics.precision_score(y_true, y_pred, average="macro"),
                "recall": metrics.recall_score(y_true, y_pred, average="macro"),
                "f1": metrics.f1_score(y_true, y_pred, average="macro")
            }

        results = {
            "normal": compute_all(y_true, preds_normal),
            "abnormal": compute_all(y_true, preds_abnormal),
            "hybrid": compute_all(y_true, preds_hybrid),
            "ensemble": compute_all(y_true, preds_ensemble),
            "n_samples": len(y_true)
        }
        return results

    # ---------------------------
    # Data loaders
    # ---------------------------
    def load_train_data(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
        if self.id in self.poisoned_worker_ids:
            train_data = self.apply_poisoning(train_data,
                        noise_type=self.noise_type,
                        poisoned_sample_ratio=self.poisoned_sample_ratio,
                        attack_noise_std=self.attack_noise_std)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=1):
        if batch_size is None: batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    # ---------------------------
    # Fed parameter exchange
    # ---------------------------
    def set_parameters(self, model, ae_type="normal"):
        for new_param, old_param in zip(model.parameters(), self.models[ae_type].parameters()):
            old_param.data = new_param.data.clone()
