import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics


class ClientUS(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
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
        self.input_channels = args.input_channels
        self.image_size = args.image_size

        self.ce_loss = nn.CrossEntropyLoss()
        self.re_loss = nn.MSELoss()


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_encoder = torch.optim.SGD(self.model.encoder.parameters(), lr=self.learning_rate)
        self.optimizer_decoder = torch.optim.SGD(self.model.decoder.parameters(), lr=self.learning_rate)
        self.optimizer_predictor = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)

        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.num_classes = args.num_classes

        self.data_type = args.data_type
        self.global_rounds = args.global_rounds

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.train_data = read_client_data(self.dataset, self.id, self.num_clients,data_type='train', type=self.data_type)
        if self.id in self.poisoned_worker_ids:
            print("client nay bi label flipping")
            self.train_data = self.apply_poisoning(train_data=self.train_data,
                        noise_type=self.noise_type,
                        poisoned_sample_ratio=self.poisoned_sample_ratio,
                        attack_noise_std=self.attack_noise_std)
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()
        # self.ALA = ALA(self.id, self.re_loss, self.train_data, self.batch_size,
        #                self.rand_percent, self.layer_idx, self.eta, self.device)
    
    def _process_batch(self, x, y):
        """Xử lý shape và device cho từng batch dữ liệu."""
        x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
        y = y.to(self.device).long()

        # Nếu là model CNN/CNNAE và dữ liệu đang ở dạng vector phẳng (như kết quả từ script của bạn)
        if self.input_channels>1 and len(x.shape) == 2 and self.image_size > 0:
            x = x.view(-1, self.input_channels, self.image_size, self.image_size)
            
        return x, y
    
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

                                                     
        # for epoch in range(self.local_steps):
        #     for i, (x, y) in enumerate(self.train_loader):
        #         if type(x) == type([]):
        #             x[0] = x[0].to(self.device)
        #         else:
        #             x = x.to(self.device)
        #         y = y.to(self.device)
        #         output, x_reconstruct = self.model(x)
        #         if R < self.global_rounds - 1:
        #             loss = self.re_loss(x_reconstruct, x)
        #             self.optimizer_encoder.zero_grad()
        #             self.optimizer_decoder.zero_grad()
        #             loss.backward()
        #             self.optimizer_encoder.step()
        #             self.optimizer_decoder.step()
        #         else:
        #             loss = self.ce_loss(output, y) + self.re_loss(x_reconstruct, x)
        #             self.opjjjjjjjjjtimizer.zero_grad()
        #             loss.backward()
        #             self.optimizer.step()

    def load_train_data(self, batch_size=None):
        train_data = self.train_data
        batch_size = min(self.batch_size, len(train_data))
        # if batch_size is None:
        #     batch_size = self.batch_size
        # train_data = self.train_data
        # if self.id in self.poisoned_worker_ids:
        #     train_data = self.apply_poisoning(train_data,
        #                 noise_type=self.noise_type,
        #                 poisoned_sample_ratio=self.poisoned_sample_ratio,
        #                 attack_noise_std=self.attack_noise_std)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=1):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        y_true = []
        y_predict = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = self._process_batch(x, y)
                output, _ = self.model(x)
                y_predict.append(torch.argmax(output, dim=1).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        y_predict = np.concatenate(y_predict, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        acc = metrics.accuracy_score(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict, average='macro')
        recall = metrics.recall_score(y_true, y_predict, average='macro')
        f1 = metrics.f1_score(y_true, y_predict, average='macro')
        test_num = len(y_predict)

        return acc, precision, recall, f1, test_num
    
    def train(self, R):
        self.model.train()
        ce_losses = []
        if R < self.global_rounds - 1:
            for param in self.model.predictor.parameters():
                param.requires_grad = False
        else:
            for param in self.model.predictor.parameters():
                param.requires_grad = True
        if R < self.global_rounds - 1:
            for epoch in range(self.local_steps):
                for i, (x, y) in enumerate(self.train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output, x_reconstruct = self.model(x)               
                    loss = self.re_loss(x_reconstruct, x)
                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()
                    loss.backward()
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
        if R == self.global_rounds - 1:
            for epoch in range(2500):
                epoch_loss_ce = []
                for i, (x, y) in enumerate(self.train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output, x_reconstruct = self.model(x)                                        
                    ce_loss = self.ce_loss(output, y)
                    epoch_loss_ce.append(ce_loss.item())
                    loss = ce_loss
                    self.optimizer_predictor.zero_grad()
                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()
                    loss.backward()
                    self.optimizer_predictor.step()
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
                ce_losses.append(round(sum(epoch_loss_ce) / len(epoch_loss_ce), 9))          
        return ce_losses
    
    def train_metrics(self, model=None, epoch=1):
        if model is None:
            model = self.model
        model.eval()
        list_loss = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = self._process_batch(x, y)
                output, _ = self.model(x)
                if epoch < self.global_rounds - 1:
                    loss = self.re_loss(_, x)
                else:
                    loss = self.ce_loss(output, y)
                list_loss.append(loss.item())
        return list_loss

    def set_parameters(self, model):
       for new_param, old_param in zip(model.parameters(), self.model.parameters()):
          old_param.data = new_param.data.clone()
    # def local_initialization(self, received_global_model):
    #     self.ALA.adaptive_local_aggregation(received_global_model, self.model)

# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# from utils.data_utils import read_client_data
# from torch.utils.data import DataLoader
# from sklearn import metrics


# class ClientUS(object):
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

#         # optimizers
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
#         self.optimizer_encoder = torch.optim.SGD(self.model.encoder.parameters(), lr=self.learning_rate)
#         self.optimizer_decoder = torch.optim.SGD(self.model.decoder.parameters(), lr=self.learning_rate)
#         self.optimizer_predictor = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)

#         # noise attack configs
#         self.noise_type = args.noise_type
#         self.num_of_poisoned_workers = args.num_of_poisoned_workers
#         self.poisoned_sample_ratio = args.poisoned_sample_ratio
#         self.attack_noise_std = args.attack_noise_std
#         self.num_classes = args.num_classes

#         # training configs
#         self.data_type = args.data_type
#         self.global_rounds = args.global_rounds
#         self.eta = args.eta
#         self.rand_percent = args.rand_percent
#         self.layer_idx = args.layer_idx

#         # load data
#         self.train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
#         if self.id in self.poisoned_worker_ids:
#             print(f"[Client {self.id}] ⚠️ Bị label flipping")
#             self.train_data = self.apply_poisoning(train_data=self.train_data,
#                         noise_type=self.noise_type,
#                         poisoned_sample_ratio=self.poisoned_sample_ratio,
#                         attack_noise_std=self.attack_noise_std)
#         self.train_loader = self.load_train_data()
#         self.test_loader = self.load_test_data()

#     # ================================
#     # Data poisoning
#     # ================================
#     def apply_poisoning(self, train_data, noise_type, poisoned_sample_ratio, attack_noise_std):
#         """Apply data poisoning to the client's training data."""
#         X = np.array([x.numpy() for x, _ in train_data])
#         Y = np.array([y.item() for _, y in train_data])

#         if noise_type == "label_flipping":
#             possible_labels = np.arange(self.num_classes)
#             flip_indices = np.random.choice(len(Y), size=int(len(Y) * poisoned_sample_ratio), replace=False)
#             for idx in flip_indices:
#                 Y[idx] = np.random.choice(possible_labels[possible_labels != Y[idx]])
#         elif noise_type == "gaussian_noise":
#             for idx in range(int(len(X) * poisoned_sample_ratio)):
#                 X[idx] += attack_noise_std * np.random.randn(*X[idx].shape)

#         train_data = [(torch.tensor(x), torch.tensor(y)) for x, y in zip(X, Y)]
#         return train_data

#     # ================================
#     # Training loop
#     # ================================
#     def train(self, R):
#         self.model.train()
#         ce_losses = []
#         checkpoint_path = f"checkpoint_client_{self.id}_R{R}.pt"

#         # Freeze predictor for all rounds except last
#         if R < self.global_rounds - 1:
#             for param in self.model.predictor.parameters():
#                 param.requires_grad = False
#         else:
#             for param in self.model.predictor.parameters():
#                 param.requires_grad = True

#         # ========== PHASE 1: AutoEncoder training (unsupervised) ==========
#         if R < self.global_rounds - 1:
#             for epoch in range(self.local_steps):
#                 for i, (x, y) in enumerate(self.train_loader):
#                     x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                     y = y.to(self.device)
#                     output, x_reconstruct = self.model(x)
#                     loss = self.re_loss(x_reconstruct, x)
#                     self.optimizer_encoder.zero_grad()
#                     self.optimizer_decoder.zero_grad()
#                     loss.backward()
#                     self.optimizer_encoder.step()
#                     self.optimizer_decoder.step()

#         # ========== PHASE 2: Fine-tuning with labels ==========
#         if R == self.global_rounds - 1:
#             start_epoch = 0

#             # Resume from checkpoint if available
#             if os.path.exists(checkpoint_path):
#                 checkpoint = torch.load(checkpoint_path, map_location=self.device)
#                 print(f"[Client {self.id}] 🔁 Resume checkpoint từ epoch {checkpoint['epoch'] + 1}")
#                 self.model.load_state_dict(checkpoint["model_state"])
#                 self.optimizer_encoder.load_state_dict(checkpoint["optimizer_encoder"])
#                 self.optimizer_decoder.load_state_dict(checkpoint["optimizer_decoder"])
#                 self.optimizer_predictor.load_state_dict(checkpoint["optimizer_predictor"])
#                 start_epoch = checkpoint["epoch"] + 1
#                 ce_losses = checkpoint["ce_losses"]

#             print(f"[Client {self.id}] ▶️ Bắt đầu fine-tune từ epoch {start_epoch}")

#             for epoch in range(start_epoch, 4000):
#                 epoch_loss_ce = []
#                 for i, (x, y) in enumerate(self.train_loader):
#                     x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                     y = y.to(self.device)

#                     output, x_reconstruct = self.model(x)
#                     ce_loss = self.ce_loss(output, y)
#                     epoch_loss_ce.append(ce_loss.item())

#                     self.optimizer_predictor.zero_grad()
#                     self.optimizer_encoder.zero_grad()
#                     self.optimizer_decoder.zero_grad()
#                     ce_loss.backward()
#                     self.optimizer_predictor.step()
#                     self.optimizer_encoder.step()
#                     self.optimizer_decoder.step()

#                 mean_loss = round(sum(epoch_loss_ce) / len(epoch_loss_ce), 9)
#                 ce_losses.append(mean_loss)

#                 # Save checkpoint every 50 epochs
#                 if (epoch + 1) % 50 == 0 or (epoch + 1) == 4000:
#                     torch.save({
#                         "round": R,
#                         "epoch": epoch,
#                         "model_state": self.model.state_dict(),
#                         "optimizer_encoder": self.optimizer_encoder.state_dict(),
#                         "optimizer_decoder": self.optimizer_decoder.state_dict(),
#                         "optimizer_predictor": self.optimizer_predictor.state_dict(),
#                         "ce_losses": ce_losses,
#                     }, checkpoint_path)
#                     print(f"[Client {self.id}] 💾 Lưu checkpoint tại epoch {epoch + 1}")

#             print(f"[Client {self.id}] ✅ Hoàn tất fine-tuning ({len(ce_losses)} epochs)")

#         return ce_losses

#     # ================================
#     # Data loaders
#     # ================================
#     def load_train_data(self, batch_size=None):
#         train_data = self.train_data
#         batch_size = min(self.batch_size, len(train_data)) if batch_size is None else batch_size
#         return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

#     def load_test_data(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#         test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test')
#         return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

#     # ================================
#     # Metrics
#     # ================================
#     def test_metrics(self, model=None):
#         if model is None:
#             model = self.model
#         model.eval()
#         y_true, y_predict = [], []

#         with torch.no_grad():
#             for x, y in self.test_loader:
#                 x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y = y.to(self.device)
#                 output, _ = model(x)
#                 y_predict.append(torch.argmax(output, dim=1).cpu().numpy())
#                 y_true.append(y.cpu().numpy())

#         y_predict = np.concatenate(y_predict, axis=0)
#         y_true = np.concatenate(y_true, axis=0)

#         acc = metrics.accuracy_score(y_true, y_predict)
#         precision = metrics.precision_score(y_true, y_predict, average='macro')
#         recall = metrics.recall_score(y_true, y_predict, average='macro')
#         f1 = metrics.f1_score(y_true, y_predict, average='macro')
#         test_num = len(y_predict)
#         return acc, precision, recall, f1, y_true, y_predict, test_num

#     def train_metrics(self, model=None, epoch=1):
#         if model is None:
#             model = self.model
#         model.eval()
#         list_loss = []

#         with torch.no_grad():
#             for x, y in self.train_loader:
#                 x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
#                 y = y.to(self.device)
#                 output, x_reconstruct = model(x)
#                 if epoch < self.global_rounds - 1:
#                     loss = self.re_loss(x_reconstruct, x)
#                 else:
#                     loss = self.ce_loss(output, y) + self.re_loss(x_reconstruct, x)
#                 list_loss.append(loss.item())

#         return list_loss

#     # ================================
#     # Utilities
#     # ================================
#     def set_parameters(self, model):
#         for new_param, old_param in zip(model.parameters(), self.model.parameters()):
#             old_param.data = new_param.data.clone()
