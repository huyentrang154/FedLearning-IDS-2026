import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, random_split
from sklearn import metrics

class ClientRep(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.num_clients = args.num_clients
        self.poisoned_worker_ids = poisoned_worker_ids
        self.num_classes = args.num_classes
        self.data_type = args.data_type
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        
        # Quan trọng: Cập nhật số lượng mẫu từ dữ liệu thực tế
        self.train_samples = train_samples
        self.test_samples = test_samples
        
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.encoder.parameters(), lr=self.learning_rate)
        self.optimizer_per = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)

        # Load dữ liệu
        full_train_data = read_client_data(
            self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type
        )
        if self.id in self.poisoned_worker_ids:
            full_train_data = self.apply_poisoning(
                full_train_data,
                noise_type=self.noise_type,
                poisoned_sample_ratio=self.poisoned_sample_ratio,
                attack_noise_std=self.attack_noise_std,)
        self.train_loader = DataLoader(full_train_data, self.batch_size, shuffle=True, drop_last=True)
        test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type)
        self.test_loader = DataLoader(test_data, self.batch_size, drop_last=False, shuffle=False)

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
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.predictor.parameters():
            param.requires_grad = True

        for epoch in range(self.local_steps-1):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, x_reconstruction = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()

        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.predictor.parameters():
            param.requires_grad = False

        for epoch in range(1):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, x_reconstruction = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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

    def train_metrics(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                if isinstance(out, tuple): out = out[0]
                losses.append(self.loss(out, y.long()).item())
        return losses

    def set_parameters(self, global_encoder):
        for new_p, old_p in zip(global_encoder.parameters(), self.model.encoder.parameters()):
            old_p.data = new_p.data.clone()
