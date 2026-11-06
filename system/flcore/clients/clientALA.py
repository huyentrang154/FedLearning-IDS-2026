import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.ALA import ALA


class ClientALA(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.poisoned_worker_ids = poisoned_worker_ids
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.num_clients = args.num_clients
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.data_type = args.data_type

        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.num_classes = args.num_classes

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
        if self.id in self.poisoned_worker_ids:
            print("client này bị label flipping")
            self.train_data = self.apply_poisoning(train_data,
                        noise_type=self.noise_type,
                        poisoned_sample_ratio=self.poisoned_sample_ratio,
                        attack_noise_std=self.attack_noise_std)
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()
        self.ALA = ALA(self.id, self.loss, self.train_data, self.batch_size,
                       self.rand_percent, self.layer_idx, self.eta, self.device)
    def apply_poisoning(self, train_data, noise_type, poisoned_sample_ratio, attack_noise_std):
        """Apply data poisoning to the client's training data."""
        X = np.array([x.numpy() for x, _ in train_data])
        Y = np.array([y.item() for _, y in train_data])

        if noise_type == "label_flipping":
            possible_labels = np.arange(self.num_classes)
            flip_indices = np.random.choice(
                len(Y), size=int(len(Y) * poisoned_sample_ratio), replace=False
            )
            for idx in flip_indices:
                Y[idx] = np.random.choice(
                    possible_labels[possible_labels != Y[idx]]
                )
        elif noise_type == "gaussian_noise":
            for idx in range(int(len(X) * poisoned_sample_ratio)):
                X[idx] += attack_noise_std * np.random.randn(*X[idx].shape)

        # Update poisoned data
        train_data = [(torch.tensor(x), torch.tensor(y)) for x, y in zip(X, Y)]
        return train_data
    
    def train(self):
        self.model.train()

        for epoch in range(self.local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = self.train_data
        # train_data = read_client_data(self.dataset, self.id, data_type='train', type=self.data_type)
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
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
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
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.loss(output, y)
                list_loss.append(loss.item())
        return list_loss

    # def set_parameters(self, model):
    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
    #         old_param.data = new_param.data.clone()
    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)
