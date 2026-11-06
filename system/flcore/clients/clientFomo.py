import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics


class ClientFomo(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.data_type = args.data_type
        self.val_ratio = 0.2
        self.train_loader, self.val_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

        self.num_clients = args.num_clients
        self.old_model = copy.deepcopy(self.model)
        self.received_ids = []
        self.received_models = []
        self.weight_vector = torch.zeros(self.num_clients, device=self.device)

        self.train_samples = self.train_samples * (1 - self.val_ratio)

    def train(self):
        self.aggregate_parameters(self.val_loader)
        self.clone_model(self.model, self.old_model)
        self.model.train()

        for epoch in range(self.local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, data_type='train', type=self.data_type)
        val_idx = -int(self.val_ratio * len(train_data))
        val_data = train_data[val_idx:]
        train_data = train_data[:val_idx]
        train_loader = DataLoader(train_data, batch_size, drop_last=True, shuffle=False)
        val_loader = DataLoader(val_data, self.batch_size, drop_last=True, shuffle=False)
        return train_loader, val_loader

    def load_test_data(self, batch_size=1):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, data_type='test', type=self.data_type)
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
                loss = self.loss(output, y)
                list_loss.append(loss.item())
        return list_loss

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def receive_models(self, ids, models):
        self.received_ids = ids
        self.received_models = models

    def weight_cal(self, val_loader):
        weight_list = []
        L = self.recalculate_loss(self.old_model, val_loader)
        for received_model in self.received_models:
            params_dif = []
            for param_n, param_i in zip(received_model.parameters(), self.old_model.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)

            weight_list.append(
                (L - self.recalculate_loss(received_model, val_loader)) / (torch.norm(params_dif) + 1e-5))

        self.weight_vector_update(weight_list)

        return torch.tensor(weight_list)

    def weight_vector_update(self, weight_list):
        self.weight_vector = np.zeros(self.num_clients)
        for w, id in zip(weight_list, self.received_ids):
            self.weight_vector[id] += w.item()
        self.weight_vector = torch.tensor(self.weight_vector).to(self.device)

    def recalculate_loss(self, new_model, val_loader):
        L = 0
        for x, y in val_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = new_model(x)
            loss = self.loss(output, y)
            L += loss.item()
        # try:
        return L / len(val_loader)
        # except:
        #     import pdb
        #     pdb.set_trace()


    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.parameters(), received_model.parameters()):
            param.data += received_param.data.clone() * w

    def aggregate_parameters(self, val_loader):
        weights = self.weight_scale(self.weight_cal(val_loader))

        if len(weights) > 0:
            for param in self.model.parameters():
                param.data.zero_()

            for w, received_model in zip(weights, self.received_models):
                self.add_parameters(w, received_model)

    def weight_scale(self, weights):
        weights = torch.maximum(weights, torch.tensor(0))
        w_sum = torch.sum(weights)
        if w_sum > 0:
            weights = [w/w_sum for w in weights]
            return torch.tensor(weights)
        else:
            return torch.tensor([])

