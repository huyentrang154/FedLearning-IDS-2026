import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics
import math


class ClientApple(object):
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
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()
        # self.num_classes = args.num_classesz

        self.drlr = args.dr_learning_rate
        self.num_clients = args.num_clients
        self.lamda = 1
        self.mu = args.mu
        self.L = int(args.L * args.global_rounds)
        self.learning_rate = self.learning_rate * self.num_clients

        self.model_cs = []

        self.ps = [1/args.num_clients for _ in range(self.num_clients)]
        self.p0 = None
        self.model_c = copy.deepcopy(self.model)

    def train(self, R):
        self.model.train()

        for epoch in range(self.local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.aggregate_parameters()

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for param_c, param in zip(self.model_cs[self.id].parameters(), self.model.parameters()):
                    param_c.data = param_c - self.learning_rate * param.grad * self.ps[self.id]

                for cid in range(self.num_clients):
                    cnt = 0
                    p_grad = 0
                    for param_c, param in zip(self.model_cs[cid].parameters(), self.model.parameters()):
                        p_grad += torch.mean(param.grad * param_c).item()
                        cnt += 1
                    p_grad = p_grad / cnt
                    p_grad = p_grad + self.lamda * self.mu * (self.ps[cid] - self.p0[cid])
                    self.ps[cid] = self.ps[cid] - self.drlr * p_grad

        if R < self.L:
            self.lamda = (math.cos(R * math.pi / self.L) + 1) / 2
        else:
            self.lamda = 0

        # recover self.model_cs[self.id] for other clients
        for param_c, param_ in zip(self.model_cs[self.id].parameters(), self.model_c.parameters()):
            param_c.data = param_.data.clone()

        self.model_c = copy.deepcopy(self.model)

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, data_type='train', type=self.data_type)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

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
                loss = self.loss(output, y)
                list_loss.append(loss.item())
        return list_loss

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_models(self, model_cs):
        self.model_cs = model_cs

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.model_cs) > 0)

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.ps, self.model_cs):
            self.add_parameters(w, client_model)