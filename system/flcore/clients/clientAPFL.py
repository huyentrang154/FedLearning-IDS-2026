import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class ClientAPFL(object):
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
        # self.num_classes = args.num_classes

        self.alpha = args.alpha
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()
        self.model_per.train()

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

                output_per = self.model_per(x)
                loss_per = self.loss(output_per, y)
                self.optimizer_per.zero_grad()
                loss_per.backward()
                self.optimizer_per.step()

                self.alpha_update()

        for lp, p in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = (1 - self.alpha) * p + self.alpha * lp

    # https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.model.parameters(), self.model_per.parameters()):
            dif = p_params.data - l_params.data
            grad = self.alpha * p_params.grad.data + (1 - self.alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * self.alpha
        self.alpha = self.alpha - self.learning_rate * grad_alpha
        self.alpha = np.clip(self.alpha.item(), 0.0, 1.0)

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
        # if model is None:
        #     model = self.model
        model = self.model_per
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
        # if model is None:
        #     model = self.model
        model = self.model_per
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
