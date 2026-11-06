import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, random_split
from sklearn import metrics
from collections import defaultdict

class ClientGH(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model).to(args.device)
        self.dataset = args.dataset
        self.num_clients = args.num_clients
        self.device = args.device
        self.id = id

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.noise_type = args.noise_type
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std
        self.num_classes = args.num_classes
        self.poisoned_worker_ids = poisoned_worker_ids

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.data_type = args.data_type

        # Load data
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
        for epoch in range(self.local_steps):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).long()

                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]

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

    def test_metrics_global(self, model=None):
        """Đánh giá trên tập test toàn cục (dùng cho Server log)"""
        if model is None:
            model = self.model
        model.eval()

        # Giả sử hàm read_client_data hỗ trợ lấy full test bằng cách truyền id đặc biệt hoặc concat bên ngoài
        # Ở đây dùng tạm test_loader nội bộ nếu dataset là IID, 
        # nhưng tốt nhất nên dùng chung 1 loader global từ server truyền vào.
        return self.test_metrics(model)

    def train_metrics(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device); y = y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple): output = output[0]
                loss = self.loss(output, y.long())
                losses.append(loss.item())
        return losses

    def set_parameters(self, predictor_model):
        """FedGH: Chỉ cập nhật phần Predictor từ Server"""
        for new_param, old_param in zip(predictor_model.parameters(), self.model.predictor.parameters()):
            old_param.data = new_param.data.clone()

    def collect_protos(self):
        self.model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                # Lấy output từ encoder (Feature Representation)
                rep = self.model.encoder(x)
                rep = torch.flatten(rep, 1)
                for i, yy in enumerate(y):
                    protos[yy.item()].append(rep[i, :].detach().clone())

        self.protos = agg_func(protos)

    def load_train_data(self):
        train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
        if self.id in self.poisoned_worker_ids:
            train_data = self.apply_poisoning(train_data, self.noise_type, self.poisoned_sample_ratio, self.attack_noise_std)
        return DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)

    def load_test_data(self):
        test_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type)
        return DataLoader(test_data, self.batch_size, drop_last=False, shuffle=False)

def agg_func(protos):
    for label, proto_list in protos.items():
        if len(proto_list) > 0:
            protos[label] = torch.stack(proto_list).mean(dim=0)
    return protos
