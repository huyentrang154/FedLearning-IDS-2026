import copy
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import read_client_data, NORMAL_LABEL
from torch.utils.data import DataLoader
from sklearn import metrics
import torch.nn.functional as F


class ClientNoRo(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.pat = args.patition
        self.alpha = args.alpha_dir
        self.poison_attack = args.poison_attack
        self.num_classes = args.num_classes
        self.poisoned_worker_ids = poisoned_worker_ids

        self.dataset = args.dataset
        self.normal_label = NORMAL_LABEL.get(self.dataset, 0)
        self.device = args.device
        self.id = id

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.num_clients = getattr(args, 'num_clients', 20)

        # Poisoning params
        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std

        # self.loss = nn.BCELoss()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.data_type = args.data_type
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

        # FedNoRo-specific state set by the server each round
        self.is_noisy = False      # True after GMM identifies this client as noisy
        self.teacher_model = None  # Global model from server used as KD teacher
        self.w_kd = 0.0            # KD loss weight, ramped up via sigmoid schedule

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
        if self.is_noisy and self.teacher_model is not None:
            self._train_kd()
        else:
            self._train_standard()

    def _train_standard(self):
        """Standard CE training used for warm-up and for clean clients in Stage 2."""
        for epoch in range(self.local_steps):
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)          # logits [B,C]
                if isinstance(output, (list, tuple)):
                    output = output[0]
                loss = self.loss(output, y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _train_kd(self):
        """Knowledge-distillation training for noisy clients in Stage 2.

        Combines soft KL-divergence against the teacher's predictions (KD term)
        and hard Cross-Entropy against the (potentially noisy) local labels,
        weighted by w_kd.
        """
        self.teacher_model.eval()

        for epoch in range(self.local_steps):
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(x)
                    if isinstance(teacher_logits, (list, tuple)):
                        teacher_logits = teacher_logits[0]
                student_logits = self.model(x)
                if isinstance(student_logits, (list, tuple)):
                    student_logits = student_logits[0]
                hard_loss = self.loss(student_logits, y.long())
                temperature = 2.0

                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature * temperature)

                loss = self.w_kd * kd_loss + (1 - self.w_kd) * hard_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_per_class_losses(self, model=None):
        """Return per-class mean CE loss used by the server for GMM identification.

        Returns a list of length `num_classes` computed on this client's training
        data using the provided model (typically the stage-1 global model snapshot).
        NaN is returned when a class is absent so the server can impute with column min.
        """
        if model is None:
            model = self.model
        model.eval()

        class_losses = {c: [] for c in range(self.num_classes)}

        with torch.no_grad():
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                sample_losses = F.cross_entropy(
                    logits,
                    y.long(),
                    reduction='none'
                )
                labels = y.long()
                for cls in range(self.num_classes):
                    mask = labels == cls
                    if mask.sum() > 0:
                        class_losses[cls].extend(
                            sample_losses[mask].cpu().numpy().tolist()
                        )
        return [
            float(np.mean(class_losses[c])) if len(class_losses[c]) > 0 else float('nan')
            for c in range(self.num_classes)
        ]

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(
            self.dataset, self.id, self.num_clients, partition=self.pat, alpha=self.alpha,
            data_type='train', type=self.data_type, poison_attack=self.poison_attack
        )
        if self.id in self.poisoned_worker_ids:
            train_data = self.apply_poisoning(train_data, self.noise_type, self.poisoned_sample_ratio, self.attack_noise_std)
        batch_size = min(batch_size, len(train_data)) if len(train_data) > 0 else batch_size
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(
            self.dataset, self.id, self.num_clients, partition=self.pat, alpha=self.alpha,
            data_type='test', type=self.data_type, poison_attack=self.poison_attack
        )
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

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
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1)
                y_predict.append(pred.cpu().numpy())
                y_true.append(y.cpu().numpy())
        y_predict = np.concatenate(y_predict, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        acc = metrics.accuracy_score(y_true, y_predict)
        precision = metrics.precision_score(y_true, y_predict, average='macro', zero_division=0)
        recall = metrics.recall_score(y_true, y_predict, average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true, y_predict, average='macro', zero_division=0)
        test_num = len(y_predict)

        return acc, precision, recall, f1, test_num

    def train_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        list_loss = []
        with torch.no_grad():
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss = self.loss(logits, y.long())
                list_loss.append(loss.item())
        return list_loss

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
