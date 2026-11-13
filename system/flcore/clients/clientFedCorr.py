import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from sklearn import metrics


class ClientFedCorr(object):
    def __init__(self, args, id, poisoned_worker_ids, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.poisoned_worker_ids = poisoned_worker_ids
        self.num_clients = args.num_clients
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.num_classes = args.num_classes
        self.data_type = args.data_type

        self.noise_type = args.noise_type
        self.num_of_poisoned_workers = args.num_of_poisoned_workers
        self.poisoned_sample_ratio = args.poisoned_sample_ratio
        self.attack_noise_std = args.attack_noise_std

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.5
        )

        full_train_data = read_client_data(self.dataset, self.id, self.num_clients, data_type='train', type=self.data_type)
        if self.id in self.poisoned_worker_ids:
            full_train_data = self.apply_poisoning(
                full_train_data,
                noise_type=self.noise_type,
                poisoned_sample_ratio=self.poisoned_sample_ratio,
                attack_noise_std=self.attack_noise_std,
            )

        # Mutable train data — label correction modifies this list in place
        self.train_data = list(full_train_data)
        test_data = read_client_data(
            self.dataset, self.id, self.num_clients, data_type='test', type=self.data_type
        )
        self.test_loader = DataLoader(test_data, batch_size=1, drop_last=False, shuffle=False)

        # Set by server each round before train()
        self.mu = 0.0          # estimated noise level (proximal term scale)
        self.global_model = None  # copy of global model for proximal term

    # ------------------------------------------------------------------
    # Local training (with optional FedProx-style proximal term)
    # ------------------------------------------------------------------
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
    
    def train(self, beta=0.0):
        """SGD with proximal regularisation: loss += beta * mu * ||w - w_g||.

        Mirrors the original FedCorr update_weights():
        - proximal term skipped on batch_idx==0 (local model == global → w_diff=0
          which makes grad of sqrt(0) blow up)
        - eps inside sqrt as extra safety
        """
        self.model.train()
        # original LocalUpdate: shuffled loader without drop_last, and a fresh SGD optimizer
        # (momentum 0.5) for every local update, so no momentum is carried between rounds
        loader = DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)

        for _ in range(self.local_steps):
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device).long()

                self.optimizer.zero_grad()
                out = self.model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = self.loss_fn(out, y)

                # Skip batch 0: local model == global model → w_diff = 0 → grad explodes
                if beta > 0 and self.mu > 0 and self.global_model is not None and batch_idx > 0:
                    w_diff = sum(
                        torch.pow(torch.norm(p - pg), 2)
                        for p, pg in zip(
                            self.model.parameters(),
                            self.global_model.parameters()
                        )
                    )
                    loss = loss + beta * self.mu * torch.sqrt(w_diff + 1e-12)

                loss.backward()
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Output / loss extraction (used by server for LID and correction)
    # ------------------------------------------------------------------

    def get_outputs_and_loss(self, model=None):
        """Return softmax probs (N, C) and per-sample CE losses (N,).

        Uses the provided model — typically the locally trained model (Stage 1
        LID computation) or the current global model (label correction).
        """
        if model is None:
            model = self.model
        model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loader = DataLoader(
            self.train_data, self.batch_size, shuffle=False, drop_last=False
        )
        all_outputs, all_losses = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device).long()
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                probs = F.softmax(out, dim=1)
                # as in the original get_output(): the per-sample loss is the cross-entropy
                # applied to the softmax outputs (not to the logits)
                losses = loss_fn(probs, y)
                all_outputs.append(probs.cpu().numpy())
                all_losses.append(losses.cpu().numpy())
        return (
            np.concatenate(all_outputs, axis=0),
            np.concatenate(all_losses, axis=0),
        )

    # ------------------------------------------------------------------
    # Label correction
    # ------------------------------------------------------------------

    def relabel(self, indices, new_labels):
        """Replace labels at the given sample positions in train_data."""
        for idx, lbl in zip(indices, new_labels):
            x, _ = self.train_data[idx]
            self.train_data[idx] = (x, torch.tensor(int(lbl), dtype=torch.long))

    # ------------------------------------------------------------------
    # Standard evaluation helpers (mirror clientAvg.py)
    # ------------------------------------------------------------------

    def test_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device).long()
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                y_pred.append(torch.argmax(out, dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, precision, recall, f1, len(y_true)

    def train_metrics(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        loader = DataLoader(
            self.train_data, self.batch_size, shuffle=False, drop_last=False
        )
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device).long()
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                losses.append(self.loss_fn(out, y).item())
        return losses

    def set_parameters(self, model):
        for new_p, old_p in zip(model.parameters(), self.model.parameters()):
            old_p.data = new_p.data.clone()
