import copy
import torch
import torch.nn as nn
import numpy as np

from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, random_split
from sklearn import metrics

torch.cuda.empty_cache()


class ClientTrimmedMean(object):

    def __init__(
        self,
        args,
        id,
        poisoned_worker_ids,
        train_samples,
        test_samples
    ):

        self.model = copy.deepcopy(
            args.model
        ).to(args.device)

        self.dataset = args.dataset
        self.device = args.device

        self.id = id

        self.num_clients = args.num_clients

        self.input_channels = (
            args.input_channels
        )

        self.image_size = args.image_size

        self.poisoned_worker_ids = (
            poisoned_worker_ids
        )

        self.train_samples = train_samples
        self.test_samples = test_samples

        self.batch_size = args.batch_size

        self.learning_rate = (
            args.local_learning_rate
        )

        self.local_steps = args.local_steps

        # =====================================
        # POISONING
        # =====================================
        self.noise_type = args.noise_type

        self.num_of_poisoned_workers = (
            args.num_of_poisoned_workers
        )

        self.poisoned_sample_ratio = (
            args.poisoned_sample_ratio
        )

        self.attack_noise_std = (
            args.attack_noise_std
        )

        self.num_classes = args.num_classes

        # =====================================
        # LOSS & OPTIMIZER
        # =====================================
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # =====================================
        # LOAD DATA
        # =====================================
        self.data_type = args.data_type

        full_train_data = read_client_data(
            self.dataset,
            self.id,
            self.num_clients,
            data_type='train',
            type=self.data_type
        )

        # =====================================
        # APPLY POISONING
        # =====================================
        if self.id in self.poisoned_worker_ids:

            full_train_data = (
                self.apply_poisoning(
                    full_train_data,
                    noise_type=self.noise_type,
                    poisoned_sample_ratio=self.poisoned_sample_ratio,
                    attack_noise_std=self.attack_noise_std
                )
            )

        # =====================================
        # TRAIN / VAL SPLIT
        # =====================================
        self.train_data, self.val_data = (
            self._split_train_val(
                full_train_data,
                val_ratio=0.1
            )
        )

        self.train_loader = DataLoader(
            self.train_data,
            self.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=len(self.val_data),
            shuffle=False
        )

        self.test_loader = self.load_test_data()

        # =====================================
        # EARLY STOPPING
        # =====================================
        self.patience = getattr(
            args,
            "patience",
            5
        )

        self.min_delta = getattr(
            args,
            "min_delta",
            1e-4
        )

        self.early_stop_metric = getattr(
            args,
            "early_stop_metric",
            "f1"
        )

    # =====================================================
    # PROCESS BATCH
    # =====================================================
    def _process_batch(self, x, y):

        x = (
            x[0].to(self.device)
            if isinstance(x, list)
            else x.to(self.device)
        )

        y = y.to(self.device).long()

        # =====================================
        # RESHAPE FOR CNN
        # =====================================
        if (
            self.input_channels > 1
            and len(x.shape) == 2
            and self.image_size > 0
        ):

            expected_dim = (
                self.input_channels
                * self.image_size
                * self.image_size
            )

            if x.shape[1] == expected_dim:

                x = x.view(
                    -1,
                    self.input_channels,
                    self.image_size,
                    self.image_size
                )

        return x, y

    # =====================================================
    # APPLY POISONING
    # =====================================================
    def apply_poisoning(
        self,
        train_data,
        noise_type,
        poisoned_sample_ratio,
        attack_noise_std
    ):

        X = np.array([
            x.detach().cpu().numpy()
            for x, _ in train_data
        ])

        Y = np.array([
            y.detach().cpu().item()
            for _, y in train_data
        ])

        # =====================================
        # LABEL FLIPPING
        # =====================================
        if noise_type == "label_flipping":

            possible_labels = np.arange(
                self.num_classes
            )

            flip_indices = np.random.choice(
                len(Y),
                size=int(
                    len(Y)
                    * poisoned_sample_ratio
                ),
                replace=False
            )

            for idx in flip_indices:

                Y[idx] = np.random.choice(
                    possible_labels[
                        possible_labels != Y[idx]
                    ]
                )

        # =====================================
        # GAUSSIAN NOISE
        # =====================================
        elif noise_type == "gaussian_noise":

            for idx in range(
                int(
                    len(X)
                    * poisoned_sample_ratio
                )
            ):

                X[idx] += (
                    attack_noise_std
                    * np.random.randn(
                        *X[idx].shape
                    )
                )

        train_data = [

            (
                torch.tensor(
                    x,
                    dtype=torch.float32
                ),

                torch.tensor(
                    y,
                    dtype=torch.long
                )
            )

            for x, y in zip(X, Y)
        ]

        return train_data

    # =====================================================
    # SPLIT TRAIN / VAL
    # =====================================================
    def _split_train_val(
        self,
        data_list,
        val_ratio=0.1,
        seed=42
    ):

        n_total = len(data_list)

        if n_total <= 1:
            return data_list, data_list

        n_val = max(
            1,
            int(n_total * val_ratio)
        )

        n_train = n_total - n_val

        train_set, val_set = random_split(
            data_list,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        return list(train_set), list(val_set)

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self):

        self.model.train()

        best_state = None

        no_improve = 0

        best_score = (
            -float("inf")
            if self.early_stop_metric == "f1"
            else float("inf")
        )

        for epoch in range(self.local_steps):

            for x, y in self.train_loader:

                x, y = self._process_batch(x, y)

                output = self.model(x)

                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss(output, y)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            # =================================
            # VALIDATION
            # =================================
            val_stats = self.evaluate_on_val()

            val_loss = val_stats["loss"]

            val_f1 = val_stats["f1"]

            improved = False

            if self.early_stop_metric == "f1":

                if val_f1 > (
                    best_score + self.min_delta
                ):

                    improved = True
                    best_score = val_f1

            else:

                if val_loss < (
                    best_score - self.min_delta
                ):

                    improved = True
                    best_score = val_loss

            # =================================
            # SAVE BEST MODEL
            # =================================
            if improved:

                best_state = {

                    k: v.detach().cpu().clone()

                    for k, v in self.model.state_dict().items()
                }

                no_improve = 0

            else:

                no_improve += 1

                if no_improve >= self.patience:
                    break

            torch.cuda.empty_cache()

        self.model.to("cpu")

        torch.cuda.empty_cache()

        if best_state is not None:

            self.model.load_state_dict(
                best_state
            )

            self.model.to(self.device)

    # =====================================================
    # VALIDATION
    # =====================================================
    def evaluate_on_val(self):

        self.model.eval()

        losses = []

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for x, y in self.val_loader:

                x, y = self._process_batch(x, y)

                output = self.model(x)

                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss(output, y)

                losses.append(loss.item())

                preds = torch.argmax(
                    output,
                    dim=1
                )

                all_preds.append(preds.cpu())

                all_labels.append(y.cpu())

                del x, y, output, preds, loss

            torch.cuda.empty_cache()

        y_pred = torch.cat(
            all_preds
        ).numpy()

        y_true = torch.cat(
            all_labels
        ).numpy()

        f1 = metrics.f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        return {

            "loss": float(np.mean(losses)),

            "f1": f1
        }

    # =====================================================
    # LOAD TEST DATA
    # =====================================================
    def load_test_data(
        self,
        batch_size=1
    ):

        test_data = read_client_data(
            self.dataset,
            self.id,
            self.num_clients,
            data_type='test',
            type=self.data_type
        )

        return DataLoader(
            test_data,
            batch_size,
            drop_last=False,
            shuffle=False
        )

    # =====================================================
    # TEST METRICS
    # =====================================================
    def test_metrics(self, model=None):

        if model is None:
            model = self.model

        model.eval()

        y_true = []
        y_predict = []

        with torch.no_grad():

            for x, y in self.test_loader:

                x, y = self._process_batch(x, y)

                output = model(x)

                if isinstance(output, tuple):
                    output = output[0]

                y_predict.append(

                    torch.argmax(
                        output,
                        dim=1
                    ).detach().cpu().numpy()
                )

                y_true.append(
                    y.detach().cpu().numpy()
                )

        y_predict = np.concatenate(
            y_predict,
            axis=0
        )

        y_true = np.concatenate(
            y_true,
            axis=0
        )

        acc = metrics.accuracy_score(
            y_true,
            y_predict
        )

        precision = metrics.precision_score(
            y_true,
            y_predict,
            average='macro',
            zero_division=0
        )

        recall = metrics.recall_score(
            y_true,
            y_predict,
            average='macro',
            zero_division=0
        )

        f1 = metrics.f1_score(
            y_true,
            y_predict,
            average='macro',
            zero_division=0
        )

        test_num = len(y_predict)

        return (
            acc,
            precision,
            recall,
            f1,
            test_num
        )

    # =====================================================
    # TRAIN METRICS
    # =====================================================
    def train_metrics(self, model=None):

        if model is None:
            model = self.model

        model.eval()

        list_loss = []

        with torch.no_grad():

            for x, y in self.train_loader:

                x, y = self._process_batch(x, y)

                output = model(x)

                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss(
                    output,
                    y.long()
                )

                list_loss.append(
                    loss.item()
                )

        return list_loss

    # =====================================================
    # SET PARAMETERS
    # =====================================================
    def set_parameters(self, model):

        for new_param, old_param in zip(
            model.parameters(),
            self.model.parameters()
        ):

            old_param.data = (
                new_param.data.clone()
            )