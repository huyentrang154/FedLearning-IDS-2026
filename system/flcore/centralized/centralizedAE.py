import copy
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

from sklearn import metrics
from torch.utils.data import DataLoader, random_split, ConcatDataset

from utils.data_utils import read_centralized_data


class CentralizedAE(object):

    def __init__(self, args, times):

        # ==========================================================
        # Basic Information
        # ==========================================================
        self.device = args.device
        self.dataset = args.dataset
        self.data_type = args.data_type
        self.global_rounds = getattr(args, "global_rounds", getattr(args, "epochs", 100))

        self.algo = getattr(args, "algorithm", "CentralizedAE")
        self.exp_name = getattr(args, "exp_name", "default")
        self.seed = getattr(args, "seed", 0)
        self.times = times

        # ==========================================================
        # Model
        # ==========================================================
        self.model = copy.deepcopy(args.model).to(self.device)

        # ==========================================================
        # Data / Training parameters
        # ==========================================================
        self.batch_size = args.batch_size
        self.local_steps = args.local_steps
        self.learning_rate = args.local_learning_rate

        self.input_channels = getattr(args, "input_channels", 1)
        self.image_size = getattr(args, "image_size", 0)
        self.num_classes = getattr(args, "num_classes", 2)

        # ==========================================================
        # Loss & Optimizer
        # ==========================================================
        self.cls_loss = nn.CrossEntropyLoss()
        self.rec_loss = nn.MSELoss()

        self.lambda_rec = getattr(
            args,
            "lambda_rec",
            1.0
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # ==========================================================
        # READ TRAIN DATA (Centralized)
        # ==========================================================
        full_train_data = read_centralized_data(
            self.dataset,
            data_type="train"
        )

        # ==========================================================
        # Split TRAIN / VALIDATION (90% train, 10% validation)
        # ==========================================================
        self.train_data, self.val_data = self._split_train_val(
            full_train_data,
            val_ratio=0.1,
            seed=self.seed
        )

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        # ==========================================================
        # READ TEST DATA (Centralized Test Set - Unified)
        # ==========================================================
        test_data = read_centralized_data(
            self.dataset,
            data_type="test"
        )
        if isinstance(test_data, list):
            test_data = ConcatDataset(test_data)

        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        # ==========================================================
        # Early stopping
        # ==========================================================
        self.patience = getattr(args, "patience", 10)
        self.min_delta = getattr(args, "min_delta", 1e-4)
        self.early_stop_metric = getattr(args, "early_stop_metric", "f1")

        # ==========================================================
        # History
        # ==========================================================
        self.rs_train_loss = []
        self.rs_val_loss = []
        self.rs_val_f1 = []

        self.rs_test_acc = []
        self.rs_test_precision = []
        self.rs_test_recall = []
        self.rs_test_f1 = []

        # ==========================================================
        # Best model
        # ==========================================================
        self.best_model = None
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_precision = 0.0
        self.best_recall = 0.0
        self.best_epoch = 0

        self.budget = []

        # ==========================================================
        # Save Path
        # ==========================================================
        self.result_path = os.path.join(
            "..",
            "results",
            self.dataset,
            self.data_type,
            self.algo,
            self.exp_name,
            f"seed_{self.seed}"
        )
        os.makedirs(self.result_path, exist_ok=True)
        self.log_file = os.path.join(self.result_path, "tracking.log")

        # Initial log
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Algorithm : {self.algo}\n")
            f.write(f"Dataset   : {self.dataset}\n")
            f.write(f"Seed      : {self.seed}\n")
            f.write(f"Device    : {self.device}\n")
            f.write("=" * 70 + "\n")

        print("\nAlgorithm :", self.algo)
        print("Dataset   :", self.dataset)
        print("Train Samples :", len(self.train_data))
        print("Valid Samples :", len(self.val_data))
        print("Test Samples  :", len(test_data))

    # ==============================================================
    # Split Train / Validation
    # ==============================================================
    def _split_train_val(
        self,
        dataset,
        val_ratio=0.1,
        seed=0
    ):
        total = len(dataset)
        val_size = max(1, int(total * val_ratio))
        train_size = total - val_size

        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return train_set, val_set

    # ==============================================================
    # Process batch
    # ==============================================================
    def _process_batch(self, x, y):
        if isinstance(x, list):
            x = x[0]

        x = x.to(self.device, non_blocking=True)
        y = y.long().to(self.device, non_blocking=True)

        if (
            self.input_channels > 1
            and len(x.shape) == 2
            and self.image_size > 0
        ):
            x = x.view(
                -1,
                self.input_channels,
                self.image_size,
                self.image_size
            )
        return x, y

    # ==============================================================
    # Forward model
    # ==============================================================
    def _forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0], output[1]
        return output, None

    # ==============================================================
    # Validation
    # ==============================================================
    def evaluate_on_val(self):
        self.model.eval()
        losses = []
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = self._process_batch(x, y)
                output, reconstruction = self._forward(x)

                loss_cls = self.cls_loss(output, y)
                loss_rec = self.rec_loss(reconstruction, x) if reconstruction is not None else 0.0
                loss = loss_cls + self.lambda_rec * loss_rec

                losses.append(loss.item())
                pred = torch.argmax(output, dim=1)

                y_true.extend(y.detach().cpu().numpy())
                y_pred.extend(pred.detach().cpu().numpy())

        val_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        val_f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "loss": val_loss,
            "f1": val_f1
        }

    # ==============================================================
    # Test Evaluation (Centralized Standard)
    # ==============================================================
    def evaluate_on_test(self):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = self._process_batch(x, y)
                output, _ = self._forward(x)
                pred = torch.argmax(output, dim=1)

                y_true.extend(y.detach().cpu().numpy())
                y_pred.extend(pred.detach().cpu().numpy())

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0

        acc = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)

        return acc, precision, recall, f1

    # ==============================================================
    # Train
    # ==============================================================
    def train(self):
        print("\n" + "=" * 70)
        print("Start Centralized Training")
        print("=" * 70)

        for round_idx in range(self.global_rounds + 1):
            start = time.time()
            print(f"\n------------- Round {round_idx} -------------")
            with open(self.log_file, "a") as f:
                f.write(f"\n------------- Round {round_idx} -------------\n")

            self.model.train()
            best_state = None
            best_score = -float("inf")
            no_improve = 0

            for epoch in range(self.local_steps):
                running_loss = []
                self.model.train()

                for x, y in self.train_loader:
                    x, y = self._process_batch(x, y)
                    output, reconstruction = self._forward(x)

                    loss_cls = self.cls_loss(output, y)
                    loss_rec = self.rec_loss(reconstruction, x) if reconstruction is not None else 0.0
                    loss = loss_cls + self.lambda_rec * loss_rec

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss.append(loss.item())

                epoch_train_loss = float(np.mean(running_loss)) if len(running_loss) > 0 else 0.0

                val_result = self.evaluate_on_val()
                val_loss = val_result["loss"]
                val_f1 = val_result["f1"]

                self.rs_val_loss.append(val_loss)
                self.rs_val_f1.append(val_f1)

                score = val_f1 if self.early_stop_metric == "f1" else -val_loss

                if score > (best_score + self.min_delta):
                    best_score = score
                    no_improve = 0
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    no_improve += 1

                print(
                    f"Epoch {epoch + 1}/{self.local_steps} | "
                    f"Train Loss={epoch_train_loss:.4f} | "
                    f"Val Loss={val_loss:.4f} | "
                    f"Val F1={val_f1:.4f}"
                )

                with open(self.log_file, "a") as f:
                    f.write(
                        f"Epoch {epoch + 1}/{self.local_steps} | "
                        f"Train Loss={epoch_train_loss:.4f} | "
                        f"Val Loss={val_loss:.4f} | "
                        f"Val F1={val_f1:.4f}\n"
                    )

                if no_improve >= self.patience:
                    print("Early stopping.")
                    with open(self.log_file, "a") as f:
                        f.write("Early stopping.\n")
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)
                self.model.to(self.device)

            # Đánh giá trên tập test centralized chuẩn
            test_acc, test_precision, test_recall, test_f1 = self.evaluate_on_test()

            self.rs_test_acc.append(test_acc)
            self.rs_test_precision.append(test_precision)
            self.rs_test_recall.append(test_recall)
            self.rs_test_f1.append(test_f1)
            self.rs_train_loss.append(epoch_train_loss)

            elapsed = time.time() - start
            self.budget.append(elapsed)

            print(f"\nRound {round_idx} Time: {elapsed:.2f}s")
            print(
                f"Test Results -> Acc: {test_acc:.4f} | "
                f"Precision: {test_precision:.4f} | "
                f"Recall: {test_recall:.4f} | "
                f"F1: {test_f1:.4f}"
            )

            if test_f1 > self.best_f1:
                self.best_f1 = test_f1
                self.best_acc = test_acc
                self.best_precision = test_precision
                self.best_recall = test_recall
                self.best_epoch = round_idx
                self.best_model = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
            self.model.to(self.device)

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Algorithm : {self.algo}")
        print(f"Dataset   : {self.dataset}")
        print(f"Best Round: {self.best_epoch}")
        print(f"Accuracy  : {self.best_acc:.4f}")
        print(f"Precision : {self.best_precision:.4f}")
        print(f"Recall    : {self.best_recall:.4f}")
        print(f"F1-score  : {self.best_f1:.4f}")

        avg_time = (
            sum(self.budget[1:]) / len(self.budget[1:])
            if len(self.budget) > 1
            else (self.budget[0] if len(self.budget) == 1 else 0.0)
        )
        print(f"Avg Time/Round : {avg_time:.4f} sec")

        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 70 + "\nFINAL RESULT\n" + "=" * 70 + "\n")
            f.write(f"Algorithm : {self.algo}\n")
            f.write(f"Dataset   : {self.dataset}\n")
            f.write(f"Best Round: {self.best_epoch}\n")
            f.write(f"Accuracy  : {self.best_acc:.4f}\n")
            f.write(f"Precision : {self.best_precision:.4f}\n")
            f.write(f"Recall    : {self.best_recall:.4f}\n")
            f.write(f"F1-score  : {self.best_f1:.4f}\n")
            f.write(f"Avg Time/Round : {avg_time:.4f} sec\n")

        self.save_results()
        self.save_model()

    # ==============================================================
    # Train metrics
    # ==============================================================
    def train_metrics(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = self._process_batch(x, y)
                output, _ = self._forward(x)
                loss = self.cls_loss(output, y)
                losses.append(loss.item())
        return losses

    # ==============================================================
    # Evaluate
    # ==============================================================
    def evaluate(self):
        train_losses = self.train_metrics()
        train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0

        acc, precision, recall, f1 = self.evaluate_on_test()

        self.rs_train_loss.append(train_loss)
        self.rs_test_acc.append(acc)
        self.rs_test_precision.append(precision)
        self.rs_test_recall.append(recall)
        self.rs_test_f1.append(f1)

        print("\nOverall Performance Metrics")
        print(f"\tTrain Loss : {train_loss:.4f}")
        print(f"\tAccuracy   : {acc:.4f}")
        print(f"\tPrecision  : {precision:.4f}")
        print(f"\tRecall     : {recall:.4f}")
        print(f"\tF1-score   : {f1:.4f}")

        with open(self.log_file, "a") as f:
            f.write("\nOverall Performance Metrics\n")
            f.write(f"\tTrain Loss : {train_loss:.4f}\n")
            f.write(f"\tAccuracy   : {acc:.4f}\n")
            f.write(f"\tPrecision  : {precision:.4f}\n")
            f.write(f"\tRecall     : {recall:.4f}\n")
            f.write(f"\tF1-score   : {f1:.4f}\n")

    # ==============================================================
    # Save results
    # ==============================================================
    def save_results(self):
        metrics_results = {
            "train_loss": self.rs_train_loss,
            "val_loss": self.rs_val_loss,
            "val_f1": self.rs_val_f1,
            "test_acc": self.rs_test_acc,
            "test_precision": self.rs_test_precision,
            "test_recall": self.rs_test_recall,
            "test_f1": self.rs_test_f1,
            "best_acc": self.best_acc,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "budget": self.budget
        }

        save_path = os.path.join(self.result_path, "metrics_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(metrics_results, f)

        print(f"\nMetrics saved to: {save_path}")

    # ==============================================================
    # Save model
    # ==============================================================
    def save_model(self, filename="best_model.pt"):
        save_path = os.path.join(self.result_path, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        with open(self.log_file, "a") as f:
            f.write(f"\nModel saved to {save_path}\n")

    # ==============================================================
    # Load model
    # ==============================================================
    def load_model(self, filename="best_model.pt"):
        load_path = os.path.join(self.result_path, filename)
        self.model.load_state_dict(
            torch.load(load_path, map_location=self.device)
        )
        self.model.to(self.device)
        print(f"Loaded model from {load_path}")
