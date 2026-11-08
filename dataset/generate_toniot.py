"""Sinh dữ liệu federated cho TON-IoT Network (10 lớp).

    python generate_toniot.py pat          # noniid20      (mỗi client đúng 2 lớp)
    python generate_toniot.py exdir        # noniidexdir20 (C=2, alpha=0.1)
    python generate_toniot.py dir          # noniid20dir   (Dirichlet alpha=0.5)
    python generate_toniot.py centralized
Tuỳ chọn: --num_clients 20 --class_per_client 2 --alpha 0.1 --seed 1 --force
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.fed_generator import run_cli

DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toniot") + os.sep


def load_toniot():
    """Đọc và tiền xử lý TON-IoT: encode categorical, MinMax, nhãn đa lớp (cột `type`).

    LabelEncoder xếp theo alphabet: backdoor=0, ddos=1, dos=2, injection=3, mitm=4,
    normal=5, password=6, ransomware=7, scanning=8, xss=9 (NORMAL_LABEL['toniot'] = 5).
    """
    csv_path = os.path.join(DIR_PATH, "rawdata", "train_test_network.csv")
    print("Loading TON-IoT dataset ...")
    df = pd.read_csv(csv_path)
    print("Original shape :", df.shape)

    df = df.replace("-", "Unknown").fillna("Unknown")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["type"])
    print("\nClass Mapping")
    for i, c in enumerate(label_encoder.classes_):
        print(f"{i:2d} -> {c}")

    X = df.drop(columns=["label", "type"])
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.astype(np.float32)
    X = np.asarray(MinMaxScaler().fit_transform(X), dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print("Samples :", len(y), "| Features :", X.shape[1],
          "| Classes :", len(label_encoder.classes_))
    for idx, cls in enumerate(label_encoder.classes_):
        print(f"{cls:20s}: {(y == idx).sum()}")
    return X, y


if __name__ == "__main__":
    run_cli("generate_toniot.py", DIR_PATH, load_toniot)
