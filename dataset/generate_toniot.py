import os
import sys
import random
import numpy as np
import pandas as pd
import ujson

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.dataset_utils import (
    check,
    separate_data,
    split_data,
    save_file
)

#########################################################
# Configuration
#########################################################

random.seed(1)
np.random.seed(1)

num_clients = 20

dir_path = "toniot/"

#########################################################
# Preprocess TON-IoT
#########################################################

def preprocess_toniot(dir_path):
    """
    Đọc và tiền xử lý TON-IoT Network Dataset.

    Output
    ------
    X : np.ndarray
        Feature matrix đã encode + normalize

    y : np.ndarray
        Label (0/1)
    """

    csv_path = os.path.join(
        dir_path,
        "rawdata",
        "train_test_network.csv"
    )

    print("=" * 60)
    print("Loading TON-IoT dataset ...")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print("Original shape :", df.shape)

    ####################################################
    # Replace missing value '-'
    ####################################################

    df = df.replace("-", "Unknown")
    df = df.fillna("Unknown")

    ####################################################
    # Multi-class label
    ####################################################

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["type"])

    print("\nClass Mapping")
    print("-" * 40)

    for i, c in enumerate(label_encoder.classes_):
        print(f"{i:2d} -> {c}")

    ####################################################
    # Features
    ####################################################

    X = df.drop(columns=["label", "type"])

    ####################################################
    # Encode categorical columns
    ####################################################

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns

    print("Categorical columns:")
    print(list(categorical_cols))

    for col in categorical_cols:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

    ####################################################
    # Convert numeric
    ####################################################

    X = X.astype(np.float32)

    ####################################################
    # Normalize
    ####################################################

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    ####################################################
    # Dataset information
    ####################################################

    print("=" * 60)
    print("TON-IoT preprocessing completed")
    print("=" * 60)
    print("Samples      :", len(y))
    print("Features     :", X.shape[1])
    print("Number of classes :", len(label_encoder.classes_))

    print("\nClass distribution")

    for idx, cls in enumerate(label_encoder.classes_):
        print(f"{cls:20s}: {(y == idx).sum()}")
    print("=" * 60)

    return X, y
#########################################################
# Generate Federated Dataset
#########################################################

def generate_dataset(
        dir_path,
        num_clients,
        niid,
        balance,
        partition
):
    """
    Sinh dataset cho Federated Learning.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    ####################################################
    # Dataset type
    ####################################################

    if niid:
        data_type = "noniid"
    else:
        data_type = "iid"

    save_dir = (
        dir_path
        + data_type
        + str(num_clients)
        + partition
    )

    config_path = os.path.join(
        save_dir,
        "config.json"
    )

    train_path = os.path.join(
        save_dir,
        "train/"
    )

    test_path = os.path.join(
        save_dir,
        "test/"
    )

    ####################################################
    # Check existed dataset
    ####################################################

    if check(
            config_path,
            train_path,
            test_path,
            num_clients,
            balance
    ):
        return

    ####################################################
    # Preprocess TON-IoT
    ####################################################

    X, y = preprocess_toniot(dir_path)

    num_classes = len(np.unique(y))

    print("=" * 60)
    print("Generate Federated Dataset")
    print("=" * 60)
    print("Samples      :", len(y))
    print("Features     :", X.shape[1])
    print("Classes      :", num_classes)
    print("Clients      :", num_clients)
    print("Partition    :", partition)
    print("Non-IID      :", niid)
    print("=" * 60)

    ####################################################
    # Partition dataset
    ####################################################

    X,y,statistic = separate_data(
        (X, y),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=2
    )

    ####################################################
    # Split train/test
    ####################################################

    train_data, test_data = split_data(
        X,
        y
    )

    ####################################################
    # Save dataset
    ####################################################

    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition
    )

    print("=" * 60)
    print("TON-IoT Federated dataset generated successfully.")
    print("=" * 60)

#########################################################
# Generate Centralized Dataset
#########################################################

def generate_centralized_dataset(dir_path):
    """
    Sinh dataset cho Centralized Learning.

    Output:
        toniot/
            centralized/
                train.npz
                test.npz
                config.json
    """

    ####################################################
    # Preprocess dataset
    ####################################################

    X, y = preprocess_toniot(dir_path)

    num_classes = len(np.unique(y))

    print("=" * 60)
    print("Generate Centralized Dataset")
    print("=" * 60)
    print("Samples      :", len(y))
    print("Features     :", X.shape[1])
    print("Classes      :", num_classes)
    print("=" * 60)

    ####################################################
    # Train/Test Split
    ####################################################

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    ####################################################
    # Save directory
    ####################################################

    save_dir = os.path.join(
        dir_path,
        "centralized"
    )

    os.makedirs(save_dir, exist_ok=True)

    ####################################################
    # Save train
    ####################################################

    np.savez_compressed(
        os.path.join(save_dir, "train.npz"),
        data={
            "x": X_train,
            "y": y_train
        }
    )

    ####################################################
    # Save test
    ####################################################

    np.savez_compressed(
        os.path.join(save_dir, "test.npz"),
        data={
            "x": X_test,
            "y": y_test
        }
    )

    ####################################################
    # Save config
    ####################################################

    config = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "num_classes": num_classes,
        "input_dim": X.shape[1],
        "dataset": "TON-IoT",
        "task": "Network Intrusion Detection",
        "train_ratio": 0.8,
        "test_ratio": 0.2
    }

    with open(
        os.path.join(save_dir, "config.json"),
        "w"
    ) as f:
        ujson.dump(config, f, indent=4)

    ####################################################
    # Dataset Information
    ####################################################

    print("=" * 60)
    print("Centralized dataset generated successfully.")
    print("=" * 60)
    print("Train samples :", len(y_train))
    print("Test samples  :", len(y_test))
    print("Features      :", X.shape[1])
    print("Classes       :", num_classes)
    print("Save path     :", save_dir)
    print("=" * 60)
#########################################################
# Main
#########################################################

def print_usage():

    print("=" * 60)
    print("TON-IoT Dataset Generator")
    print("=" * 60)
    print("Federated Learning:")
    print("python generate_toniot.py noniid balance pat")
    print("python generate_toniot.py noniid balance dir")
    print("python generate_toniot.py iid balance -")
    print()
    print("Centralized Learning:")
    print("python generate_toniot.py centralized")
    print("=" * 60)


if __name__ == "__main__":

    ####################################################
    # Centralized dataset
    ####################################################

    if len(sys.argv) == 2:

        if sys.argv[1].lower() == "centralized":

            generate_centralized_dataset(dir_path)

            print("=" * 60)
            print("Centralized dataset generation completed!")
            print("=" * 60)

            sys.exit()

    ####################################################
    # Federated dataset
    ####################################################

    if len(sys.argv) != 4:

        print_usage()
        sys.exit()

    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False

    partition = (
        sys.argv[3]
        if sys.argv[3] != "-"
        else None
    )

    generate_dataset(
        dir_path=dir_path,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        partition=partition
    )

    print("=" * 60)
    print("Federated dataset generation completed!")
    print("=" * 60)