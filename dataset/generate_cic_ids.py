import os
import sys
import random
import numpy as np
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ujson

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "cic_ids/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if niid:
        data_type = 'noniid'
    else:
        data_type = 'iid'
    # Setup directory for train/test data
    config_path = dir_path + data_type + str(num_clients) +partition+ "01/config.json"
    train_path = dir_path + data_type + str(num_clients) +partition+ "01/train/"
    test_path = dir_path + data_type + str(num_clients) +partition+ "01/test/"

    if check(config_path, train_path, test_path, num_clients, balance):
        return

    # load CIC_IDS data
    train_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Train-10percent.csv', sep=",", header=None)
    test_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Test-10percent.csv', sep=',', header=None)

    # drop nan value
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)

    # concatenate
    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)
    # y = data[85]
    # y.loc[y != "BENIGN"] = 1  # abnormal
    # y.loc[y == "BENIGN"] = 0  # normal
    # y = data[85].map(lambda x: 0 if x == "BENIGN" else 1)
    mapping = {
        'DoS GoldenEye': 0,
        'PortScan': 1,
        'SSH-Patator': 2,
        'DoS slowloris': 3,
        'Bot': 4,
        'Heartbleed': 5,
        'BENIGN': 6,
        'DDoS': 7,
        'DoS Slowhttptest': 8,
        'Infiltration': 9,
        'DoS Hulk': 10,
        'FTP-Patator': 11
    }
    y = data[85].map(mapping)
    droppedColumn = [0, 1, 2, 4, 7, 85]
    X = data
    for idx in droppedColumn:
        X = X.drop([idx], axis=1)

    X = X.to_numpy()

    X = np.array(X, dtype=np.float64)

    # normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = y.to_numpy()
    num_classes = len(set(y))
    print("Original Size: ", (X.shape, y.shape))
    print(f'Number of classes: {num_classes}')
    X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data, = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, balance)

def generate_centralized_dataset(dir_path):
    """
    Sinh dataset cho Centralized Learning.
    Không chia client.
    """

    train_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Train-10percent.csv', sep=",", header=None)
    test_data = pd.read_csv(f'{dir_path}/rawdata/CIC-IDS-2017-Test-10percent.csv', sep=',', header=None)

    # drop nan value
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)

    # concatenate
    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)
    # y = data[85]
    # y.loc[y != "BENIGN"] = 1  # abnormal
    # y.loc[y == "BENIGN"] = 0  # normal
    # y = data[85].map(lambda x: 0 if x == "BENIGN" else 1)
    mapping = {
        'DoS GoldenEye': 0,
        'PortScan': 1,
        'SSH-Patator': 2,
        'DoS slowloris': 3,
        'Bot': 4,
        'Heartbleed': 5,
        'BENIGN': 6,
        'DDoS': 7,
        'DoS Slowhttptest': 8,
        'Infiltration': 9,
        'DoS Hulk': 10,
        'FTP-Patator': 11
    }
    y = data[85].map(mapping)
    droppedColumn = [0, 1, 2, 4, 7, 85]
    X = data
    for idx in droppedColumn:
        X = X.drop([idx], axis=1)

    X = X.to_numpy()

    X = np.array(X, dtype=np.float64)

    # normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = y.to_numpy()

    print("====================================")
    print("Centralized Dataset")
    print("Samples :", len(y))
    print("Features:", X.shape[1])
    print("Classes :", len(np.unique(y)))
    print("====================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    save_dir = os.path.join(dir_path, "centralized")
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, "train.npz"),
        data={
            "x": X_train,
            "y": y_train
        }
    )

    np.savez_compressed(
        os.path.join(save_dir, "test.npz"),
        data={
            "x": X_test,
            "y": y_test
        }
    )

    config = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "num_classes": len(np.unique(y)),
        "input_dim": X.shape[1]
    }

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        ujson.dump(config, f)

    print("Centralized dataset saved.")
    print("Train:", len(y_train))
    print("Test :", len(y_test))

if __name__ == "__main__":
    
    generate_centralized_dataset(
        dir_path=dir_path
    )

    print("=" * 60)
    print("Dataset generation completed!")
    print("=" * 60)
    
# if __name__ == "__main__":
#     niid = True if sys.argv[1] == "noniid" else False
#     balance = True if sys.argv[2] == "balance" else False
#     partition = sys.argv[3] if sys.argv[3] != "-" else None

#     generate_dataset(dir_path, num_clients, niid, balance, partition)

#
# cd ./dataset
# # python generate_MNIST.py iid - - # for iid and unbalanced scenario
# # python generate_MNIST.py iid balance - # for iid and balanced scenario
# # python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
# python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario
