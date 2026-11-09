import os
import sys
import random
import numpy as np
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.preprocessing import MinMaxScaler
from functools import partial

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "bot_iot/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if niid:
        data_type = 'noniid'
    else:
        data_type = 'iid'
    # Setup directory for train/test data
    config_path = dir_path + data_type + "/config.json"
    train_path = dir_path + data_type + "/train/"
    test_path = dir_path + data_type + "/test/"

    if check(config_path, train_path, test_path, num_clients, balance):
        return

    # load bot_iot data
    data = pd.DataFrame()
    for i in range(1, 5):
        sub_data = pd.read_csv(f'{dir_path}/rawdata/UNSW_2018_IoT_Botnet_Full5pc_{i}.csv', converters={"dport": partial(int, base=16), "sport": partial(int, base=16)})
        # concatenate
        data = pd.concat([data, sub_data], ignore_index=True)

    data.reset_index(drop=True, inplace=True)
    mapping = {
        'DDoS': 0,
        'DoS': 1,
        'Normal': 2,
        'Reconnaissance': 3,
        'Theft': 4
    }
    y = data["category"].map(mapping)
    # y = data["attack"]

    droppedColumn = [
        "pkSeqID",
        "stime",
        "flgs",
        "proto",
        "saddr",
        "daddr",
        "state",
        "ltime",
        "seq",
        "attack",
        "category",
        "subcategory",
    ]

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


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
