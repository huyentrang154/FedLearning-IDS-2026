import numpy as np
import os
import sys
import random
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.preprocessing import MinMaxScaler

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "ctu13_08/"


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

    # Get CTU13_08 data
    data = np.genfromtxt(f'{dir_path}/rawdata/CTU13_08.csv', delimiter=",")
    print('Data size:', data.shape)

    # split x, y
    X = data[:, :-1]
    y = data[:, -1]
    X = np.array(X, dtype=np.float64)

    # normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # reassign labels
    y[y == 1] = 0  # normal
    y[y == 2] = 1  # abnormal

    # y = y.to_numpy()
    num_classes = len(set(y))
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
