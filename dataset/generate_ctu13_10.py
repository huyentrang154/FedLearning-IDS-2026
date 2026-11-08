import numpy as np
import os
import sys
import random
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "ctu13_10/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, balance):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    mal_path = dir_path + 'mal/'

    if check(config_path, train_path, test_path, mal_path, num_clients, balance):
        return

    # Get CTU13_08 data
    data = np.genfromtxt(f'{dir_path}/rawdata/CTU13_10.csv', delimiter=",")
    print('Data size:', data.shape)

    # split x, y
    X = data[:, :-1]
    y = data[:, -1]

    # reassign labels
    y[y <= 1] = 0  # normal
    y[y > 1] = 1  # abnormal

    num_classes = len(set(y))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data(X, y, num_clients, num_classes, balance)
    train_data, test_data, mal_data = split_data(X, y)
    save_file(config_path, train_path, test_path, mal_path, train_data, test_data, mal_data, num_clients, num_classes,
              statistic, balance)


if __name__ == "__main__":
    balance = True if sys.argv[1] == "balance" else False
    generate_dataset(dir_path, num_clients, balance)
