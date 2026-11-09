import os
import sys
import random
import numpy as np
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "unsw_nb15/"
features = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "Label",
]

cat_cols = [
    "proto",
    "state",
    "service",
    "attack_cat",
]


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

    # load UNSW-big data
    train_data = pd.read_csv(f'{dir_path}/rawdata/UNSW_NB15_training-set.csv', sep=",", header=None)
    test_data = pd.read_csv(f'{dir_path}/rawdata/UNSW_NB15_testing-set.csv', sep=',', header=None)

    # concatenate
    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)
    data = data.to_numpy()

    num_classes = len(set(y))
    print("Original Size: ", (X.shape, y.shape))
    print(f'Number of classes: {num_classes}')

    X = X.to_numpy()
    y = y.to_numpy()

    X, y, statistic = separate_data(X, y, num_clients, num_classes, balance)
    train_data, test_data, mal_data = split_data(X, y)
    save_file(config_path, train_path, test_path, mal_path, train_data, test_data, mal_data, num_clients, num_classes,
              statistic, balance)


if __name__ == "__main__":
    balance = True if sys.argv[1] == "balance" else False
    generate_dataset(dir_path, num_clients, balance)
