import os
import sys
import random
import numpy as np
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.preprocessing import MinMaxScaler

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "unsw_big/"
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

    # load UNSW-big data
    data = pd.DataFrame()
    for i in range(1, 5):
        sub_data = pd.read_csv(f'{dir_path}/rawdata/UNSW-NB15_{i}.csv', names=features, sep=",", header=None)
        # concatenate
        data = pd.concat([data, sub_data], ignore_index=True)

    data.reset_index(drop=True, inplace=True)
    print('Before:', data.shape)
    data = data.drop(["srcip", "dstip"], axis=1)
    data.dropna()
    print('After:', data.shape)

    # y = data["Label"].map(lambda x: 1 if x > 0 else 0)
    y = data['Label']
    # import pdb
    # pdb.set_trace()
    X = data.drop("Label", axis=1)
    X = pd.get_dummies(X, columns=cat_cols, dtype=int)
    print(X.sample())

    X = np.array(X, dtype=np.float64)

    # normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # X.apply(lambda x: x if (isinstance(x, int) or isinstance(x, float)) else int(str(x), 16))
    num_classes = len(set(y))
    print("Original Size: ", (X.shape, y.shape))
    print(f'Number of classes: {num_classes}')

    # X = X.to_numpy()
    y = y.to_numpy()

    X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data, = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, balance)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
