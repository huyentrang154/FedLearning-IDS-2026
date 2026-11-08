"""Sinh dữ liệu federated cho CIC-IDS-2017 (12 lớp).

    python generate_cic_ids.py pat          # noniid20      (mỗi client đúng 2 lớp)
    python generate_cic_ids.py exdir        # noniidexdir20 (C=2, alpha=0.1)
    python generate_cic_ids.py dir          # noniid20dir   (Dirichlet alpha=0.5)
    python generate_cic_ids.py centralized
Tuỳ chọn: --num_clients 20 --class_per_client 2 --alpha 0.1 --seed 1 --force
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.fed_generator import run_cli

DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cic_ids") + os.sep

LABEL_MAPPING = {
    'DoS GoldenEye': 0,
    'PortScan': 1,
    'SSH-Patator': 2,
    'DoS slowloris': 3,
    'Bot': 4,
    'Heartbleed': 5,
    'BENIGN': 6,          # nhãn normal (khớp NORMAL_LABEL['cic_ids'] = 6)
    'DDoS': 7,
    'DoS Slowhttptest': 8,
    'Infiltration': 9,
    'DoS Hulk': 10,
    'FTP-Patator': 11,
}
DROPPED_COLUMNS = [0, 1, 2, 4, 7, 85]


def load_cic_ids():
    train_data = pd.read_csv(f'{DIR_PATH}rawdata/CIC-IDS-2017-Train-10percent.csv', sep=",", header=None)
    test_data = pd.read_csv(f'{DIR_PATH}rawdata/CIC-IDS-2017-Test-10percent.csv', sep=",", header=None)

    # drop nan / inf
    for df in (train_data, test_data):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)

    y = data[85].map(LABEL_MAPPING)
    X = data.drop(columns=DROPPED_COLUMNS).to_numpy()
    X = np.array(X, dtype=np.float64)
    X = MinMaxScaler().fit_transform(X)
    return X, y.to_numpy()


if __name__ == "__main__":
    run_cli("generate_cic_ids.py", DIR_PATH, load_cic_ids)
