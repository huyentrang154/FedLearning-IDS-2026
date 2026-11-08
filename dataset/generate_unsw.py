"""Sinh dữ liệu federated cho UNSW-NB15.

    python generate_unsw.py pat          # noniid20      (mỗi client đúng 2 lớp)
    python generate_unsw.py exdir        # noniidexdir20 (C=2, alpha=0.1)
    python generate_unsw.py dir          # noniid20dir   (Dirichlet alpha=0.5)
    python generate_unsw.py centralized
Tuỳ chọn: --num_clients 20 --class_per_client 2 --alpha 0.1 --seed 1 --force
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.fed_generator import run_cli

DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsw") + os.sep


def load_unsw():
    train_data = pd.read_csv(f'{DIR_PATH}rawdata/UNSW_Train.csv', sep=",", header=None)
    test_data = pd.read_csv(f'{DIR_PATH}rawdata/UNSW_Test.csv', sep=",", header=None)

    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)
    data = data.to_numpy()

    X = np.array(data[:, :-1], dtype=np.float64)
    y = data[:, -1]
    X = MinMaxScaler().fit_transform(X)
    return X, y


if __name__ == "__main__":
    run_cli("generate_unsw.py", DIR_PATH, load_unsw)
