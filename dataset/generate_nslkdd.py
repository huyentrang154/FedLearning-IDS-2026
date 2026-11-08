"""Sinh dữ liệu federated cho NSL-KDD.

Cách dùng (chạy ở bất kỳ thư mục nào):
    # noniid20      : pathological, mỗi client đúng k=2 lớp
    python generate_nslkdd.py pat
    # noniidexdir20 : extended Dirichlet (C=2 lớp/client, alpha=0.1)
    python generate_nslkdd.py exdir
    # noniid20dir   : Dirichlet thuần (alpha=0.5)
    python generate_nslkdd.py dir
    # dataset centralized
    python generate_nslkdd.py centralized

Tuỳ chọn: --num_clients 20 --class_per_client 2 --alpha 0.1 --seed 1 --force
Thư mục xuất ra: nslkdd/<tên>/{train,test,config.json}. Nếu thư mục đã tồn tại với
cấu hình khác (hoặc dùng --force) thì bản cũ được đổi tên thành <tên>_old_<timestamp>.
"""
import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import ujson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import dataset_utils
from utils.dataset_utils import check, separate_data, split_data, save_file, set_alpha

DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nslkdd") + os.sep

# Mặc định cho từng kiểu chia: (tên thư mục, partition, class_per_client, alpha)
PRESETS = {
    "pat":  dict(folder="noniid{n}",       partition="pat",   class_per_client=2,    alpha=None),
    "exdir": dict(folder="noniidexdir{n}", partition="exdir", class_per_client=2,    alpha=0.1),
    "dir":  dict(folder="noniid{n}dir",    partition="dir",   class_per_client=None, alpha=0.5),
}


def load_nslkdd():
    train_data = np.genfromtxt(f'{DIR_PATH}rawdata/NSLKDD_Train.csv', delimiter=",")
    test_data = np.genfromtxt(f'{DIR_PATH}rawdata/NSLKDD_Test.csv', delimiter=",")
    data = np.concatenate((train_data, test_data), axis=0)
    print('Data size:', data.shape)

    X = np.array(data[:, :-1], dtype=np.float64)
    y = data[:, -1]
    X = MinMaxScaler().fit_transform(X)
    return X, y


def generate_dataset(mode, num_clients=20, class_per_client=None, alpha=None, force=False):
    preset = PRESETS[mode]
    partition = preset["partition"]
    if class_per_client is None:
        class_per_client = preset["class_per_client"]
    if alpha is None:
        alpha = preset["alpha"]
    if alpha is not None:
        set_alpha(alpha)

    folder = preset["folder"].format(n=num_clients)
    out_dir = DIR_PATH + folder
    config_path = out_dir + "/config.json"
    train_path = out_dir + "/train/"
    test_path = out_dir + "/test/"

    niid, balance = True, True
    # pat không dùng alpha -> lưu alpha mặc định của module để check() khớp với config
    if not force and check(config_path, train_path, test_path, num_clients, niid,
                           balance, partition, class_per_client):
        print(f"{folder} đã được sinh với cấu hình này, bỏ qua (dùng --force để sinh lại).")
        return

    # Sao lưu thư mục cũ để không ghi đè lẫn dữ liệu cũ/mới
    if os.path.exists(out_dir):
        backup = f"{out_dir}_old_{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(out_dir, backup)
        print(f"Đã sao lưu thư mục cũ -> {backup}")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    X, y = load_nslkdd()
    num_classes = len(set(y))
    print("Original Size: ", (X.shape, y.shape))
    print(f'Number of classes: {num_classes}')
    print(f'Partition={partition} | class_per_client={class_per_client} | '
          f'alpha={dataset_utils.alpha} | num_clients={num_clients}')

    X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance,
                                    partition, class_per_client=class_per_client)

    # Kiểm tra nhanh số lớp trên mỗi client
    n_cls = [len(np.unique(yc)) for yc in y]
    print("Số lớp trên mỗi client:", n_cls)
    if partition in ("pat", "exdir") and class_per_client is not None:
        assert max(n_cls) <= class_per_client, \
            f"Có client vượt quá {class_per_client} lớp: {n_cls}"

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid, balance, partition, class_per_client)


def generate_centralized_dataset(dir_path=DIR_PATH):
    """Sinh dataset cho Centralized Learning (không chia client)."""
    X, y = load_nslkdd()
    X = X.astype(np.float32)

    print("====================================")
    print("Centralized Dataset")
    print("Samples :", len(y))
    print("Features:", X.shape[1])
    print("Classes :", len(np.unique(y)))
    print("====================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y, shuffle=True)

    save_dir = os.path.join(dir_path, "centralized")
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(os.path.join(save_dir, "train.npz"), data={"x": X_train, "y": y_train})
    np.savez_compressed(os.path.join(save_dir, "test.npz"), data={"x": X_test, "y": y_test})

    config = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "num_classes": len(np.unique(y)),
        "input_dim": X.shape[1],
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        ujson.dump(config, f)

    print("Centralized dataset saved.")
    print("Train:", len(y_train))
    print("Test :", len(y_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=list(PRESETS) + ["centralized"])
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--class_per_client", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "centralized":
        generate_centralized_dataset()
    else:
        generate_dataset(args.mode, args.num_clients, args.class_per_client,
                         args.alpha, args.force)

    print("=" * 60)
    print("Dataset generation completed!")
    print("=" * 60)
