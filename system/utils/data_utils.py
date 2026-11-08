# import random

# import numpy as np
# import os
# import torch


# def read_data(dataset, idx, data_type, type):
#     data_dirs = {
#         'train': 'train',
#         'test': 'test',
#     }

#     if data_type not in data_dirs:
#         raise ValueError("Invalid data_type. Expected one of: 'train', 'test', 'mal'")

#     data_dir = os.path.join('../dataset', dataset, type, data_dirs[data_type])
#     data_file = os.path.join(data_dir, f'{idx}.npz')

#     with open(data_file, 'rb') as f:
#         data = np.load(f, allow_pickle=True)['data'].tolist()

#     return data


# def read_client_data(dataset, idx, data_type='train', type='noniid'):
#     data = read_data(dataset=dataset, idx=idx, data_type=data_type, type=type)
#     X, y = data['x'], data['y']
#     X = torch.Tensor(X).type(torch.float32)
#     y = torch.Tensor(y).type(torch.int64)

#     data = [(x, y) for x, y in zip(X, y)]
#     return data

import random

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

NORMAL_LABEL = {
    'nslkdd': 0,
    'cic_ids': 6,   # BENIGN = 6 (xem mapping trong dataset/generate_cic_ids.py)
    'ctu13_13': 0,
    'ctu13_08': 0,
    'ctu13_09': 0,
    'ctu13_10': 0,
    'internet_ad': 0,
    'spambase': 0,
    'unsw': 0,
    'bot_iot': 0,
    'crawl': 0,
    'cifar10': 0,
    'toniot': 5,    # LabelEncoder alphabet: backdoor,ddos,dos,injection,mitm,normal=5,...
}


def resolve_data_type(data_type, num_clients=20):
    """Chuẩn hóa tên thư mục dữ liệu (noniid20, noniidexdir20, ...)."""
    if num_clients is None:
        num_clients = 20
    suffix = str(num_clients)
    dt = str(data_type or 'noniid').strip()
    dt_lower = dt.lower()

    if dt_lower in ('noniid', f'noniid{suffix}'):
        return f'noniid{suffix}'
    if dt_lower in ('exdir', 'noniidexdir', f'exdir{suffix}', f'noniidexdir{suffix}'):
        return f'noniidexdir{suffix}'
    if dt_lower == 'iid':
        return 'iid'
    if dt_lower.endswith(suffix):
        return dt
    return f'{dt}{suffix}'


def read_data(dataset, idx, data_type, type):
    data_dirs = {
        'train': 'train',
        'test': 'test',
    }

    if data_type not in data_dirs:
        raise ValueError("Invalid data_type. Expected one of: 'train', 'test', 'mal'")

    data_dir = os.path.join('../dataset', dataset, type, data_dirs[data_type])
    data_file = os.path.join(data_dir, f'{idx}.npz')

    with open(data_file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()

    return data


def read_client_data(dataset, idx, num_clients=None, data_type='train', type='noniid',
                     partition=None, alpha=None, poison_attack=None):
    if num_clients is None:
        num_clients = 20

    dir_type = resolve_data_type(type, num_clients)

    # Nếu idx là None hoặc -1, ta sẽ lấy dữ liệu của TẤT CẢ các client
    if idx is None or idx == -1:
        all_X = []
        all_y = []
        for i in range(num_clients):
            data = read_data(dataset=dataset, idx=i, data_type=data_type, type=dir_type)
            all_X.append(torch.Tensor(data['x']).type(torch.float32))
            all_y.append(torch.Tensor(data['y']).type(torch.int64))

        # Gộp tất cả tensor lại thành 1 tensor lớn
        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)
    else:
        # Lấy dữ liệu của 1 client cụ thể như cũ
        data = read_data(dataset=dataset, idx=idx, data_type=data_type, type=dir_type)
        X = torch.Tensor(data['x']).type(torch.float32)
        y = torch.Tensor(data['y']).type(torch.int64)

    data_list = [(x, y) for x, y in zip(X, y)]
    return data_list
# +'dir'

# import torch
# from torch.utils.data import TensorDataset

# def read_client_data(dataset, idx, num_clients, data_type='train', type='noniid'):
#     data = read_data(dataset=dataset, idx=idx, data_type=data_type, type=type + str(num_clients))
#     X, y = data['x'], data['y']

#     # Convert sang Tensor
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.int64)

#     # Map nhãn sang nhị phân: 0 = normal, 1 = abnormal
#     y_bin = (y > 0).long()

#     # Trả về TensorDataset (X, y_bin)
#     dataset = TensorDataset(X, y_bin)
#     return dataset
def read_centralized_data(dataset, data_type="train"):
    """
    Đọc dữ liệu cho Centralized Learning.

    Cấu trúc:
        ../dataset/
            dataset/
                noniid20/
                    centralized/
                        train/
                            all.npz
                        test/
                            0.npz
                            1.npz
                            ...
                            19.npz

    TRAIN:
        Đọc trực tiếp train/all.npz.

    TEST:
        Đọc test của 20 client và trả về list gồm
        TensorDataset của từng client.
    """

    num_clients = 20

    data_path = os.path.join(
        "..",
        "dataset",
        dataset,
        "noniid20",
        "centralized"
    )

    # ============================================================
    # TRAIN
    # ============================================================

    if data_type == "train":

        file_path = os.path.join(
            data_path,
            "train",
            "all.npz"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Không tìm thấy file train: {file_path}"
            )

        data = np.load(
            file_path,
            allow_pickle=True
        )

        train_data = data["data"].item()

        X = train_data["x"]
        y = train_data["y"]

        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

    # ============================================================
    # TEST
    # ============================================================

    elif data_type == "test":

        test_clients = []

        for client_id in range(num_clients):

            file_path = os.path.join(
                data_path,
                "test",
                f"{client_id}.npz"
            )

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Không tìm thấy file test client {client_id}: "
                    f"{file_path}"
                )

            data = np.load(
                file_path,
                allow_pickle=True
            )

            client_data = data["data"].item()

            X = client_data["x"]
            y = client_data["y"]

            client_dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )

            test_clients.append(client_dataset)

        return test_clients

    else:
        raise ValueError(
            "data_type must be 'train' or 'test'"
        )