import os
import numpy as np
import ujson
from sklearn.model_selection import train_test_split

batch_size = 8
train_ratio = 0.8


def save_centralized_dataset(
        X,
        y,
        save_dir,
):
    """
    Tạo dataset cho Centralized Learning.

    Output:
        train.npz
        test.npz
        config.json
    """

    os.makedirs(save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_ratio,
        shuffle=True,
        stratify=y,
        random_state=42
    )

    train_dict = {
        "x": X_train,
        "y": y_train
    }

    test_dict = {
        "x": X_test,
        "y": y_test
    }

    np.savez_compressed(
        os.path.join(save_dir, "train.npz"),
        data=train_dict
    )

    np.savez_compressed(
        os.path.join(save_dir, "test.npz"),
        data=test_dict
    )

    config = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "batch_size": batch_size
    }

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        ujson.dump(config, f)

    print("====================================")
    print("Centralized dataset generated.")
    print("Train:", len(y_train))
    print("Test :", len(y_test))
    print("====================================")
