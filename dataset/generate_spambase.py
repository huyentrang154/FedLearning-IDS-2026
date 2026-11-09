import numpy as np
import pandas as pd
import os
import sys
import random
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "spambase/"
col_names = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "class",
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

    # Get spambase data
    data = np.genfromtxt(f'{dir_path}/rawdata/spambase.data', delimiter=",", dtype=np.float32, names=col_names)
    np.random.shuffle(data)
    data = pd.DataFrame(data)

    y = data["class"].to_numpy()
    X = data.drop("class", axis=1).to_numpy()

    print("Categorical Processed Size: ", (X.shape, y.shape))
    num_classes = len(set(y))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data(X, y, num_clients, num_classes, balance)
    train_data, test_data, mal_data = split_data(X, y)
    save_file(config_path, train_path, test_path, mal_path, train_data, test_data, mal_data, num_clients, num_classes,
              statistic, balance)


if __name__ == "__main__":
    balance = True if sys.argv[1] == "balance" else False
    generate_dataset(dir_path, num_clients, balance)
