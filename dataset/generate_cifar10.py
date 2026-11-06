import os
import sys
import numpy as np
import torchvision
from utils.dataset_utils import check, separate_data, split_data, save_file

# Cấu hình giống hệt file cũ của bạn
num_clients = 20
dir_path = "cifar10/"  # Thư mục cũ bạn muốn lưu vào

from PIL import Image # Thư viện để lưu ảnh
# ... các import cũ của bạn ...

# def generate_dataset(dir_path, num_clients, niid, balance, partition):
#     # Tạo thư mục rawdata nếu chưa có
#     raw_preview_path = os.path.join(dir_path, "rawdata", "preview")
#     if not os.path.exists(raw_preview_path):
#         os.makedirs(raw_preview_path)

#     # 1. Tải dữ liệu từ torchvision
#     print("Đang tải dữ liệu...")
#     raw_train = torchvision.datasets.CIFAR10(root='./raw_data', train=True, download=True)
    
#     # --- ĐOẠN LƯU ẢNH ĐỂ XEM THỬ ---
#     print(f"Đang lưu 10 ảnh mẫu vào {raw_preview_path}...")
#     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
#     for i in range(10): # Lưu 10 ảnh đầu tiên
#         img_array = raw_train.data[i] # Lấy mảng 32x32x3
#         label = raw_train.targets[i]
#         img = Image.fromarray(img_array)
#         img.save(os.path.join(raw_preview_path, f"sample_{i}_{classes[label]}.png"))

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_type = 'noniid' if niid else 'iid'
    
    # Thiết lập đường dẫn lưu file theo cấu trúc cũ
    config_path = os.path.join(dir_path, f"{data_type}{num_clients}{partition}", "config.json")
    train_path = os.path.join(dir_path, f"{data_type}{num_clients}{partition}", "train/")
    test_path = os.path.join(dir_path, f"{data_type}{num_clients}{partition}", "test/")

    # Kiểm tra nếu đã có dữ liệu thì không chạy lại
    if check(config_path, train_path, test_path, num_clients, balance):
        print("Dữ liệu đã tồn tại, bỏ qua bước tạo mới.")
        return

    # --- TẢI DỮ LIỆU CHUẨN (Thay thế cho TFRecord bị lỗi) ---
    print("Đang tải CIFAR-10 chuẩn từ torchvision...")
    raw_train = torchvision.datasets.CIFAR10(root='./raw_data', train=True, download=True)
    raw_test = torchvision.datasets.CIFAR10(root='./raw_data', train=False, download=True)

    # Chuyển về Numpy và ép phẳng (Flatten) thành vector 3072 chiều
    X_train = raw_train.data.reshape(50000, -1)
    y_train = np.array(raw_train.targets)
    
    X_test = raw_test.data.reshape(10000, -1)
    y_test = np.array(raw_test.targets)

    # Gộp dữ liệu train/test lại để chia cho Client theo logic của bạn
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Ép kiểu và chuẩn hóa về [0, 1] (giống MinMaxScaler)
    X = X.astype(np.float64) / 255.0

    num_classes = 10
    print(f"Kích thước dữ liệu gốc: {X.shape}, Số lớp: {num_classes}")

    # --- CHIA DỮ LIỆU CHO CLIENTS (Giữ nguyên logic cũ của bạn) ---
    # Phân chia Non-IID với 2 lớp mỗi Client (phù hợp cho bài toán học biểu diễn)
    X, y, statistic = separate_data((X, y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    
    # Chia nhỏ thành tập train/test cho từng Client
    train_data, test_data = split_data(X, y)
    
    # Lưu lại vào folder cũ theo đúng định dạng luận văn yêu cầu
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
              statistic, balance)
    print(f"Đã lưu dữ liệu vào: {dir_path}")

if __name__ == "__main__":
    # Chạy lệnh: python generate_cifar10.py noniid balance -
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
# thuc chat la pat dang thu de train thoi