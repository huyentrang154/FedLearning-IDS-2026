import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from math import sqrt, ceil

from typing import Tuple, List



# Thiết lập thiết bị tính toán (CPU/GPU)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- 1. Định nghĩa Mô hình Autoencoder Classifier (Giữ nguyên) ---

class AE_Classifier(nn.Module):

    """

    Mạng Autoencoder kết hợp Bộ Phân loại (Predictor)

    """

    def __init__(self, n_features: int, num_classes: int):

        super(AE_Classifier, self).__init__()

        

        self.latent_size = int(ceil(sqrt(n_features))) + 1

        layer1_size = int(round(n_features * 0.75))

        layer2_size = int(round(n_features * 0.5))

        

        # Encoder

        self.encoder = nn.Sequential(

            nn.Linear(n_features, layer1_size), nn.Tanh(),

            nn.Linear(layer1_size, layer2_size), nn.Tanh(),

            nn.Linear(layer2_size, self.latent_size),

        )



        # Decoder

        self.decoder = nn.Sequential(

            nn.Linear(self.latent_size, layer2_size), nn.Tanh(),

            nn.Linear(layer2_size, layer1_size), nn.Tanh(),

            nn.Linear(layer1_size, n_features),

        )



        # Predictor

        predictor_layer1_size = int(round(self.latent_size * 0.75))

        predictor_layer2_size = int(round(self.latent_size * 0.5))



        self.predictor = nn.Sequential(

            nn.Linear(self.latent_size, predictor_layer1_size), nn.ReLU(),

            nn.Linear(predictor_layer1_size, predictor_layer2_size), nn.ReLU(),

            nn.Linear(predictor_layer2_size, num_classes),

        )

        self.to(DEVICE) # Đưa mô hình lên thiết bị



    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        latent = self.encoder(x)

        x_reconstruction = self.decoder(latent)

        out = self.predictor(latent)

        return out, x_reconstruction



# --- 2. Định nghĩa Custom Dataset cho PyTorch ---

class IDS_Dataset(Dataset):

    def __init__(self, features: np.ndarray, labels: np.ndarray):

        self.features = torch.tensor(features, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)



    def __len__(self) -> int:

        return len(self.labels)



    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.features[idx], self.labels[idx]



# --- 3. Hàm TIỀN XỬ LÝ (SỬ DỤNG FILE THỰC CỦA BẠN) ---

def load_and_preprocess_nslkdd(dir_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:

    """

    Đọc dữ liệu NSL-KDD và tiền xử lý (Scaling + One-Hot Encoding).

    >>> BẠN CẦN CHỈNH SỬA VÀ THAY THẾ HÀM NÀY BẰNG LOGIC ĐỌC DỮ LIỆU THỰC CỦA MÌNH <<<

    """

    feature_names = [

        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',

        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',

        'num_compromised', 'root_shell', 'su_attempted', 'num_root',

        'num_file_creations', 'num_shells', 'num_access_files',

        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',

        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',

        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',

        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',

        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',

        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',

        'dst_host_serror_rate', 'dst_host_srv_serror_rate',

        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',

        'attack_type', 'difficulty'

    ]



    try:

        # Thay thế bằng đường dẫn thực tế của bạn

        df_train = pd.read_csv(f'{dir_path}rawdata/NSLKDD_Train.csv', header=None, names=feature_names)

        df_test  = pd.read_csv(f'{dir_path}rawdata/NSLKDD_Test.csv', header=None, names=feature_names)
        # Trong load_and_preprocess_nslkdd, sau khi đọc file:
        SAMPLE_FRAC = 0.3 # Chỉ sử dụng 30% dữ liệu
        df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=42)
        # df_test = df_test.sample(frac=SAMPLE_FRAC, random_state=42)

    except FileNotFoundError:

        print("CẢNH BÁO: KHÔNG TÌM THẤY FILE NSL-KDD. TẠO DỮ LIỆU MÔ PHỎNG.")

        # Nếu không tìm thấy file, tạo dữ liệu mô phỏng để mã vẫn chạy được

        N_RAW_FEATURES = 40

        N_SAMPLES = 1000

        X = np.random.rand(N_SAMPLES, N_RAW_FEATURES)

        y = np.random.randint(0, 2, N_SAMPLES)

        return train_test_split(X, y, test_size=0.2, random_state=42) + (N_RAW_FEATURES, 2)



    # Logic tiền xử lý tương tự như mã Scikit-learn đã thống nhất

    X_train = df_train.drop(['attack_type', 'difficulty'], axis=1)

    X_test  = df_test.drop(['attack_type', 'difficulty'], axis=1)

    y_train = df_train['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values

    y_test  = df_test['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values

    

    categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    numerical_features = list(X_train.columns.drop(categorical_features))



    preprocessor = ColumnTransformer(

        transformers=[

            ('num', StandardScaler(), numerical_features),

            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)

        ]

    )



    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)

    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    

    n_features = X_train_processed.shape[1]

    num_classes = 2 # Nhị phân



    return X_train_processed, X_test_processed, y_train, y_test, n_features, num_classes



# --- 4. Hàm ĐÁNH GIÁ Mô hình ---

def evaluate_model(model: AE_Classifier, data_loader: DataLoader, criterion: nn.Module) -> dict:

    model.eval()

    all_labels = []

    all_preds = []

    total_loss = 0

    

    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            

            classification_output, reconstruction_output = model(inputs)

            

            # Tính loss tái tạo (không dùng để đánh giá hiệu suất phân loại, nhưng cần để kiểm tra)

            loss_reconstruction = criterion(reconstruction_output, inputs) 

            total_loss += loss_reconstruction.item()

            

            # Dự đoán

            _, predicted = torch.max(classification_output.data, 1)

            

            all_labels.extend(labels.cpu().numpy())

            all_preds.extend(predicted.cpu().numpy())



    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds, zero_division=0)

    recall = recall_score(all_labels, all_preds, zero_division=0)

    f1 = f1_score(all_labels, all_preds, zero_division=0)

    

    return {

        'Accuracy': accuracy,

        'Precision': precision,

        'Recall': recall,

        'F1-score': f1,

        'Avg_Rec_Loss': total_loss / len(data_loader)

    }



# --- 5. Quá trình Huấn luyện Mô hình Tổng thể ---

def train_ae_classifier(train_loader, test_loader, n_features, num_classes, epochs: int, learning_rate: float, alpha: float = 0.5):

    

    model = AE_Classifier(n_features, num_classes)

    

    # Loss Functions

    reconstruction_criterion = nn.MSELoss() 

    classification_criterion = nn.CrossEntropyLoss()

    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    

    print("\n--- Bắt đầu huấn luyện AE-Classifier ---")

    

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            

            optimizer.zero_grad()

            

            classification_output, reconstruction_output = model(inputs)

            

            loss_reconstruction = reconstruction_criterion(reconstruction_output, inputs)

            loss_classification = classification_criterion(classification_output, labels)

            

            loss = alpha * loss_reconstruction + (1 - alpha) * loss_classification

            

            loss.backward()

            optimizer.step()

            

            total_loss += loss.item()



        if (epoch + 1) % 10 == 0 or epoch == 1:

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {avg_loss:.4f}")

    

    print("Huấn luyện hoàn tất. Đánh giá trên tập kiểm tra...")

    

    # Đánh giá cuối cùng

    results = evaluate_model(model, test_loader, reconstruction_criterion)

    

    print("\n## 🎯 Kết quả AE-Classifier (Binary Classification)")

    print("--------------------------------------------------")

    print(f"| Accuracy:  {results['Accuracy']:.4f}")

    print(f"| Precision: {results['Precision']:.4f}")

    print(f"| Recall:    {results['Recall']:.4f}")

    print(f"| F1-score:  {results['F1-score']:.4f}")

    print(f"| Avg Rec Loss: {results['Avg_Rec_Loss']:.4f}")

    print("--------------------------------------------------")



# --- 6. Thực thi Chính (Main Execution) ---

if __name__ == '__main__':

    # 1. Định nghĩa các tham số

    DIR_PATH = 'nslkdd/' 

    BATCH_SIZE = 1     # Theo yêu cầu của bạn

    EPOCHS = 5         # Theo yêu cầu của bạn

    LEARNING_RATE = 1e-3

    

    print(f"Sử dụng thiết bị: {DEVICE}")



    # 2. Tải và tiền xử lý dữ liệu

    X_train, X_test, y_train, y_test, n_features, num_classes = load_and_preprocess_nslkdd(DIR_PATH)

    

    if n_features > 0:

        # 3. Tạo Dataset và DataLoader

        train_dataset = IDS_Dataset(X_train, y_train)

        test_dataset = IDS_Dataset(X_test, y_test)



        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        

        # 4. Bắt đầu huấn luyện

        train_ae_classifier(

            train_loader, test_loader, 

            n_features, num_classes, EPOCHS, LEARNING_RATE

        )