# import torch
# import torch.nn as nn
# from math import sqrt


# class MLP(nn.Module):
#     def __init__(self, n_features, num_classes):
#         super(MLP, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(n_features, round(n_features * 0.75)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),
#         )

#         self.predictor = nn.Sequential(nn.Linear(round(sqrt(n_features)) + 1, 128),
#                                        nn.ReLU(),
#                                        nn.Linear(128, num_classes)
#                                        )

#     def forward(self, x):
#         # encoder
#         out = self.encoder(x)
#         # predictor
#         out = torch.flatten(out, 1)
#         out = self.predictor(out)
#         return out


# class AE(nn.Module):
#     def __init__(self, n_features, num_classes):
#         super(AE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(n_features, round(n_features * 0.75)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(round(sqrt(n_features)) + 1, round(n_features * 0.5)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.75), n_features),
#         )
#         # Bộ dự đoán (Predictor)
#         self.predictor = nn.Sequential(
#             nn.Linear(round(sqrt(n_features)) + 1, round(round(sqrt(n_features)) + 1 * 0.75)),  # Giảm xuống 75% bottleneck
#             nn.ReLU(),
#             nn.Linear(round(round(sqrt(n_features)) + 1 * 0.75), round(round(sqrt(n_features)) + 1 * 0.5)),  # Giảm xuống 50% bottleneck
#             nn.ReLU(),
#             nn.Linear(round(round(sqrt(n_features)) + 1 * 0.5), num_classes),  # Ánh xạ cuối cùng tới số lớp
#         )

#     def forward(self, x):
#         latent = self.encoder(x)
#         x_reconstruction = self.decoder(latent)
#         out = torch.flatten(latent, 1)
#         out = self.predictor(out)
#         return out, x_reconstruction
# import torch
# import torch.nn as nn
# from math import sqrt


# # Encoder riêng để tái sử dụng
# class Encoder(nn.Module):
#     def __init__(self, n_features):
#         super(Encoder, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(n_features, round(n_features * 0.75)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),
#         )

#     def forward(self, x):
#         return self.layers(x)


# class MLP(nn.Module):
#     def __init__(self, n_features, num_classes):
#         super(MLP, self).__init__()
#         self.encoder = Encoder(n_features)

#         self.predictor = nn.Sequential(
#             nn.Linear(round(sqrt(n_features)) + 1, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         latent = self.encoder(x)        # encoder
#         out = torch.flatten(latent, 1)  # flatten
#         out = self.predictor(out)       # predictor
#         return out


# class AE(nn.Module):
#     def __init__(self, n_features, num_classes):
#         super(AE, self).__init__()
#         self.encoder = Encoder(n_features)

#         self.decoder = nn.Sequential(
#             nn.Linear(round(sqrt(n_features)) + 1, round(n_features * 0.5)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),
#             nn.Tanh(),
#             nn.Linear(round(n_features * 0.75), n_features),
#         )

#         # Predictor riêng
#         bottleneck_dim = round(sqrt(n_features)) + 1
#         self.predictor = nn.Sequential(
#             nn.Linear(bottleneck_dim, round(bottleneck_dim * 0.75)),
#             nn.ReLU(),
#             nn.Linear(round(bottleneck_dim * 0.75), round(bottleneck_dim * 0.5)),
#             nn.ReLU(),
#             nn.Linear(round(bottleneck_dim * 0.5), num_classes),
#         )

#     def forward(self, x):
#         latent = self.encoder(x)  # encoder
#         x_reconstruction = self.decoder(latent)  # decoder
#         out = torch.flatten(latent, 1)
#         out = self.predictor(out)  # predictor
#         return out, x_reconstruction
import torch
import torch.nn as nn
from math import sqrt


class MLP(nn.Module):
    def __init__(self, n_features, num_classes):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),
        )

        self.predictor = nn.Sequential(nn.Linear(round(sqrt(n_features)) + 1, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, num_classes)
                                       )

    def forward(self, x):
        # encoder
        out = self.encoder(x)
        # predictor
        out = torch.flatten(out, 1)
        out = self.predictor(out)
        return out


class AE(nn.Module):
    def __init__(self, n_features, num_classes):
        super(AE, self).__init__()

        self.latent_dim = round(sqrt(n_features)) + 1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), self.latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), n_features),
        )

        # Predictor / Classifier
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, round(self.latent_dim * 0.75)),
            nn.ReLU(),
            nn.Linear(
                round(self.latent_dim * 0.75),
                round(self.latent_dim * 0.5)
            ),
            nn.ReLU(),
            nn.Linear(
                round(self.latent_dim * 0.5),
                num_classes
            ),
        )

    def forward(self, x):
        latent = self.encoder(x)

        x_reconstruction = self.decoder(latent)

        out = self.predictor(latent)

        return out, x_reconstruction
# class CNN(nn.Module):
#     def __init__(self, input_channels, image_size, num_classes):
#         super(CNN, self).__init__()
#         self.input_channels = input_channels
#         self.image_size = image_size
        
#         # Encoder: Trích xuất đặc trưng không gian
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # Giảm size đi 2 lần
            
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)  # Giảm size thêm 2 lần (tổng cộng giảm 4)
#         )

#         # Tính toán kích thước đầu ra sau khi qua các lớp Conv/Pool
#         # Với CIFAR (32x32), sau 2 lần MaxPool sẽ còn 8x8
#         self.reduced_size = image_size // 4
#         self.flat_features = 64 * self.reduced_size * self.reduced_size

#         self.predictor = nn.Sequential(
#             nn.Linear(self.flat_features, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         # Đảm bảo dữ liệu ở dạng 4D [Batch, C, H, W]
#         if len(x.shape) == 2:
#             x = x.view(-1, self.input_channels, self.image_size, self.image_size)
        
#         latent = self.encoder(x)
#         out = torch.flatten(latent, 1)
#         out = self.predictor(out)
#         return out

# class CNNAE(nn.Module):
#     def __init__(self, input_channels, image_size, num_classes):
#         super(CNNAE, self).__init__()
#         self.input_channels = input_channels
#         self.image_size = image_size
        
#         # 1. Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 16, 3, stride=2, padding=1), # size / 2
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),             # size / 4
#             nn.ReLU()
#         )
        
#         # 2. Decoder (Tái tạo ảnh)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid() # Giữ pixel trong khoảng [0, 1]
#         )

#         # 3. Predictor (Phân loại từ không gian latent)
#         reduced_size = image_size // 4
#         self.latent_dim = 32 * reduced_size * reduced_size
        
#         self.predictor = nn.Sequential(
#             nn.Linear(self.latent_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         if len(x.shape) == 2:
#             x = x.view(-1, self.input_channels, self.image_size, self.image_size)
            
#         latent = self.encoder(x)
#         x_reconstruction = self.decoder(latent)
        
#         # Flatten latent để đưa vào bộ dự đoán
#         out = torch.flatten(latent, 1)
#         logits = self.predictor(out)
        
#         return logits, x_reconstruction
# 
