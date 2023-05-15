import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import ConvBlock, DeConvBlock
from tqdm import tqdm
import torch.nn.functional as F


def normalize_pixels(img):
    """
    归一化图片像素值
    :param img: 待归一化的图片
    :return: 归一化后的图片
    """
    # 将像素值转换为 float 类型
    img = img.astype(np.float32)
    # 归一化像素值，将像素值范围缩放到 [0, 1]
    img = img / 255.0
    return img


origin_data = sio.loadmat('Data/rgbd_mtv.mat')
label = origin_data['gt'][:, 0]
all_features = origin_data['X']

view_shape = []
views = []
for v in all_features[0]:
    view_shape.append(v.shape[1])
    views.append(v)

# 我们先从single_view 开始，以第一个视图为例
single_view = views[0]
num_classes = np.unique(label).shape[0]

reg1 = 1.0
reg2 = 1.0
alpha = max(0.4 - (num_classes - 1) / 10 * 0.1, 0.1)
lr = 1e-3

views[0] = np.transpose(views[0], [0, 3, 1, 2])
views[1] = np.transpose(views[1], [0, 3, 1, 2])
# views[0] = normalize_pixels(views[0])
del views[2]
tensors = [torch.from_numpy(arr) for arr in (views)]

from MvDSCN import MsDSCN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# init_model = MsDSCN(views_data=tensors, n_samples=label.shape[0], device=device, learning_rate=1e-3, epochs=200,
#                     ft=False)
# model = init_model.train()
# torch.cuda.empty_cache()


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.de_conv1 = nn.ConvTranspose2d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)
        self.de_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)
        # self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
        # output_padding=1, padding=1)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)

    def forward(self, x):
        x = self.de_conv1(x)
        x = F.relu(x)
        x = self.de_conv2(x)
        x = F.relu(x)
        x = self.de_conv3(x)
        x = F.relu(x)
        return x


class AutoEncoderInit(nn.Module):
    def __init__(self, batch_size):
        super(AutoEncoderInit, self).__init__()
        # different view feature input
        self.batch_size = batch_size

        self.encoder1 = Encoder(input_dim=3, output_dim=64)
        self.encoder2 = Encoder(input_dim=1, output_dim=64)
        self.encoder1_single = Encoder(input_dim=3, output_dim=64)
        self.encoder2_single = Encoder(input_dim=1, output_dim=64)

        self.decoder1 = Decoder(input_dim=64, output_dim=3)
        self.decoder2 = Decoder(input_dim=64, output_dim=1)
        self.decoder1_single = Decoder(input_dim=64, output_dim=3)
        self.decoder2_single = Decoder(input_dim=64, output_dim=1)

    def forward(self, all_views_data):
        view_1_data = all_views_data[0]
        view_2_data = all_views_data[1]
        rec_x1 = self.decoder1(self.encoder1(view_1_data))
        rec_x2 = self.decoder2(self.encoder2(view_2_data))
        rec_x1_single = self.decoder1_single(self.encoder1_single(view_1_data))
        rec_x2_single = self.decoder2_single(self.encoder2_single(view_2_data))
        return rec_x1, rec_x2, rec_x1_single, rec_x2_single


model = AutoEncoderInit(batch_size=500)
tensors[0] = tensors[0].to(device)
tensors[1] = tensors[1].to(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
loss_values = []
epoch_iter = tqdm(range(300))
for epoch in epoch_iter:
    model.train()
    view1_out, view2_out, view1_out_single, view2_out_single = model(tensors)
    loss1 = 0.5 * torch.norm(view1_out - tensors[0], p=2) ** 2
    loss2 = 0.5 * torch.norm(view2_out - tensors[1], p=2) ** 2
    loss3 = 0.5 * torch.norm(view1_out_single - tensors[0], p=2) ** 2
    loss4 = 0.5 * torch.norm(view2_out_single - tensors[1], p=2) ** 2
    loss = loss1 + loss2 + loss3 + loss4
    epoch_iter.set_description(
        f"# Epoch {epoch}, train_loss: {loss.item():.4f}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}, loss3: {loss3.item():.4f},loss4: {loss4.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

rec_1_data = view1_out.detach().cpu().numpy()
rec_2_data = view2_out.detach().cpu().numpy()
rec_3_data = view1_out_single.detach().cpu().numpy()
rec_4_data = view2_out_single.detach().cpu().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(np.transpose(np.clip(rec_1_data[1], 0, 1), [1, 2, 0]))
plt.show()
plt.imshow(np.transpose(np.clip(rec_2_data[1], 0, 1), [1, 2, 0]))
plt.show()
plt.imshow(np.transpose(np.clip(rec_3_data[1], 0, 1), [1, 2, 0]))
plt.show()
plt.imshow(np.transpose(np.clip(rec_4_data[1], 0, 1), [1, 2, 0]))
plt.show()
plt.imshow(np.transpose(np.clip(views[0][1], 0, 1), [1, 2, 0]))
plt.show()
