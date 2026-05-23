import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class CNN1D(nn.Module):
    def __init__(self, in_channels=7, num_classes=5,
                 channels=(64, 128, 256), kernels=(7, 5, 3),
                 strides=(2, 2, 2), dropout=0.5):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(channels)):
            self.blocks.append(ConvBlock(
                in_channels if i == 0 else channels[i - 1],
                channels[i], kernels[i], strides[i], dropout,
            ))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

    def get_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return x
