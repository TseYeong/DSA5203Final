import torch
import torch.nn as nn


class BasicBlockSem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.ca(out) * out
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SemBranch(nn.Module):
    def __init__(self, out_classes, semantic_classes=151):
        super().__init__()
        self.in_block_sem = nn.Sequential(
            nn.Conv2d(semantic_classes + 1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_block_sem1 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.in_block_sem2 = BasicBlockSem(128, 256, kernel_size=3, stride=2, padding=1)
        self.in_block_sem3 = BasicBlockSem(256, 512, kernel_size=3, stride=2, padding=1)

        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_SEM = nn.Linear(512, out_classes)

    def forward(self, x, sem):
        y = self.in_block_sem(sem)
        y2 = self.in_block_sem_1(y)
        y3 = self.in_block_sem_2(y2)
        y4 = self.in_block_sem_3(y3)

        act_sem = self.avgpool(y4)
        act_sem = act_sem.view(act_sem.size(0), -1)
        act_sem = self.dropout(act_sem)
        act_sem = self.fc_SEM(act_sem)

        act = act_sem
        e5 = y4
        act_rgb = act_sem

        return act, e5, act_rgb, act_sem
