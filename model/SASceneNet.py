import torch
import torch.nn as nn
from torchvision.models import resnet
from SemBranch import BasicBlockSem, ChannelAttention
from model.resnet import build_resnet


class SASceneNet(nn.Module):
    def __init__(self, num_classes, depth=18, semantic_classes=151, pretrained=True):
        super().__init__()

        # Base Network
        if pretrained:
            if depth == 18:
                base = resnet.resnet18(pretrained=True)
                size_fc_RGB = 512
                size_lastConv = [512, 512, 512]
            elif depth == 50:
                base = resnet.resnet50(pretrained=True)
                size_fc_RGB = 2048
                size_lastConv = [2048, 2048, 2048]
            else:
                raise ValueError("Depth must be 18 or 50 for pretrained model.")
        else:
            base = build_resnet(output_dim=15, depth=depth)
            base.load_state_dict()  # Load parameters
            size_fc_RGB = 512 if depth == 18 else 2048
            size_lastConv = [512, 512, 512] if depth == 18 else [2048, 2048, 2048]

        # RGB branch
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # Semantic Branch
        self.in_block_sem = nn.Sequential(
            nn.Conv2d(semantic_classes + 1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.in_block_sem1 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.in_block_sem2 = BasicBlockSem(128, 256, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_3 = BasicBlockSem(256, 512, kernel_size=3, stride=2, padding=1)

        self.fc_RGB = nn.Linear(size_fc_RGB, num_classes)
        self.fc_SEM = nn.Linear(512, num_classes)

        # Attention Module
        self.lastConvRGB1 = nn.Sequential(
            nn.Conv2d(size_lastConv[0], size_lastConv[1], kernel_size=3, bias=False),
            nn.BatchNorm2d(size_lastConv[1]),
            nn.ReLU(inplace=True)
        )

        self.lastConvRGB2 = nn.Sequential(
            nn.Conv2d(size_lastConv[2], 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.lastConvSEM1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.lastConvSEM2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.avgpool7 = nn.AvgPool2d(7, stride=1)
        self.avgpool3 = nn.AvgPool2d(3, stride=1)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, sem):
        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        act_rgb = self.avgpool7(e4)
        act_rgb = act_rgb.view(act_rgb.size(0), -1)
        act_rgb = self.dropout(act_rgb)
        act_rgb = self.fc_RGB(act_rgb)

        y = self.in_block_sem(sem)
        y1 = self.in_block_sem1(y)
        y2 = self.in_block_sem2(y1)
        y3 = self.in_block_sem_3(y2)

        act_sem = self.avgpool7(y3)
        act_sem = act_sem.view(act_sem.size(0), -1)
        act_sem = self.dropout(act_sem)
        act_sem = self.fc_SEM(act_sem)

        e5 = self.lastConvRGB1(e4)
        e6 = self.lastConvSEM2(e5)
        y4 = self.lastConvSEM1(y3)
        y5 = self.lastConvSEM2(y4)

        e7 = e6 * self.sigmoid(y5)

        e8 = self.avgpool3(e7)
        act = e8.view(e8.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)

        return act, e7, act_rgb, act_sem
