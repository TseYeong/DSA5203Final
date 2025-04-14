import torch
import torch.nn as nn
from torchvision.models import resnet
from resnet import build_resnet


class RGBBranch(nn.Module):
    def __init__(self, depth=18, num_classes=15, pretrained=True):
        super().__init__()

        if pretrained:
            if depth == 18:
                base = resnet.resnet18(pretrained=True)
                size_fc_RGB = 512
            elif depth == 50:
                base = resnet.resnet50(pretrained=True)
                size_fc_RGB = 2048
        else:
            base = build_resnet(output_dim=15, depth=depth)
            base.load_state_dict()  # Load parameters
            size_fc_RGB = 512 if depth == 18 else 2048

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

        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(size_fc_RGB, num_classes)

    def forward(self, x):
        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        act = self.avgpool(e4)
        act = self.view(act.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)

        act_rgb = act
        act_sem = act

        return act, e4, act_rgb, act_sem
