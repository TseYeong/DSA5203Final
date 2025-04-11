import torch
import torch.nn as nn


class ResNet(nn.Module):
    """
    A configurable ResNet architecture.
    """
    def __init__(self, config, output_dim):
        """
        Args:
            config (tuple): (block, n_blocks, channels) configuration.
            output_dim (int): Number of output classes.
        """
        super().__init__()
        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, n_blocks[0], channels[0])
        self.layer2 = self._make_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self._make_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def _make_layer(self, block, n_blocks, channels, stride=1):
        """
        Constructs one ResNet layer with repeated blocks.

        Args:
            block (class): Block type of ResNet.
            n_blocks (int): Number of blocks in the layer.
            channels (int): Output channel size for this layer.
            stride (int): Stride of the first block.

        Returns:
            Sequential: A sequence of residual blocks.
        """
        downsample = False
        if stride != block.expansion * channels:
            downsample = True

        layers = [block(self.in_channels, channels, stride, downsample)]

        for _ in range(1, n_blocks):
            layers.append(block(channels * block.expansion, channels))

        self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the full ResNet model.
        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tuple[Tensor, Tensor]: Classification logits and feature vector.
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h


class BasicBlock(nn.Module):
    """
    A basic residual block with two convolutional layers.
    Used in ResNet-18 and ResNet-34.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution.
            downsample (bool): Whether to apply down-sampling in the skip connection.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if downsample else None

    def forward(self, x):
        """
        Forward pass of the BasicBlock.
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor after residual addition and activation.
        """
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """
    A bottleneck residual block used in large scale ResNet.
    """
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=False):
        """
        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate channels.
            stride (int): Stride for the 3x3 convolution.
            downsample (bool): Whether to apply down-sampling in the skip connection.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, self.expansion * mid_channels,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * mid_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, self.expansion * mid_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * mid_channels)
        ) if downsample else None

    def forward(self, x):
        """
        Forward pass of the bottleneck block.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor after residual addition and activation.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


def build_resnet(output_dim, depth=18):
    """
    Factory function to build ResNet of varying depths.

    Args:
        output_dim (int): Number of output classes.
        depth (int): Depth of the ResNet, one of [18, 34, 50, 101].

    Returns:
        ResNet: A ResNet model instance.
    """
    if depth == 18:
        config = (BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512])
    elif depth == 34:
        config = (BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512])
    elif depth == 50:
        config = (Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512])
    elif depth == 101:
        config = (Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512])
    else:
        raise ValueError("Unsupported depth. Use one of: 18, 34, 50, 101")

    return ResNet(config, output_dim)
