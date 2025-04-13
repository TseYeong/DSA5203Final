import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNActivation(nn.Sequential):
    """
    A convolution block consisting Conv2D -> BatchNorm -> Activation
    Used as a basic unit in EfficientNet architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1,
                 norm_layer=None, activation_layer=None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            stride (int): Stride for the convolution.
            groups (int): Number of groups for grouped convolution.
            norm_layer (Callable): Normalization layer constructor.
            activation_layer (Callable): Activation function constructor.
        """
        padding = (kernel_size - 1) // 2
        norm_layer = norm_layer or nn.BatchNorm2d
        activation_layer = activation_layer or nn.SiLU
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            norm_layer(out_channels),
            activation_layer()
        )


class SqueezeExcitation(nn.Module):
    """
    Implements Squeeze-and-Excitation attention mechanism.
    """

    def __init__(self, in_channels, ex_channels, squeeze_factor=4):
        """
        Args:
            in_channels (int): Number of input channels before expansion.
            ex_channels (int): Number of channels after expansion.
            squeeze_factor (int): Reduction ratio for intermediate channels.
        """
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.fc1 = nn.Conv2d(ex_channels, squeeze_channels, 1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channels, ex_channels, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass that computes channel-wise attention and re-weights input.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Re-weighted output tensor.
        """
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.ac1(self.fc1(scale))
        scale = self.ac2(self.fc2(scale))
        return scale * x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck block with optional Squeeze-and-Excitation.
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 expand_ratio=6, se_ratio=0.25, norm_layer=None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Convolution stride.
            expand_ratio (int): Expansion ratio.
            se_ratio (float): Ratio for SE block.
            norm_layer (Callable): Normalization layer constructor.
        """
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        activation_layer = nn.SiLU
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBNActivation(
                    in_channels, hidden_dim, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=activation_layer
                )
            )
        layers.append(
            ConvBNActivation(
                hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim,
                norm_layer=norm_layer, activation_layer=activation_layer
            )
        )

        if se_ratio > 0:
            layers.append(
                SqueezeExcitation(
                    in_channels, hidden_dim, squeeze_factor=int(1 / se_ratio)
                )
            )
        layers.append(
            ConvBNActivation(
                hidden_dim, out_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Identity
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MBConv block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class EfficientNet(nn.Module):
    """
    A minimal version of EfficientNet for small-scale image classification.
    """
    def __init__(self, num_classes=15, width_mult=1.0):
        """
        Args:
            num_classes (int): Number of output classes.
            width_mult (float): Width scaling coefficient.
        """
        super().__init__()
        norm_layer = nn.BatchNorm2d
        adjust_channels = lambda c: int(c * width_mult)

        self.stem = ConvBNActivation(3, adjust_channels(32), stride=2, norm_layer=norm_layer)

        self.blocks = nn.Sequential(
            MBConvBlock(adjust_channels(32), adjust_channels(16), stride=1, expand_ratio=1, norm_layer=norm_layer),
            MBConvBlock(adjust_channels(16), adjust_channels(24), stride=2, norm_layer=norm_layer),
            MBConvBlock(adjust_channels(24), adjust_channels(24), stride=1, norm_layer=norm_layer),
            MBConvBlock(adjust_channels(24), adjust_channels(40), stride=2, norm_layer=norm_layer),
            MBConvBlock(adjust_channels(40), adjust_channels(40), stride=1, norm_layer=norm_layer)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(adjust_channels(40), num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the EfficientNetMini model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Classification logits and extracted features.
        """
        x = self.stem(x)
        x = self.blocks(x)
        features = self.avgpool(x).flatten(1)
        out = self.classifier(features)
        return out, features


def build_efficientnet(output_dim):
    return EfficientNet(output_dim)
