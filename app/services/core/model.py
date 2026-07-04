"""
PyTorch model architectures for Segment Recolor
- FeatureEncoder: CNN encoder with ResNet blocks for feature extraction
- RecoloringDecoder: Decoder with skip connections for image recoloring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict


class Conv2dAuto(nn.Conv2d):
    """Conv2d with automatic same padding"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    """Base residual block with identity mapping"""
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


class ResNetResidualBlock(ResidualBlock):
    """ResNet residual block with configurable expansion and downsampling"""
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=2, conv=conv3x3):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(in_channels, out_channels * expansion,
                              kernel_size=1, stride=downsampling, bias=False, padding=0),
            'bn': nn.InstanceNorm2d(out_channels * expansion)
        })) if in_channels != out_channels * expansion else None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    """Helper: Conv2d + InstanceNorm2d"""
    return nn.Sequential(OrderedDict({
        'conv': conv(in_channels, out_channels, *args, **kwargs),
        'bn': nn.InstanceNorm2d(out_channels)
    }))


class ResNetBasicBlock(ResNetResidualBlock):
    """Basic ResNet block: 2 conv layers"""
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(in_channels, out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            nn.LeakyReLU(negative_slope=0.02),
            conv_bn(out_channels, out_channels, conv=self.conv, bias=False),
        )


class FeatureEncoder(nn.Module):
    """
    CNN Feature Encoder for extracting multi-scale features from images.
    
    Architecture:
    - Initial conv + norm + pool → 64 channels
    - ResNet blocks → 128, 256, 512 channels
    - Returns 4 feature maps: c1 (512ch), c2 (256ch), c3 (128ch), c4 (64ch)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.res1 = ResNetBasicBlock(64, 128)
        self.res2 = ResNetBasicBlock(128, 256)
        self.res3 = ResNetBasicBlock(256, 512)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, H, W) input image tensor in LAB color space
        
        Returns:
            Tuple of 4 feature maps: (c1, c2, c3, c4)
        """
        x = F.relu(self.norm(self.conv(x)))
        c4 = self.pool(x)
        c3 = self.res1(c4)
        c2 = self.res2(c3)
        c1 = self.res3(c2)
        return c1, c2, c3, c4


def de_conv(in_channels, out_channels):
    """Helper: Transpose Conv + InstanceNorm + LeakyReLU"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                           stride=2, output_padding=1, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.02, inplace=True)
    )


class RecoloringDecoder(nn.Module):
    """
    Decoder for recoloring images using feature maps and target color palette.
    
    Architecture:
    - 4 deconvolution layers with skip connections from encoder
    - Concatenates color palette at multiple scales
    - Returns 2-channel output (a, b channels in LAB space)
    - L channel (illumination) is preserved from input
    """
    def __init__(self):
        super().__init__()
        self.dconv_up_4 = de_conv(18 + 512, 256)
        self.dconv_up_3 = de_conv(256 + 256, 128)
        self.dconv_up_2 = de_conv(18 + 128 + 128, 64)
        self.dconv_up_1 = de_conv(18 + 64 + 64, 64)
        self.conv_last = nn.Conv2d(1 + 64, 2, kernel_size=3, padding=1)

    def forward(self, c1, c2, c3, c4, target_palettes_1d, illu):
        """
        Args:
            c1, c2, c3, c4: Feature maps from encoder
            target_palettes_1d: (batch, 18) flattened color palette (6 colors × 3 channels)
            illu: (batch, H, W) illumination channel from input
        
        Returns:
            (batch, 2, H, W) output tensor - a,b channels for recolored image
        """
        bz, h, w = c1.shape[0], c1.shape[2], c1.shape[3]
        tp_reshaped = target_palettes_1d.reshape(bz, 18, 1, 1)

        # Decoder layer 4
        x = torch.cat((c1, tp_reshaped.repeat(1, 1, h, w)), dim=1)
        x = self.dconv_up_4(x)

        # Decoder layer 3
        x = torch.cat([c2, x], dim=1)
        x = self.dconv_up_3(x)

        # Decoder layer 2
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        x = torch.cat([tp_reshaped.repeat(1, 1, h, w), c3, x], dim=1)
        x = self.dconv_up_2(x)

        # Decoder layer 1
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        x = torch.cat([tp_reshaped.repeat(1, 1, h, w), c4, x], dim=1)
        x = self.dconv_up_1(x)

        # Final output with illumination
        illu = illu.view(illu.size(0), 1, illu.size(2), illu.size(3))
        x = torch.cat((x, illu), dim=1)
        x = self.conv_last(x)
        return torch.tanh(x)
