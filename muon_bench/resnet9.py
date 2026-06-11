"""ResNet-9 for CIFAR-10.

The 9-layer residual network popularized by David Page's "How to Train Your
ResNet" series (DAWNBench) -- the standard architecture for fast
time-to-94%-accuracy CIFAR-10 benchmarks. 9 weight layers (8 conv + 1
linear), ~6.57M parameters.

Layout:
    prep:   conv3-64                          32x32
    layer1: conv3-128, maxpool                16x16
            residual(conv3-128, conv3-128)
    layer2: conv3-256, maxpool                 8x8
    layer3: conv3-512, maxpool                 4x4
            residual(conv3-512, conv3-512)
    head:   maxpool4, flatten, linear-10, logits * scale

Every conv is 3x3 / pad 1 / no bias, followed by BatchNorm + ReLU. The
final logit scale (default 0.125) compensates for the un-normalized
linear head and stabilizes the loss early in training.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ResNet9", "build_resnet9"]


def conv_bn_relu(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    """y = x + f(x) where f is two conv-bn-relu blocks at constant width."""

    def __init__(self, channels: int):
        super().__init__()
        self.inner = nn.Sequential(
            conv_bn_relu(channels, channels),
            conv_bn_relu(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inner(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 10, logit_scale: float = 0.125):
        super().__init__()
        self.logit_scale = logit_scale

        self.prep = conv_bn_relu(3, 64)
        self.layer1 = nn.Sequential(conv_bn_relu(64, 128, pool=True), Residual(128))
        self.layer2 = conv_bn_relu(128, 256, pool=True)
        self.layer3 = nn.Sequential(conv_bn_relu(256, 512, pool=True), Residual(512))
        self.head = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x) * self.logit_scale


def build_resnet9(device: torch.device | str = "cuda", channels_last: bool = True) -> ResNet9:
    """Build ResNet-9 on `device`, optionally in channels-last memory format
    (faster convolutions with bf16 autocast on Ampere+ GPUs)."""
    model = ResNet9().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model
