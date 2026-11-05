"""Self-contained ResNet-34 encoder used by the MDPNet variants.

The upstream repository imports torchvision at module import time.  Keeping the
encoder here makes architecture tests possible in a minimal PyTorch environment
and preserves MDPNet's deliberate omission of the stem max-pooling operation.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn


MODEL_URL = "https://download.pytorch.org/models/resnet34-b627a593.pth"


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet34Encoder(nn.Module):
    """ResNet-34 feature encoder with output strides 2, 4, 8 and 16.

    Standard ResNet uses a max-pool after the stem.  The public MDPNet code does
    not, so ``use_maxpool=False`` is the default and reproduces its feature sizes.
    """

    def __init__(self, pretrained: bool = True, use_maxpool: bool = False) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.use_maxpool = use_maxpool

        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self._initialize_weights()
        if pretrained:
            self.load_imagenet_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: List[nn.Module] = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_imagenet_weights(
        self,
        progress: bool = True,
        loader: Callable[..., dict[str, Tensor]] = torch.hub.load_state_dict_from_url,
    ) -> None:
        state_dict = loader(MODEL_URL, progress=progress, map_location="cpu")
        encoder_state = {key: value for key, value in state_dict.items() if not key.startswith("fc.")}
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Unexpected ResNet-34 checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        if self.use_maxpool:
            x = self.maxpool(x)

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        return layer_1, layer_2, layer_3, layer_4
