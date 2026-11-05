"""MDPNet-V2: a controlled improvement of the released MDPNet code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from backbone import ResNet34Encoder
from modules import DynamicScaleFusion, LegacyDPFocus, LegacyFusion, RefinedDPFocus


@dataclass(frozen=True)
class MDPNetConfig:
    """Architecture switches used for fair ablations."""

    pretrained_backbone: bool = False
    use_sk_lmap: bool = True
    use_position_attention: bool = True
    refine_dpfocus: bool = True
    dynamic_fusion: bool = True
    deep_supervision: bool = False
    max_position_tokens: int = 64
    # Keep the released DPFocus probability boundary only for an explicit
    # compatibility run. Fair structural ablations use logits throughout.
    legacy_dpfocus_sigmoid: bool = False

    def __post_init__(self) -> None:
        if self.legacy_dpfocus_sigmoid and self.refine_dpfocus:
            raise ValueError(
                "legacy_dpfocus_sigmoid is only valid when refine_dpfocus=False"
            )


class MDPNetV2(nn.Module):
    """Improved MDPNet with independently switchable contributions.

    The stage channel counts and feature resolutions intentionally match the
    released MDPNet implementation: (2, 4, 8, 16) output channels at strides
    (2, 4, 8, 16). This makes checkpoint loading and ablation interpretation
    straightforward while allowing the new modules to be tested in isolation.
    """

    def __init__(self, out_channels: int = 1, config: Optional[MDPNetConfig] = None) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("The current MDPNet-V2 head is binary; out_channels must be 1")
        self.config = config or MDPNetConfig()
        self.backbone = ResNet34Encoder(pretrained=self.config.pretrained_backbone)

        focus_class = RefinedDPFocus if self.config.refine_dpfocus else LegacyDPFocus
        focus_kwargs = dict(
            use_sk_lmap=self.config.use_sk_lmap,
            use_position_attention=self.config.use_position_attention,
            max_position_tokens=self.config.max_position_tokens,
        )
        if not self.config.refine_dpfocus:
            focus_kwargs["apply_sigmoid"] = self.config.legacy_dpfocus_sigmoid
        self.dpfocus_4 = focus_class(512, 256, **focus_kwargs)
        self.dpfocus_3 = focus_class(272, 128, **focus_kwargs)
        self.dpfocus_2 = focus_class(136, 64, **focus_kwargs)
        self.dpfocus_1 = focus_class(68, 32, **focus_kwargs)

        if self.config.dynamic_fusion:
            self.fusion = DynamicScaleFusion()
        else:
            self.fusion = LegacyFusion()

    @staticmethod
    def _up_to(x: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x: Tensor, return_aux: Optional[bool] = None):
        layer_1, layer_2, layer_3, layer_4 = self.backbone(x)

        d4 = self.dpfocus_4(layer_4)
        d4_up = self._up_to(d4, layer_3)
        d3 = self.dpfocus_3(torch.cat((layer_3, d4_up), dim=1))
        d3_up = self._up_to(d3, layer_2)
        d2 = self.dpfocus_2(torch.cat((layer_2, d3_up), dim=1))
        d2_up = self._up_to(d2, layer_1)
        d1 = self.dpfocus_1(torch.cat((layer_1, d2_up), dim=1))

        output_size = x.shape[-2:]
        features = [
            F.interpolate(d1, size=output_size, mode="bilinear", align_corners=True),
            F.interpolate(d2, size=output_size, mode="bilinear", align_corners=True),
            F.interpolate(d3, size=output_size, mode="bilinear", align_corners=True),
            F.interpolate(d4, size=output_size, mode="bilinear", align_corners=True),
        ]
        logits, auxiliary = self.fusion(features)

        use_aux = self.config.deep_supervision if return_aux is None else return_aux
        if use_aux:
            return {"logits": logits, "auxiliary": auxiliary}
        return logits


def build_model(**kwargs) -> MDPNetV2:
    return MDPNetV2(config=MDPNetConfig(**kwargs))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


if __name__ == "__main__":
    model = MDPNetV2(config=MDPNetConfig(pretrained_backbone=False, max_position_tokens=32))
    sample = torch.randn(1, 3, 128, 128)
    output = model(sample)
    print(f"parameters={count_parameters(model):,}")
    print(f"input={tuple(sample.shape)} output={tuple(output.shape)}")
