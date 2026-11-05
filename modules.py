"""Attention, dynamic-focus and multiscale-fusion modules for MDPNet-V2."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: bool = True,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class LegacyConv1x1(nn.Module):
    """The two-layer 1x1 transform used in the released MDPNet code."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LegacyLMAP(nn.Module):
    """Robust form of the released LMAP operation.

    Adaptive pooling matches fixed strides 16/8/4/2 for the 256x256 MDPNet
    pathway while also accepting other spatial sizes.
    """

    def __init__(self, channels: int, pool_strides: Sequence[int] = (16, 8, 4, 2)) -> None:
        super().__init__()
        self.pool_strides = tuple(pool_strides)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.conv5(self.conv3(x))

    def forward(self, x: Tensor) -> Tensor:
        feature = self.forward_features(x)
        _, _, height, width = feature.shape
        pooled = []
        for stride in self.pool_strides:
            size = (max(1, height // stride), max(1, width // stride))
            branch = F.adaptive_avg_pool2d(feature, size)
            pooled.append(branch.flatten(2))
        pooled.append(self.alpha * feature.flatten(2))
        return torch.cat(pooled, dim=2)


class SelectiveKernelLMAP(nn.Module):
    """Original LMAP plus an SK-Net-style residual enhancement.

    Split uses parallel dilated 3x3 group convolutions. Fuse summarizes their
    sum. Select normalizes channel-wise branch weights along the branch axis.
    The selected feature is added to LMAP's original two-convolution feature and
    the result then follows the original multiscale pooling path. ``sk_scale``
    starts at zero, so the module initially behaves exactly like LegacyLMAP and
    learns to use the extra receptive fields during optimization.
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 3),
        pool_strides: Sequence[int] = (16, 8, 4, 2),
        reduction: int = 4,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(dilations) < 2:
            raise ValueError("Selective-kernel LMAP needs at least two branches")

        self.channels = channels
        self.pool_strides = tuple(pool_strides)
        groups = math.gcd(channels, max_groups)
        hidden_channels = max(channels // reduction, 4)

        # Preserve the released LMAP transformation as the main path. The SK
        # branch only learns a residual correction, so it cannot erase the
        # original representation.
        self.base = LegacyLMAP(channels)

        self.branches = nn.ModuleList(
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                dilation=dilation,
                groups=groups,
            )
            for dilation in dilations
        )
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.selectors = nn.ModuleList(
            nn.Conv2d(hidden_channels, channels, kernel_size=1) for _ in dilations
        )
        self.sk_scale = nn.Parameter(torch.zeros(1))

    def forward_features(self, x: Tensor) -> Tensor:
        base_feature = self.base.forward_features(x)
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        descriptor = self.fuse(branch_features.sum(dim=1))
        logits = torch.stack([selector(descriptor) for selector in self.selectors], dim=1)
        weights = F.softmax(logits, dim=1)
        sk_feature = (branch_features * weights).sum(dim=1)
        return base_feature + self.sk_scale * sk_feature

    def forward(self, x: Tensor) -> Tensor:
        feature = self.forward_features(x)
        _, _, height, width = feature.shape
        pooled = []
        for stride in self.pool_strides:
            size = (max(1, height // stride), max(1, width // stride))
            pooled.append(F.adaptive_avg_pool2d(feature, size).flatten(2))
        pooled.append(self.base.alpha * feature.flatten(2))
        return torch.cat(pooled, dim=2)


class DualDomainNMAP(nn.Module):
    """NMAP channel attention plus memory-bounded DA-Net position attention.

    Every spatial position remains a query. Keys and values are adaptively
    pooled when needed, so the spatial affinity is N x M rather than N x N.
    This is important at MDPNet's shallow 128x128 feature stage.
    """

    def __init__(
        self,
        channels: int,
        use_sk_lmap: bool = True,
        use_position_attention: bool = True,
        max_position_tokens: int = 64,
    ) -> None:
        super().__init__()
        if max_position_tokens < 1:
            raise ValueError("max_position_tokens must be positive")

        lmap_type = SelectiveKernelLMAP if use_sk_lmap else LegacyLMAP
        self.lmap = lmap_type(channels)
        self.use_position_attention = use_position_attention
        self.max_position_tokens = max_position_tokens

        if use_position_attention:
            query_channels = max(1, channels // 8)
            self.query = nn.Conv2d(channels, query_channels, kernel_size=1, bias=False)
            self.key = nn.Conv2d(channels, query_channels, kernel_size=1, bias=False)
            self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            # Zero initialization makes the position branch an identity
            # addition before training, preserving the channel-only NMAP.
            self.position_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _pooled_size(height: int, width: int, max_tokens: int) -> tuple[int, int]:
        if height * width <= max_tokens:
            return height, width
        aspect = height / width
        pooled_h = min(height, max(1, int(round(math.sqrt(max_tokens * aspect)))))
        pooled_w = min(width, max(1, max_tokens // pooled_h))
        while pooled_h * pooled_w > max_tokens:
            if pooled_w >= pooled_h and pooled_w > 1:
                pooled_w -= 1
            elif pooled_h > 1:
                pooled_h -= 1
            else:
                break
        return pooled_h, pooled_w

    def _channel_context(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        multiscale = self.lmap(x)
        energy = torch.bmm(multiscale, multiscale.transpose(1, 2))
        energy = energy.amax(dim=-1, keepdim=True) - energy
        attention = F.softmax(energy, dim=-1)
        value = x.flatten(2)
        return torch.bmm(attention, value).reshape(batch, channels, height, width)

    def _position_context(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        pooled_size = self._pooled_size(height, width, self.max_position_tokens)
        context_source = F.adaptive_avg_pool2d(x, pooled_size)

        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(context_source).flatten(2)
        energy = torch.bmm(query, key) / math.sqrt(query.shape[-1])
        attention = F.softmax(energy, dim=-1)

        value = self.value(context_source).flatten(2)
        context = torch.bmm(value, attention.transpose(1, 2))
        return context.reshape(batch, channels, height, width)

    def forward(self, x: Tensor) -> Tensor:
        out = x + self._channel_context(x)
        if self.use_position_attention:
            out = out + self.position_scale * self._position_context(x)
        return out


class LegacyDPFocus(nn.Module):
    """Released DPFocus topology with configurable NMAP internals."""

    def __init__(
        self,
        in_channels: int,
        nominal_out_channels: int,
        use_sk_lmap: bool,
        use_position_attention: bool,
        max_position_tokens: int,
        dropout_rate: float = 0.1,
        apply_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        reduced_channels = nominal_out_channels // 16
        if reduced_channels < 1:
            raise ValueError("nominal_out_channels must be at least 16")
        self.out_channels = reduced_channels

        self.conv1 = ConvBNAct(in_channels, in_channels, bias=True)
        self.conv_s = ConvBNAct(in_channels, reduced_channels, bias=True)
        self.conv_r = LegacyConv1x1(in_channels, reduced_channels)
        self.conv_t = LegacyConv1x1(in_channels, reduced_channels)
        self.context_to_input = LegacyConv1x1(reduced_channels, in_channels)
        self.nmap = DualDomainNMAP(
            reduced_channels,
            use_sk_lmap=use_sk_lmap,
            use_position_attention=use_position_attention,
            max_position_tokens=max_position_tokens,
        )
        self.output = LegacyConv1x1(in_channels, reduced_channels)
        self.apply_sigmoid = apply_sigmoid
        self.dynamic_scaling = nn.Parameter(torch.tensor(1.0))
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.non_linearity = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        r = self.conv_r(self.conv1(x)).flatten(2)
        s = self.conv_s(x).flatten(2)
        t = self.conv_t(x).flatten(2)

        normalized = self.dynamic_scaling * F.softmax(r + s, dim=-1)
        target_weights = F.softmax(t, dim=-1).transpose(1, 2)
        polyp_target = self.attn_dropout(self.non_linearity(torch.bmm(normalized, target_weights)))

        attended = self.nmap(normalized.reshape(batch, self.out_channels, height, width))
        attended_flat = attended.flatten(2)
        channel_summary = attended_flat.mean(dim=2, keepdim=True)
        enhanced_target = polyp_target + torch.bmm(polyp_target, channel_summary)

        affinity = torch.sigmoid(torch.bmm(enhanced_target.transpose(1, 2), attended_flat))
        focused = torch.bmm(enhanced_target, affinity).reshape(
            batch, self.out_channels, height, width
        )
        fused = self.context_to_input(focused) + x * self.dynamic_scaling
        output = self.output(fused)
        return torch.sigmoid(output) if self.apply_sigmoid else output


class RefinedDPFocus(nn.Module):
    """Released DPFocus plus a zero-initialized adaptive refinement branch."""

    def __init__(
        self,
        in_channels: int,
        nominal_out_channels: int,
        use_sk_lmap: bool,
        use_position_attention: bool,
        max_position_tokens: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.base = LegacyDPFocus(
            in_channels,
            nominal_out_channels,
            use_sk_lmap=use_sk_lmap,
            use_position_attention=use_position_attention,
            max_position_tokens=max_position_tokens,
            dropout_rate=dropout_rate,
            apply_sigmoid=False,
        )
        reduced_channels = self.base.out_channels
        self.out_channels = reduced_channels

        # The parallel branch improves boundary/local evidence in the compact
        # DPFocus feature and uses global channel context to suppress noise.
        groups = math.gcd(reduced_channels, 8)
        self.local_refine = nn.Sequential(
            ConvBNAct(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                groups=groups,
            ),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1, bias=False),
        )
        hidden = max(4, reduced_channels // 2)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(reduced_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, reduced_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        base_feature = self.base(x)
        residual = self.local_refine(base_feature) * self.channel_gate(base_feature)
        return base_feature + self.refine_scale * residual


class LegacyFusion(nn.Module):
    """Original concatenate-and-1x1 MDPNet output fusion."""

    def __init__(self, stage_channels: Sequence[int] = (2, 4, 8, 16)) -> None:
        super().__init__()
        if sum(stage_channels) != 30 or stage_channels[-1] != 16:
            raise ValueError("Legacy fusion expects MDPNet stage channels (2, 4, 8, 16)")
        self.reduce = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
        )

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, list[Tensor]]:
        merged = torch.cat(features, dim=1)
        fused = self.reduce(merged) + features[-1]
        return self.head(fused), []


class DynamicScaleFusion(nn.Module):
    """Original fusion plus a zero-initialized adaptive residual branch."""

    def __init__(
        self,
        stage_channels: Sequence[int] = (2, 4, 8, 16),
        fusion_channels: int = 16,
    ) -> None:
        super().__init__()
        self.base = LegacyFusion(stage_channels)
        self.projections = nn.ModuleList(
            ConvBNAct(channels, fusion_channels, kernel_size=1) for channels in stage_channels
        )
        self.scale_scorers = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(fusion_channels, 1, kernel_size=1),
            )
            for _ in stage_channels
        )
        self.scale_prior = nn.Parameter(torch.zeros(len(stage_channels)))
        self.concat_refine = ConvBNAct(
            len(stage_channels) * fusion_channels,
            fusion_channels,
            kernel_size=1,
        )
        self.refine = ConvBNAct(fusion_channels, fusion_channels, kernel_size=3)
        self.head = nn.Conv2d(fusion_channels, 1, kernel_size=1)
        self.dynamic_scale = nn.Parameter(torch.zeros(1))
        self.aux_heads = nn.ModuleList(
            nn.Conv2d(channels, 1, kernel_size=1) for channels in stage_channels
        )

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, list[Tensor]]:
        base_logits, _ = self.base(features)
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        score_list = [scorer(feature) for scorer, feature in zip(self.scale_scorers, projected)]
        scores = torch.stack(score_list, dim=1)
        scores = scores + self.scale_prior.view(1, -1, 1, 1, 1)
        weights = F.softmax(scores, dim=1)

        stacked = torch.stack(projected, dim=1)
        weighted = (stacked * weights).sum(dim=1)
        complementary = self.concat_refine(torch.cat(projected, dim=1))
        residual_logits = self.head(self.refine(weighted + complementary))
        logits = base_logits + self.dynamic_scale * residual_logits
        auxiliary = [head(feature) for head, feature in zip(self.aux_heads, features)]
        return logits, auxiliary
