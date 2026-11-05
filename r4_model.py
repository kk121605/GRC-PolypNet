"""NMAP/LMAP feature-residual candidates on the unmodified r3 public path.

This module imports the audited r3 project from the launcher-provided path.
Old parameter names are retained. Added branches have independent seeded
initialization and bounded zero-initialized gates. These are hypotheses,
not validated performance improvements.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from model import MDPNetConfig, MDPNetV2
from modules import DualDomainNMAP, LegacyDPFocus, LegacyLMAP
from residual_ops import local_scale_residual, spatial_channel_residual


VARIANTS = {"public_control": (False, False), "public_nmap": (True, False),
            "public_lmap": (False, True), "public_nmap_lmap": (True, True)}
GATE_BOUND = 0.1


@contextmanager
def branch_seed(seed):
    with torch.random.fork_rng(devices=[]):
        # Branches are built on CPU; do not reset any CUDA generators.
        torch.random.default_generator.manual_seed(seed)
        yield


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


class FeatureLMAP(LegacyLMAP):
    """Keep original pooled descriptors and expose a local feature residual."""

    def __init__(self, channels, seed):
        super().__init__(channels)
        with branch_seed(seed):
            self.sk_norm = nn.GroupNorm(1, channels)
            self.sk_branches = nn.ModuleList(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False),
                nn.GroupNorm(1, channels), nn.GELU()) for d in (1, 2, 3))
            hidden = max(4, channels)
            self.sk_selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                             nn.GELU(), nn.Conv2d(hidden, channels * 3, 1))
            self.sk_project = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(1, channels))

    def feature_residual(self, feature):
        return local_scale_residual(self, feature)


class FeatureNMAP(DualDomainNMAP):
    """Original NMAP plus feature-domain DA and nested LMAP residuals."""

    def __init__(self, channels, use_da, use_sk, seed):
        super().__init__(channels, use_sk_lmap=False, use_position_attention=False, max_position_tokens=64)
        self.use_da, self.use_sk = use_da, use_sk
        if use_sk:
            self.lmap = FeatureLMAP(channels, seed + 10)
            self.lmap_gate = nn.Parameter(torch.zeros(1))
        if use_da:
            with branch_seed(seed + 20):
                q = max(4, channels // 2)
                self.da_norm = nn.GroupNorm(1, channels)
                self.da_query = nn.Conv2d(channels, q, 1, bias=False)
                self.da_key = nn.Conv2d(channels, q, 1, bias=False)
                self.da_value = nn.Conv2d(channels, channels, 1, bias=False)
                self.da_project = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), nn.GroupNorm(1, channels))
                self.nmap_gate = nn.Parameter(torch.zeros(1))
        self.capture = False
        self.telemetry = {}

    def feature_residual(self, feature):
        residual = torch.zeros_like(feature)
        info = {"feature_rms": rms(feature)} if self.capture else {}
        if self.use_sk:
            branch = self.lmap.feature_residual(feature)
            effective = GATE_BOUND * self.lmap_gate.tanh()
            residual = residual + effective * branch
            if self.capture:
                info.update(lmap_branch_rms=rms(branch), lmap_effective_gate=float(effective.detach()), lmap_added_rms=rms(effective * branch))
        if self.use_da:
            branch = spatial_channel_residual(self, feature)
            effective = GATE_BOUND * self.nmap_gate.tanh()
            residual = residual + effective * branch
            if self.capture:
                info.update(nmap_branch_rms=rms(branch), nmap_effective_gate=float(effective.detach()), nmap_added_rms=rms(effective * branch))
        if self.capture:
            self.telemetry = info
        return residual


class FeatureDPFocus(LegacyDPFocus):
    def __init__(self, in_channels, nominal_out_channels, use_da, use_sk, seed):
        super().__init__(in_channels, nominal_out_channels, False, False, 64, apply_sigmoid=True)
        self.nmap = FeatureNMAP(self.out_channels, use_da, use_sk, seed)

    def forward(self, x):
        b, _, h, w = x.shape
        r = self.conv_r(self.conv1(x)).flatten(2)
        feature = self.conv_s(x)
        s = feature.flatten(2)
        t = self.conv_t(x).flatten(2)
        normalized = self.dynamic_scaling * F.softmax(r + s, dim=-1)
        target_weights = F.softmax(t, dim=-1).transpose(1, 2)
        polyp_target = self.attn_dropout(self.non_linearity(torch.bmm(normalized, target_weights)))
        attended = self.nmap(normalized.reshape(b, self.out_channels, h, w))
        attended_flat = attended.flatten(2)
        channel_summary = attended_flat.mean(dim=2, keepdim=True)
        enhanced_target = polyp_target + torch.bmm(polyp_target, channel_summary)
        affinity = torch.sigmoid(torch.bmm(enhanced_target.transpose(1, 2), attended_flat))
        focused = torch.bmm(enhanced_target, affinity).reshape(b, self.out_channels, h, w)
        fused = self.context_to_input(focused) + x * self.dynamic_scaling
        output = self.output(fused)
        added = self.nmap.feature_residual(feature)
        if self.nmap.capture:
            self.nmap.telemetry.update(normalized_rms=rms(normalized), original_output_rms=rms(output),
                                       added_rms=rms(added), added_to_original_rms=rms(added) / max(rms(output), 1e-12))
        return torch.sigmoid(output + added)


class MDPNetR4(MDPNetV2):
    def __init__(self, variant, seed=42):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(config=MDPNetConfig(pretrained_backbone=False, use_sk_lmap=False,
                        use_position_attention=False, refine_dpfocus=False, dynamic_fusion=False,
                        legacy_dpfocus_sigmoid=True))
        self.variant = variant
        use_da, use_sk = VARIANTS[variant]
        if use_da or use_sk:
            for stage, cin, cout in ((4, 512, 256), (3, 272, 128), (2, 136, 64), (1, 68, 32)):
                setattr(self, f"dpfocus_{stage}", FeatureDPFocus(cin, cout, use_da, use_sk, seed + 1000 * stage))

    def gates(self):
        return {key: value for key, value in self.named_parameters() if key.split(".")[-1] in ("nmap_gate", "lmap_gate")}

    def set_capture(self, enabled):
        for module in self.modules():
            if isinstance(module, FeatureNMAP):
                module.capture = enabled

    def telemetry(self):
        return {key: module.telemetry for key, module in self.named_modules() if isinstance(module, FeatureNMAP)}

    def gate_values(self):
        return {key: {"raw": float(value.detach()), "effective": float(GATE_BOUND * value.detach().tanh())}
                for key, value in self.gates().items()}


def load_common(model, source_state):
    state = model.state_dict()
    for key, value in source_state.items():
        if key not in state or state[key].shape != value.shape or state[key].dtype != value.dtype:
            raise ValueError(f"Shared tensor mismatch: {key}")
    extras = sorted(set(state) - set(source_state))
    allowed = (".nmap.da_", ".nmap.nmap_gate", ".nmap.lmap_gate", ".nmap.lmap.sk_")
    if any(not any(part in key for part in allowed) for key in extras):
        raise ValueError("Unexpected new tensor namespace")
    state.update(source_state)
    model.load_state_dict(state, strict=True)
    restored = model.state_dict()
    if any(not torch.equal(value.cpu(), restored[key].cpu()) for key, value in source_state.items()):
        raise ValueError("Shared tensor content mismatch")
    return {"shared_tensors": len(source_state), "shared_parameters": sum(key in dict(model.named_parameters()) for key in source_state),
            "new_tensors": extras, "all_shared_equal": True}
