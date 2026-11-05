"""Behavior-preserving decomposition of the archived residual implementation.

Source: ../原正式实现/MDPNet_V2/mdpnet_r4/r4_model.py.
This release refactor changes no learned tensor names or numerical defaults.
"""

import math

import torch
import torch.nn.functional as F


def local_scale_residual(module, feature):
    transformed = module.forward_features(feature)
    normalized = module.sk_norm(transformed)
    branches = torch.stack([branch(normalized) for branch in module.sk_branches], dim=1)
    batch, _, channels, height, width = branches.shape
    weights = module.sk_selector(branches.sum(1)).reshape(batch, 3, channels, 1, 1).softmax(1)
    selected = transformed + (weights * branches).sum(1)
    pooled = [F.interpolate(
        F.adaptive_avg_pool2d(selected, (max(1, height // stride), max(1, width // stride))),
        size=(height, width), mode="bilinear", align_corners=True,
    ) for stride in module.pool_strides]
    pooled.append(module.alpha * selected)
    return module.sk_project(torch.stack(pooled).mean(0))


def spatial_channel_residual(module, feature):
    normalized = module.da_norm(feature)
    batch, channels, height, width = normalized.shape
    pooled = F.adaptive_avg_pool2d(
        normalized, module._pooled_size(height, width, module.max_position_tokens)
    )
    query = module.da_query(normalized).flatten(2).transpose(1, 2)
    key = module.da_key(pooled).flatten(2)
    weights = (torch.bmm(query, key) / math.sqrt(query.shape[-1])).softmax(-1)
    position = torch.bmm(
        module.da_value(pooled).flatten(2), weights.transpose(1, 2)
    ).reshape(batch, channels, height, width)
    flat = normalized.flatten(2)
    energy = torch.bmm(flat, flat.transpose(1, 2)) / flat.shape[-1]
    channel_weights = (energy.amax(-1, keepdim=True) - energy).softmax(-1)
    channel = torch.bmm(channel_weights, flat).reshape(batch, channels, height, width)
    return module.da_project(torch.cat((position, channel), dim=1))
