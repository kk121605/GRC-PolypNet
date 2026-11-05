# GRC-PolypNet

PyTorch research code for investigating residual context and scale selection in polyp segmentation.

## Research overview

This project examines whether adding spatial-channel relations and local-scale processing improves polyp segmentation under controlled comparisons.

The study distinguishes three questions:

1. Does an architectural addition improve performance after independent retraining?
2. Does a fitted model depend on the added computation?
3. How sensitive is the fitted model to image-specific scale-selection weights?

These questions require separate experiments. Removing a component from a fitted checkpoint does not measure the same effect as retraining a model without that component.

## Model configurations

The public interface exposes four configurations:

| Configuration | Description |
|---|---|
| `control` | Shared segmentation pipeline |
| `da` | Spatial-channel relation residual |
| `sk` | Local-scale residual |
| `joint` | Both residual paths |

The architecture uses a ResNet-34 encoder and an MDPNet-based segmentation pipeline. Separate bounded gates control the added residual paths.

## Code structure

| File | Purpose |
|---|---|
| `grc_polypnet.py` | Public model interface and reviewed-checkpoint loading |
| `r4_model.py` | Study model and residual modules |
| `residual_ops.py` | Spatial-channel and local-scale residual operations |
| `backbone.py` | ResNet-34 encoder |
| `model.py` | Shared segmentation architecture |
| `modules.py` | Shared network components |
| `checkpoint_manifest.json` | Historical checkpoint identities and SHA-256 hashes |
| `verify_equivalence.py` | Historical equivalence-audit script |
| `QA/equivalence_report.json` | Archived CPU regression report |

## Installation

Use Python 3.11 or later and install the dependencies:

```bash
pip install -r requirements.txt
```

## Model interface example

Run this example from the repository directory:

```python
import torch
from grc_polypnet import GRCPolypNet

model = GRCPolypNet(arm="joint", seed=42).eval()
image = torch.rand(1, 3, 256, 256)

with torch.inference_mode():
    logits = model(image)
    mask = logits.sigmoid() >= 0.5

print(logits.shape)
```

Expected output:

```text
torch.Size([1, 1, 256, 256])
```

This example uses random initialization and a synthetic input to illustrate the interface. It does not produce a trained segmentation model.

The study input format is RGB, resized to 256 × 256 and scaled to [0, 1], without ImageNet mean/std normalization.

## Release scope

This release contains the core model implementation and historical provenance records.

Datasets, trained checkpoints, the complete manuscript, and the full training and evaluation pipeline are not included.

The historical `verify_equivalence.py` script requires additional files from the original research workspace. It cannot run independently from this repository. Its archived report documents the recorded audit scope; it is not a new reproduction of the reported segmentation results.

## Research boundaries

The study concerns the evaluated model family and protocols. It does not establish a general advantage of adaptive context or clinical effectiveness.

## Acknowledgements and upstream code

The segmentation foundation follows MDPNet:

A. A. Kamara et al., “MDPNet: Multiscale Dynamic Polyp-Focus Network for Enhancing Medical Image Polyp Segmentation,” IEEE Transactions on Medical Imaging, 2025.

DOI: https://doi.org/10.1109/TMI.2025.3588503

The residual designs draw on Dual Attention Network and Selective Kernel Networks, both published at CVPR 2019.

The upstream MIT copyright and permission notice is preserved in `UPSTREAM_LICENSE`. Attribution distinguishes the upstream architecture from the residual additions studied in this project.