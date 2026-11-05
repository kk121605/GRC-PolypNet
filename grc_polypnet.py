"""GRC-PolypNet study interface; historical checkpoint keys remain unchanged.

Architecture foundation: Kamara et al., MDPNet, TMI 2025,
doi:10.1109/TMI.2025.3588503. Residual ideas: DANet and SKNet, CVPR 2019.
Only the original four-arm matrix is exposed here, not DA v1 or LMAP v2.
"""

import hashlib
import io
import json
from pathlib import Path

import torch

from r4_model import MDPNetR4


STUDY_ARMS = {
    "control": "public_control",
    "da": "public_nmap",
    "sk": "public_lmap",
    "joint": "public_nmap_lmap",
}


class GRCPolypNet(MDPNetR4):
    """Named study model with identical original-matrix tensor registration."""

    def __init__(self, arm="joint", seed=42):
        if arm not in STUDY_ARMS:
            raise ValueError(f"arm must be one of {tuple(STUDY_ARMS)}; got {arm!r}")
        super().__init__(STUDY_ARMS[arm], seed=seed)
        self.study_arm = arm

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        """Load only a hash-listed original-matrix checkpoint on CPU.

        DA v1 reuses some original variant labels, so tensor compatibility alone
        is insufficient. The shipped manifest identifies the 20 reviewed files.
        New experiments must use explicit construction and strict state loading.
        """
        manifest_path = Path(__file__).with_name("checkpoint_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        matches = [row for row in manifest["checkpoints"] if row["sha256"] == digest]
        if len(matches) != 1:
            raise ValueError("Checkpoint is not in the reviewed original-matrix manifest")
        record = matches[0]
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        variant = checkpoint.get("variant")
        reverse = {value: key for key, value in STUDY_ARMS.items()}
        if variant not in reverse or "model" not in checkpoint or "seed" not in checkpoint:
            raise ValueError("Expected an original-matrix checkpoint with variant, seed and model")
        if variant != record["variant"] or checkpoint["seed"] != record["seed"]:
            raise ValueError("Checkpoint metadata differs from the reviewed manifest")
        model = cls(reverse[variant], seed=int(checkpoint["seed"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model
