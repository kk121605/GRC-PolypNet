"""CPU regression against archived code; not a new segmentation benchmark."""

import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from grc_polypnet import GRCPolypNet, STUDY_ARMS
from crcm_net import CRCMNet


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
ARCHIVE = ROOT.parent / "原正式实现/MDPNet_V2"
WEIGHTS = WORK / "04_实验依据与权重/04_模型权重"
QA = ROOT / "QA"
REPORT = {}


def save_progress():
    (QA / "equivalence_report.json").write_text(
        json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(a, b, context):
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(),
            f"Nonfinite values: {context}")
    delta = float((a - b).abs().max())
    require(torch.equal(a, b), f"Tensor difference in {context}: {delta}")
    return delta


def main():
    global REPORT
    started = time.time()
    QA.mkdir(exist_ok=True)
    report = {"status": "RUNNING", "runtime": {"python": sys.version,
              "torch": torch.__version__, "platform": platform.platform(),
              "device": "cpu", "dtype": "float32", "threads": 4},
              "scope": "Code equivalence only; no retraining or metric reproduction",
              "copied_sources": [], "initialization": [], "checkpoints": [],
              "nonzero_gate_training": []}
    REPORT = report
    require(CRCMNet is GRCPolypNet, "Historical import must alias the current model")
    report["current_model"] = "GRC-PolypNet"
    report["compatibility_alias_identical"] = True
    report_path = QA / "equivalence_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for name in ("backbone.py", "model.py", "modules.py"):
        original, release = ARCHIVE / name, ROOT / name
        require(sha256(original) == sha256(release), f"Changed shared source: {name}")
        report["copied_sources"].append({"file": name, "sha256": sha256(original)})

    source = ARCHIVE / "mdpnet_r4/r4_model.py"
    spec = importlib.util.spec_from_file_location("archived_r4", source)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    report["original_r4_sha256"] = sha256(source)
    report["release_sources"] = {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))}

    records = []
    for dataset, arms in (("Kvasir", tuple(STUDY_ARMS)), ("ClinicDB", ("control", "joint"))):
        for seed in (42, 43, 44):
            for arm in arms:
                variant = STUDY_ARMS[arm]
                path = WEIGHTS / "原正式" / f"{dataset}_seed{seed}_{variant}_best.pt"
                records.append({"path": path.relative_to(WORK).as_posix(),
                                "dataset": dataset, "seed": seed, "variant": variant,
                                "sha256": sha256(path)})
    for arm in ("control", "joint"):
        variant = STUDY_ARMS[arm]
        path = WEIGHTS / "新增" / f"原正式追加seed45_seed45_{variant}_best.pt"
        records.append({"path": path.relative_to(WORK).as_posix(), "dataset": "Kvasir",
                        "seed": 45, "variant": variant, "sha256": sha256(path)})
    manifest_path = ROOT / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest == {"campaign": "original-matrix-plus-seed45", "checkpoints": records},
            "Files differ from frozen campaign manifest; do not regenerate the whitelist")
    report["manifest_sha256"] = sha256(manifest_path)
    locks_root = WORK / "04_实验依据与权重/02_逐图指标与锁"
    lock_paths = {
        "Kvasir": locks_root / "原正式/kvasir_supervised_test/manifest.json",
        "ClinicDB": locks_root / "原正式/clinicdb_supervised_test/manifest.json",
        "seed45": locks_root / "新增/r4_test_20260914_051511_362263/manifest.json",
    }
    report["historical_locks"] = {key: {"path": path.relative_to(WORK).as_posix(),
        "sha256": sha256(path)} for key, path in lock_paths.items()}
    for record in records:
        lock_key = "seed45" if record["seed"] == 45 else record["dataset"]
        locked = json.loads(lock_paths[lock_key].read_text(encoding="utf-8"))["source_files_sha256"]
        matches = [name for name, digest in locked.items()
                   if digest == record["sha256"] and name.endswith(f"/{record['variant']}/best.pt")
                   and f"seed{record['seed']}" in name]
        require(len(matches) == 1, f"Checkpoint not identified by historical test lock: {record['path']}")
    print("Frozen 20-checkpoint manifest verified", flush=True)
    save_progress()

    rng = torch.Generator().manual_seed(20260916)
    synthetic = torch.rand((1, 3, 256, 256), generator=rng)
    with zipfile.ZipFile(WORK / "03_数据集/Kvasir.zip") as archive:
        names = sorted(n for n in archive.namelist()
                       if "/images/" in "/" + n.replace("\\", "/").lower()
                       and n.lower().endswith((".jpg", ".png", ".jpeg")))
        require(bool(names), "No RGB image in source ZIP")
        raw = archive.read(names[0])
        rgb = Image.open(io.BytesIO(raw)).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        real_image = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        report["real_image"] = {"zip_member": names[0], "sha256": hashlib.sha256(raw).hexdigest(),
                                "role": "One code-regression input; no label/metric evaluation"}

    for arm, variant in STUDY_ARMS.items():
        torch.manual_seed(91)
        old = old_module.MDPNetR4(variant, seed=42)
        torch.manual_seed(91)
        new = GRCPolypNet(arm, seed=42)
        before_state, after_state = old.state_dict(), new.state_dict()
        require(list(before_state) == list(after_state), "State key/order mismatch")
        for key, value in before_state.items():
            compare(value, after_state[key], f"initial {arm}:{key}")
        report["initialization"].append({"arm": arm, "all_state_tensors_equal": True})
        print(f"Initialization PASS: {arm}", flush=True)
        save_progress()
        del old, new, before_state, after_state
        gc.collect()

    for index, record in enumerate(records, 1):
        path = WORK / record["path"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        old = old_module.MDPNetR4(record["variant"], seed=record["seed"]).eval()
        old.load_state_dict(checkpoint["model"], strict=True)
        new = GRCPolypNet.from_checkpoint(path)
        before_state, after_state = old.state_dict(), new.state_dict()
        require(list(before_state) == list(after_state), "Loaded state keys differ")
        for key, value in before_state.items():
            compare(value, after_state[key], f"loaded {record['path']}:{key}")
        parameters = sum(p.numel() for p in old.parameters())
        require(parameters == sum(p.numel() for p in new.parameters()), "Parameter count differs")
        differences = {}
        test_images = [("synthetic_64", F.interpolate(synthetic, size=(64, 64), mode="bilinear", align_corners=False)),
                       ("real_64", F.interpolate(real_image, size=(64, 64), mode="bilinear", align_corners=False))]
        if record["dataset"] == "Kvasir" and record["seed"] == 42 and new.study_arm in ("control", "joint"):
            test_images.extend((("synthetic_256", synthetic), ("real_256", real_image)))
        with torch.inference_mode():
            for label, image in test_images:
                before, after = old(image), new(image)
                require(before.shape == (1, 1, *image.shape[-2:]), "Unexpected output shape")
                differences[label] = compare(before, after, record["path"] + label)
                require(torch.equal(before.sigmoid() >= .5, after.sigmoid() >= .5), "Masks differ")
        report["checkpoints"].append({**record, "parameters": parameters,
                                      "nonzero_gate_count": sum(bool(v.detach().ne(0).any()) for v in new.gates().values()),
                                      "max_abs_output_differences": differences, "masks_equal": True})
        print(f"Checkpoint {index}/20 PASS: {record['dataset']} {record['seed']} {record['variant']}", flush=True)
        save_progress()
        del old, new, checkpoint, before_state, after_state
        gc.collect()

    small_input = torch.rand((2, 3, 64, 64), generator=rng)
    target = torch.rand((2, 1, 64, 64), generator=rng)
    for arm, variant in STUDY_ARMS.items():
        torch.manual_seed(117)
        old = old_module.MDPNetR4(variant, seed=43)
        with torch.no_grad():
            for index, gate in enumerate(old.gates().values()):
                gate.fill_(0.7 if index % 2 else -0.4)
        new = GRCPolypNet(arm, seed=43)
        new.load_state_dict(old.state_dict(), strict=True)
        old.train()
        new.train()
        outputs = []
        inputs = []
        for model in (old, new):
            x = small_input.clone().requires_grad_(True)
            torch.manual_seed(891)
            output = model(x)
            F.binary_cross_entropy_with_logits(output, target).backward()
            outputs.append(output.detach())
            inputs.append(x.grad)
        compare(outputs[0], outputs[1], f"training output:{arm}")
        compare(inputs[0], inputs[1], f"training input gradient:{arm}")
        gradients = 0
        for (key, before), (other_key, after) in zip(old.named_parameters(), new.named_parameters()):
            require(key == other_key, "Parameter ordering differs")
            require((before.grad is None) == (after.grad is None), f"Gradient presence differs:{key}")
            if before.grad is not None:
                compare(before.grad, after.grad, f"parameter gradient:{key}")
                gradients += 1
        after_state = new.state_dict()
        for key, value in old.state_dict().items():
            compare(value, after_state[key], f"post-forward buffer:{key}")
        report["nonzero_gate_training"].append({"arm": arm, "output_equal": True,
            "input_gradient_equal": True, "parameter_gradients_equal": gradients,
            "buffers_equal": True, "max_abs_difference": 0.0})
        print(f"Nonzero-gate training/backward PASS: {arm}", flush=True)
        save_progress()
        del old, new, outputs, inputs, after_state
        gc.collect()

    rejected = []
    for name in ("DA_v1_seed42_public_nmap_best.pt", "LMAP_v2_seed42_public_lmap_selector_best.pt"):
        try:
            GRCPolypNet.from_checkpoint(WEIGHTS / "新增" / name)
        except ValueError as error:
            require("not in the reviewed" in str(error), "Unexpected rejection reason")
            rejected.append(name)
        else:
            raise AssertionError("Accepted incompatible campaign")
    try:
        GRCPolypNet("unknown")
    except ValueError:
        report["invalid_arm_rejected"] = True
    else:
        raise AssertionError("Accepted invalid arm")
    report.update(status="PASS", rejected_other_campaigns=rejected,
                  elapsed_seconds=round(time.time() - started, 2))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checkpoints": len(records),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        REPORT.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        if QA.is_dir():
            save_progress()
        raise
