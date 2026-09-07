"""Build a bounded, CPU-only handoff; never reads NWB files or starts training.

Only export trusted local checkpoints: torch.load(weights_only=False) can
execute pickle code. Recipients only need NPZ (allow_pickle=False) and JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "dandi688-state-npz-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def export_state(state, path, provenance):
    arrays, entries = {}, []
    for i, (key, tensor) in enumerate(sorted(state.items())):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"Unsupported state entry: {key}")
        t = tensor.detach().cpu().contiguous()
        name = f"tensor_{i:05d}"
        a = t.view(torch.uint16).numpy() if t.dtype == torch.bfloat16 else t.numpy()
        arrays[name] = a
        entries.append(dict(name=name, key=key, shape=list(t.shape),
                            torch_dtype=str(t.dtype), numpy_dtype=a.dtype.str,
                            sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    meta = dict(schema=SCHEMA, npz_sha256=sha(path), tensors=entries,
                state_only=True, resume_training=False, **provenance)
    dump(path.with_suffix(".json"), meta)
    restored = load_state(path)
    for key, value in state.items():
        original = value.detach().cpu().contiguous()
        recovered = restored[key].contiguous()
        # Byte comparison, including signed zero and any NaN payload.
        assert original.dtype == recovered.dtype and original.shape == recovered.shape
        assert torch.equal(original.reshape(-1).view(torch.uint8),
                           recovered.reshape(-1).view(torch.uint8)), key
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                tensors=len(entries), roundtrip="BITWISE_PASS")


def load_state(path):
    """Return a CPU state_dict; caller constructs the exact class and loads strict=True."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["schema"] != SCHEMA or sha(path) != meta["npz_sha256"]:
        raise ValueError(f"Schema/hash mismatch: {path}")
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {entry["name"] for entry in meta["tensors"]}:
            raise ValueError("Unexpected NPZ keys")
        for entry in meta["tensors"]:
            a = archive[entry["name"]]
            if (list(a.shape) != entry["shape"] or a.dtype.str != entry["numpy_dtype"]
                    or hashlib.sha256(a.tobytes()).hexdigest() != entry["sha256"]):
                raise ValueError(f"Tensor mismatch: {entry['key']}")
            t = torch.from_numpy(a.copy())
            if entry["torch_dtype"] == "torch.bfloat16":
                t = t.view(torch.bfloat16)
            if str(t.dtype) != entry["torch_dtype"] or entry["key"] in result:
                raise ValueError("Dtype or duplicate-key mismatch")
            result[entry["key"]] = t
    return result


def source_paths():
    paths = set()
    # Source-only snapshot, not arbitrary caches/checkpoints or a dirty-worktree commit.
    for folder in ("btransform_unified_v1/src", "streaming_calibration_exp/src",
                   "SPINT-main/src", "sua_exploration/mc_maze",
                   "tfpd_exploration/src/m2_dual_track_v1",
                   "tfpd_exploration/src/m2_b_small_stability_v1"):
        paths.update(p for p in (ROOT / folder).rglob("*.py") if p.is_file())
    for pattern in ("sua_exploration/scripts/*dandi688*.py",
                    "sua_exploration/docs/*688*.md",
                    "btransform_unified_v1/docs/WORKORDER_EXP*_V2_20260907.md",
                    "btransform_unified_v1/docs/HANDOFF_DANDI688*.md",
                    "btransform_unified_v1/scripts/*688_handoff*.py"):
        paths.update(ROOT.glob(pattern))
    for name in ("sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json",
                 "sua_exploration/docs/FP32_T4_MAINLINE_PROTOCOL.md",
                 "sua_exploration/docs/CURRENT_RESULTS.md"):
        paths.add(ROOT / name)
    for arm in ("b0", "t4"):
        base = f"sua_spint_t4_mainline_fp32_v1_{arm}_dandi688_co_s42"
        paths.add(ROOT / f"sua_exploration/checkpoints/{base}/run_metadata.json")
        paths.add(ROOT / f"sua_exploration/results/p3_{base}_seed42.json")
    return sorted(paths)


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    assets = []
    for arm in ("b0", "t4"):
        relative = (f"sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_{arm}"
                    "_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt")
        source = ROOT / relative
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        provenance = dict(source_path=relative, source_sha256=sha(source),
                          source_epoch=ckpt.get("epoch"),
                          source_global_step=ckpt.get("global_step"),
                          hyper_parameters=ckpt.get("hyper_parameters", {}))
        assets.append(export_state(ckpt["state_dict"], out / f"{arm}_s42_e11_full.npz", provenance))
        if arm == "b0":
            prefix = "student.id_encoder."
            encoder = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items()
                       if k.startswith(prefix)}
            if len(encoder) != 12 or tuple(encoder["fc_id_out.4.weight"].shape) != (50, 512):
                raise ValueError("Unexpected B0 donor architecture")
            assets.append(export_state(encoder, out / "b0_s42_e11_encoder.npz",
                                       dict(provenance, stripped_prefix=prefix, e0_dim=50)))
    records = []
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():
            rel = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            archive.writestr(rel, payload)
            records.append(dict(path=rel, bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest()))
    dump(out / "source_manifest.json", records)
    manifest = dict(schema=SCHEMA, assets=assets,
                    source_snapshot_sha256=sha(out / "source_snapshot.zip"),
                    source_files=len(records),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    working_tree_snapshot=True, torch_version=torch.__version__,
                    numpy_version=np.__version__,
                    exclusions=["NWB/data", "dense caches", "optimizer", "scheduler", "RNG", "EMA resume state"],
                    teacher_dependency="PRESENT; embedded teacher weights included in full NPZ; roster audit still required",
                    formal_training_runner="NOT_IMPLEMENTED_IN_THIS_HANDOFF")
    dump(out / "bundle_manifest.json", manifest)
    with (out / "SHA256SUMS").open("x") as stream:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(f"{sha(path)}  {path.name}\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--trusted-local-checkpoints", action="store_true", required=True)
    check = sub.add_parser("verify")
    check.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output.resolve())
    else:
        for line in (args.bundle / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            if Path(name).name != name or sha(args.bundle / name) != expected:
                raise ValueError(f"Bundle mismatch: {name}")
        for path in sorted(args.bundle.glob("*.npz")):
            print(path.name, len(load_state(path)), "PASS")


if __name__ == "__main__":
    main()
