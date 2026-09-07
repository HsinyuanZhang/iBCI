#!/usr/bin/env python3
"""CPU-only, score-free H1 H--C carrier reliance ablation for sealed BT-EORT.

The sealed payload is loaded read-only.  Each arm receives a fresh decoder and
only its in-memory ``SessionBank.T`` tensor is replaced.  ``E0`` is explicitly
left intact because it is the fused H--C representation and this experiment
isolates the direct carrier input rather than claiming to remove all carrier
information from the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # workspace root (SPINT)
V1 = ROOT / "btransform_unified_v1"
BT = ROOT / "tfpd_exploration" / "submissions" / "evalai_h1_c2_cal1_b2_ort_v1"
for path in (HERE, ROOT / "scripts" / "rift_v1", ROOT / "src", V1 / "src", V1 / "scripts", ROOT.parent, BT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import PERMUTATION_SEEDS, clone_bank_for_arm, grouped_bootstrap, load_surface

PAYLOAD = BT / "artifacts" / "h1_c2_cal1_b2_s42_ema_e18_L200.pkl"
GRAPHS = BT / "artifacts" / "ort_graphs"
ARMS = (("REAL", "normal", None), ("C_ZERO", "zero", None)) + tuple(
    (f"C_SHUF{seed}", "shuffle", seed) for seed in PERMUTATION_SEEDS
)


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    # sklearn's variance-weighted multioutput R2, written here to leave the
    # scoring calculation transparent and avoid a hidden prediction pre-pass.
    ss_res = np.sum((y - pred) ** 2, axis=0)
    ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    per = 1.0 - ss_res / ss_tot
    return float(np.sum(per * ss_tot) / np.sum(ss_tot))


def _cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def load_reusable_partial(dest: Path, arm: str, group: str, records: list[dict]) -> tuple[dict[str, np.ndarray], dict] | None:
    """Validate a reboot-surviving group shard before incorporating it.

    The shard is accepted only when its exact record keys, fixed endpoints,
    output shapes, and finite FP32 predictions agree with the freshly loaded
    frozen surface.  Thus a resume never silently mixes a different sampling
    surface with the original run.
    """
    path = dest / f"partial_{arm}_{group}.npz"
    if not path.is_file():
        return None
    expected = {r["key"] for r in records}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != expected | {f"coords__{key}" for key in expected}:
            raise RuntimeError(f"invalid reusable shard keys: {path}")
        predictions = {}
        for record in records:
            key = record["key"]
            coords = np.asarray(data[f"coords__{key}"], np.int64)
            pred = np.asarray(data[key])
            if not np.array_equal(coords, record["selected_endpoints"]):
                raise RuntimeError(f"reusable shard coordinate mismatch: {path} {key}")
            if pred.shape != (len(coords), 7) or pred.dtype != np.float32 or not np.isfinite(pred).all():
                raise RuntimeError(f"invalid reusable predictions: {path} {key}")
            predictions[key] = pred.copy()
    return predictions, {"status": "REUSED_VALIDATED_PARTIAL", "partial_file": path.name,
                         "partial_sha256": file_sha(path), "records": sorted(expected),
                         "validation": "exact keys/endpoints, finite float32 shape (n,7)"}


def _decoder(tags: list[str]):
    """Fresh sealed ORT runtime with static caches necessarily rebuilt."""
    os.environ["RT_PACKED_DECODER"] = str(BT / "h1_trf_falcon_decoder.py")
    from falcon_challenge.config import FalconConfig, FalconTask
    from h1_exacte_ort import OrtH1ProjAddFalconDecoder
    dec = OrtH1ProjAddFalconDecoder(FalconConfig(task=FalconTask.h1), str(PAYLOAD),
                                    batch_size=len(tags), graph_dir=GRAPHS,
                                    intra_op=2, inter_op=1)
    # Falcon's hash drops a final underscore component while retaining the
    # sealed record key.
    dec.reset([f"{tag}_cr" for tag in tags])
    return dec


def bind_arm(decoder, records: list[dict], arm: str, seed: int | None) -> dict:
    """Replace only private cloned direct-T tensors; retain E0/masks exactly."""
    original = []
    changed = []
    for bank, record in zip(decoder.local_banks, records):
        source = clone_bank_for_arm(record, arm, seed)
        expected_e0 = np.asarray(source["E0"], np.float32)
        expected_t = np.asarray(record["carrier"], np.float32)
        if not np.array_equal(bank.E0.detach().cpu().numpy(), expected_e0):
            raise RuntimeError(f"sealed E0 differs from M3 source for {record['key']}")
        if not np.array_equal(bank.T.detach().cpu().numpy(), expected_t):
            raise RuntimeError(f"sealed T differs from M3 direct carrier for {record['key']}")
        original.append(array_sha(bank.T.detach().cpu().numpy()))
        # No payload dictionary or original bank tensor is mutated.  The ORT
        # engine is still absent here, so its proj/t4 static inputs are built
        # from this clone on the first predict.
        bank.T = torch.as_tensor(source["carrier"], dtype=torch.float32, device=bank.T.device).clone()
        if not torch.equal(bank.E0, torch.as_tensor(expected_e0, device=bank.E0.device)):
            raise RuntimeError("E0 changed while binding carrier arm")
        changed.append(array_sha(bank.T.detach().cpu().numpy()))
    if decoder._engine is not None:
        raise RuntimeError("arm must be bound before static engine construction")
    return {"original_direct_T_sha256": original, "bound_direct_T_sha256": changed,
            "E0_unchanged": True, "unit_mask_unchanged": True}


def run_group(records: list[dict], arm: str, mode: str, seed: int | None, oracle_limit: int) -> tuple[dict[str, np.ndarray], dict]:
    """Stream each group's raw timelines once; retain only frozen endpoints."""
    tags = [r["key"] for r in records]
    dec = _decoder(tags)
    bank_proof = bind_arm(dec, records, mode, seed)
    selected = [np.asarray(r["selected_endpoints"], np.int64) for r in records]
    lookup = [{int(end): ix for ix, end in enumerate(ends)} for ends in selected]
    pred = [np.empty((len(ends), 7), np.float32) for ends in selected]
    # Use exact eager full forward at actual sampled endpoints, before the ORT
    # call changes the rolling history.  This validates all carrier arms.
    oracle_checks = []; total_steps = max(int(r["neural"].shape[0]) for r in records)
    start = time.monotonic()
    for t in range(total_steps):
        raw = np.zeros((len(records), records[0]["neural"].shape[1]), np.float32)
        for i, record in enumerate(records):
            if t < len(record["neural"]): raw[i] = record["neural"][t]
        need_oracle = any(t in lookup[i] and len(oracle_checks) < oracle_limit for i in range(len(records)))
        expected = None
        if need_oracle:
            # observe is performed by predict; build the exact same post-observe
            # raw history locally, including its left zero padding.
            shadow = np.roll(dec.observation_buffer.copy(), -1, axis=0)
            shadow[-1] = raw
            x = torch.as_tensor(np.ascontiguousarray(shadow[:, :len(records), :].transpose(1, 0, 2)), dtype=torch.float32)
            with torch.inference_mode():
                expected = dec.decoder.forward_last(x, dec.local_banks[0] if len(records) == 1 else __import__('h1_trf_falcon_decoder').stack_banks(dec.local_banks))
            expected = expected.detach().cpu().numpy() / dec.behavior_scaling_factor
        got = dec.predict(raw)
        if not np.isfinite(got).all():
            raise RuntimeError(f"nonfinite {arm} output at raw bin {t}")
        if expected is not None:
            err = np.abs(got - expected); tol = 1e-5 + 1e-5 * np.abs(expected)
            ratio = float(np.max(err / tol))
            if not np.all(err <= tol):
                raise RuntimeError(f"ORT/full oracle mismatch {arm} at raw bin {t}: ratio={ratio}")
            oracle_checks.append({"raw_bin": t, "max_abs_error": float(err.max()), "max_tolerance_ratio": ratio})
        for i, dest in enumerate(lookup):
            j = dest.get(t)
            if j is not None:
                pred[i][j] = got[i]
    if any(len(x) == 0 for x in pred) or len(oracle_checks) < min(oracle_limit, sum(len(x) for x in selected)):
        raise RuntimeError("insufficient actual-endpoint oracle checks")
    if dec._engine is None or not dec._engine.sess_adv or not dec._engine.sess_reb:
        raise RuntimeError("ORT sessions were not exercised")
    return {r["key"]: p for r, p in zip(records, pred)}, {"arm": arm, "mode": mode, "seed": seed,
        "elapsed_s": time.monotonic() - start, "raw_bins_streamed": total_steps,
        "actual_predict_calls": dec._n_predicts, "oracle": {"count": len(oracle_checks), "tolerance": "abs <= 1e-5 + 1e-5*abs(full_fp32)", "checks": oracle_checks},
        "ort": {"version": dec._engine._ort.__version__, "advance_batches": sorted(dec._engine.sess_adv), "rebuild_batches": sorted(dec._engine.sess_reb), "session_init_s": dec._engine.session_init_s},
        "bank_proof": bank_proof}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--max-endpoints-per-group", type=int, default=2048)
    ap.add_argument("--oracle-endpoints-per-group-arm", type=int, default=32)
    ap.add_argument("--resume", action="store_true", help="reuse strictly validated completed group shards in --dest")
    args = ap.parse_args()
    if args.max_endpoints_per_group != 2048 or not 20 <= args.oracle_endpoints_per_group_arm <= 50:
        raise ValueError("protocol requires 2048 group endpoints and 20--50 oracle endpoints/group/arm")
    if args.dest.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.dest}; pass --resume to validate and continue saved shards")
    if not args.dest.exists():
        if args.resume:
            raise FileNotFoundError(f"cannot resume missing directory: {args.dest}")
        args.dest.mkdir(parents=True)
    if args.resume and (args.dest / "report.json").exists():
        raise RuntimeError(f"refusing to resume a completed report: {args.dest / 'report.json'}")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    surface = load_surface(max_endpoints_per_group=args.max_endpoints_per_group)
    groups = {g: [r for r in surface["records"] if r["group"] == g] for g in surface["groups"]}
    if len(groups) != 7 or any(len(v) != 2 for v in groups.values()):
        raise RuntimeError("expected 7 C2 groups each with two recordings")
    # Before inference, capture original immutable payload-bank identities.
    source_hashes = {r["key"]: {"E0": array_sha(r["E0"]), "T": array_sha(r["carrier"]), "unit_mask": array_sha(r["unit_mask"])} for r in surface["records"]}
    payload_sha_before = file_sha(PAYLOAD)
    resumption = {"requested": bool(args.resume), "reused_group_shards": []}
    if args.resume:
        receipt_path = args.dest / "launch_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("resume requires the original launch_receipt.json")
        prior = json.loads(receipt_path.read_text())
        if prior.get("payload_sha256_before") != payload_sha_before:
            raise RuntimeError("payload SHA changed since the interrupted run")
        if prior.get("query_inventory_sha256") != surface["query_inventory_sha256"]:
            raise RuntimeError("query inventory changed since the interrupted run")
        resume_receipt = {"schema": "h1_bt_eort_carrier_reliance_resume_v1", "status": "RUNNING",
                          "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "argv": sys.argv,
                          "original_launch_receipt": receipt_path.name, "payload_sha256": payload_sha_before,
                          "query_inventory_sha256": surface["query_inventory_sha256"]}
        (args.dest / f"resume_receipt_{os.getpid()}.json").write_text(json.dumps(resume_receipt, indent=2, sort_keys=True) + "\n")
    launch = {"schema": "h1_bt_eort_carrier_reliance_launch_v1", "status": "RUNNING",
              "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "argv": sys.argv,
              "payload_sha256_before": payload_sha_before, "query_inventory_sha256": surface["query_inventory_sha256"],
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not args.resume:
        (args.dest / "launch_receipt.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    def heartbeat(**row):
        status = row.pop("status", "RUNNING")
        row.update({"status": status, "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
        (args.dest / "heartbeat.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    all_pred: dict[str, dict[str, np.ndarray]] = {}; execution = {}
    for arm, mode, seed in ARMS:
        all_pred[arm] = {}
        execution[arm] = {}
        for group, records in groups.items():
            reusable = load_reusable_partial(args.dest, arm, group, records) if args.resume else None
            if reusable is not None:
                p, proof = reusable
                all_pred[arm].update(p); execution[arm][group] = proof
                resumption["reused_group_shards"].append(f"{arm}/{group}")
                heartbeat(phase="resume_reuse", arm=arm, group=group, partial=proof["partial_file"])
                continue
            heartbeat(phase="arm", arm=arm, group=group)
            p, proof = run_group(records, arm, mode, seed, args.oracle_endpoints_per_group_arm)
            all_pred[arm].update(p); execution[arm][group] = proof
            np.savez_compressed(args.dest / f"partial_{arm}_{group}.npz", **{r["key"]: p[r["key"]] for r in records}, **{f"coords__{r['key']}": r["selected_endpoints"] for r in records})
    # A fresh REAL replay after every intervention proves normal inference
    # restores exactly without needing to mutate a decoder backwards.
    restored = {}
    for group, records in groups.items():
        heartbeat(phase="restoration", arm="REAL_RESTORED", group=group)
        p, proof = run_group(records, "REAL_RESTORED", "normal", None, args.oracle_endpoints_per_group_arm)
        restored.update(p); execution.setdefault("REAL_RESTORED", {})[group] = proof
    restoration = {}
    for key in all_pred["REAL"]:
        diff = np.abs(all_pred["REAL"][key] - restored[key])
        if not np.array_equal(all_pred["REAL"][key], restored[key]):
            raise RuntimeError(f"REAL restoration differs for {key}; max_abs={diff.max()}")
        restoration[key] = {"bitwise_equal": True, "max_abs_error": float(diff.max())}
    metrics = {}
    real_group_r2 = {}
    for group, records in groups.items():
        y = np.concatenate([r["targets"][r["selected_endpoints"]] for r in records])
        real_group_r2[group] = r2(y, np.concatenate([all_pred["REAL"][r["key"]] for r in records]))
    for arm, _mode, _seed in ARMS:
        group_r2 = {}
        pred_delta_l2 = {}
        for group, records in groups.items():
            y = np.concatenate([r["targets"][r["selected_endpoints"]] for r in records])
            got = np.concatenate([all_pred[arm][r["key"]] for r in records])
            base = np.concatenate([all_pred["REAL"][r["key"]] for r in records])
            group_r2[group] = r2(y, got)
            pred_delta_l2[group] = float(np.sqrt(np.mean((got - base) ** 2)))
        drops = {g: float(real_group_r2[g] - group_r2[g]) for g in surface["groups"]}
        metrics[arm] = {"group_r2": group_r2, "group_r2_equal_mean": float(np.mean(list(group_r2.values()))),
                        "paired_REAL_minus_arm_r2": drops,
                        "bootstrap_REAL_minus_arm_r2": grouped_bootstrap(drops),
                        "prediction_delta_rms_by_group": pred_delta_l2,
                        "prediction_delta_rms_equal_mean": float(np.mean(list(pred_delta_l2.values())))}
    # Aggregate shuffle within each C2 group first, then bootstrap the seven
    # resulting group-level effects; individual permutations stay available.
    shuf_names = [f"C_SHUF{seed}" for seed in PERMUTATION_SEEDS]
    shuf_group_r2 = {g: float(np.mean([metrics[a]["group_r2"][g] for a in shuf_names])) for g in surface["groups"]}
    shuf_drops = {g: float(real_group_r2[g] - shuf_group_r2[g]) for g in surface["groups"]}
    metrics["C_SHUF_MEAN_3SEEDS"] = {"aggregation": "mean three permutations within group before seven-group bootstrap",
                                      "group_r2": shuf_group_r2, "group_r2_equal_mean": float(np.mean(list(shuf_group_r2.values()))),
                                      "paired_REAL_minus_arm_r2": shuf_drops, "bootstrap_REAL_minus_arm_r2": grouped_bootstrap(shuf_drops)}
    arrays = {f"pred__{arm}__{key}": value for arm, by_key in all_pred.items() for key, value in by_key.items()}
    arrays.update({f"coords__{r['key']}": r["selected_endpoints"] for r in surface["records"]})
    np.savez_compressed(args.dest / "arm_predictions_and_coords.npz", **arrays)
    report = {"schema": "h1_bt_eort_direct_carrier_reliance_v1", "status": "COMPLETE", "utc": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "cpu": _cpu_name(), "affinity": sorted(os.sched_getaffinity(0)),
              "threads": {"torch_intra": 2, "torch_inter": 1, "ort_intra": 2, "ort_inter": 1},
              "payload": {"path": str(PAYLOAD), "sha256_before": payload_sha_before, "sha256_after": file_sha(PAYLOAD), "graphs": {p.name: file_sha(p) for p in sorted(GRAPHS.glob("*.onnx"))}},
              "protocol": {"surface": "H1 C2 HO-M3 visible held-out calibration", "groups": surface["groups"], "records": len(surface["records"]), "arms": [a for a, _, _ in ARMS], "intervention": "only direct normalized H-C SessionBank.T; E0/raw/mask/weights/normalizer untouched", "endpoint_sampling": "uniform linspace on concatenated valid endpoint indices per group, before prediction and without score-guided selection", "max_endpoints_per_group": args.max_endpoints_per_group, "query_inventory_sha256": surface["query_inventory_sha256"], "permutation_seeds": list(PERMUTATION_SEEDS), "zero_semantics": "direct normalized carrier tensor set to numeric zero", "full_surface_baseline_note": "The known full B2 baseline 0.378414 is not the sampled REAL result."},
              "source_bank_hashes": source_hashes, "execution": execution, "restoration": restoration, "metrics": metrics,
              "resumption": resumption,
              "prediction_archive": "arm_predictions_and_coords.npz"}
    (args.dest / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    heartbeat(status="COMPLETE", phase="complete", report="report.json")
    print(json.dumps({"status": "COMPLETE", "dest": str(args.dest), "real_r2_equal_mean": metrics["REAL"]["group_r2_equal_mean"], "arm_r2_equal_mean": {a: metrics[a]["group_r2_equal_mean"] for a, _, _ in ARMS}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
