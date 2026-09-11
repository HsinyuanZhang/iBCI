#!/usr/bin/env python3
"""dandi688 local-benchmark runner (bench_v1).

Stages:
  preflight  read-only prepared-cache contract verification + arm transform
             digest assertions + a small-batch CPU forward/backward smoke
             (and batch/stream parity) using the rift_v1 model definition,
             imported (never copied) from btransform_unified_v2.  CPU-only.
  train      12-epoch AdamW loop mirrored from the frozen rift_v1 runner,
             with protocol/arm-filtered rows and FORMAL_TRAIN contract.
  score      exam-face scoring (equal-session mean R^2 over the protocol's
             exam sessions: 4 for exp1_narrow, 6 for exp2_full) against
             average_e8_e11.pt from a completed train destination.

Protocols (ADDENDUM-TWO-STAGE, --protocol, default exp1_narrow):
  exp1_narrow  9-session train (2015-06-29..07-10) + 4-session exam
               (0713/0714/0715/0716, 3-6 days after the last train session).
  exp2_full    the original 27-train/6-val protocol (unchanged behavior).
  exp1_poverty extreme-data-poverty ablation (user directive 2026-09-10,
               plan B): train = ONLY 20150629/0630/0701 (the earliest three
               sessions, ~65k windows), exam face unchanged.  Re-measures the
               nested ladder (t4 / f_labelfree / floor) under poverty.
  exp2015_full 2015-only remote exam (user directive 2026-09-09): train = the
               18 sessions of the manifest train split dated 2015 (5 March +
               2 June + 11 July; the 9 sessions dated 2013 are dropped -- the
               ONLY difference from exp2_full), exam = the same 6 val
               sessions as exp2_full (2015-11, ~4 months after the last
               train session 2015-07-16).  Re-measures the nested ladder
               (t4 / f_labelfree / floor) under this train face.  Cross-
               protocol disclosure of exp2_full applies (plan.CROSS_PROTOCOL_
               DISCLOSURE): the exp1_narrow train AND exam sessions are
               members of this train face -- independent measurement, faces
               are never compared across protocols.

model-geometry ablation axis (user directive 2026-09-10, --model-override,
orthogonal to --arm; readings only, never gates):
  shortwin  W 50 -> 10: the model consumes only the LAST 10 bins of each
            frozen 50-bin window (200 ms history); query targets identical
            to the W=50 runs (plan D1).
  shallow   temporal depth 4 -> 1 over the same 50-bin receptive field via
            the bench-local RiftShallowDecoder subclass (plan D2; the
            rift_v1 main package is never modified).
  The override is recorded in the bench contract and both receipts; it must
  be passed IDENTICALLY at train and score time (fail-closed receipt check).
  Plan C1 (t4_dir16) uses no override: it is the plain t4 arm on the
  t4_dir16 variant cache (scripts/build_carrier_cache.py --variant t4_dir16).

vstate-688 series (user directive 2026-09-09; AUXILIARY exploration after the
component-ablation clarification, same day): --arm additionally accepts
  vstate / vstate_concat / z_vstate_srcbank / f_labelfree.  vstate and
  vstate_concat accept the vstate-family variant caches (vstate / vstate_full
  / vstate_b_hold; scripts/build_vstate_cache.py) via --prepared-cache —
  vstate_concat is the M2-mainline concat-fusion reading of the vstate
  carrier (readings only, never gates); f_labelfree requires its own variant
  cache; z_vstate_srcbank runs on the main vstate cache and swaps each exam
  session's bank (carrier + E0) for its frozen nearest-date train session's
  bank at load time ("完全不在新日期校准"), recording the mapping and digests
  in every receipt.  No training is started by this script's preflight stage.

component ablation (user clarification 2026-09-09, plan.COMPONENT_ABLATION —
the MAIN comparison axis): --arm additionally accepts
  floor      both-zero cell (E0 zeroed + carrier zeroed) on the frozen
             contract-v2 cache: the no-calibration-information floor.
  equiv_zero capacity control on the "equiv_zero" variant cache (built with
             the param-matched fixed random projection of the E0 pathway,
             seed 42, never trained; scripts/build_vstate_cache.py --variant
             equiv_zero): carrier zeroed here, E0 keeps the random-projection
             melt from the cache bytes.  Readings enter
             plan.component_ablation_report, never the preregistered gates.

Discipline (plan.py): GPU training unblocked by the 2026-09-09 revision
(ALLOWED_REVISION_20260909); preflight stays CPU-only; no formal-test data;
no writes outside btransform_unified_v2/dandi688_bench_v1/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]        # .../dandi688_bench_v1
WS = PKG_ROOT.parents[1]                              # SPINT workspace root
for p in (PKG_ROOT / "src", WS / "btransform_unified_v2" / "src",
          WS / "btransform_unified_v1" / "src", WS / "sua_exploration", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import arms as arms_mod
from dandi688_bench_v1 import eval_local, plan, receipts

UPDATES_PER_EPOCH_FROZEN = 28076  # Q50 update count of the frozen 27-train cache


# --------------------------------------------------------------------------
# prepared-cache loading (read-only, mirrors the rift_v1 verification branch)
# --------------------------------------------------------------------------
def verify_prepared_cache(cache_dir: Path) -> dict[str, Any]:
    """Re-verify the immutable prepared cache contract; return the meta dict.

    Read-only.  Binds schema, manifest sha, split counts, formal_test_used,
    absence of the frozen formal-test sessions, per-session array hashes and
    the 2 GiB cap (the same law the rift_v1 runner enforces)."""
    meta_path = cache_dir / "prepared_contract.json"
    if not meta_path.is_file():
        raise RuntimeError(f"prepared cache contract missing: {meta_path}")
    meta = json.loads(meta_path.read_text())
    required = {
        "schema": plan.PREPARED_CACHE_SCHEMA,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "split_counts": {"train": plan.SPLIT_COUNTS["train"], "val": plan.SPLIT_COUNTS["val"]},
        "formal_test_used": False,
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise RuntimeError(f"prepared cache contract mismatch on {key!r}: {meta.get(key)!r}")
    size = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
    if size >= plan.PREPARED_CACHE_MAX_GIB * 1024 ** 3:
        raise RuntimeError(f"prepared cache exceeds {plan.PREPARED_CACHE_MAX_GIB} GiB ({size} bytes)")
    return meta


def assert_cache_variant_for_arm(
    meta: dict[str, Any], arm: str, protocol: str
) -> dict[str, Any] | None:
    """Variant-cache binding of the vstate-688 arms (plan.ARM_REQUIRED_CACHE
    _VARIANT, values = allowed variant families): vstate/vstate_concat accept
    every vstate-family variant cache (vstate / vstate_full / vstate_b_hold)
    built for the ACTIVE protocol (its rms/column normalizer was fit on that
    protocol's train sessions); z_vstate_srcbank is defined against the main
    "vstate" variant only; f_labelfree requires the "f_labelfree" variant
    cache; norm_only requires the "norm_only" variant cache built for the
    ACTIVE protocol (its source rate distribution is fit on that protocol's
    train sessions).  The five original arms run on the frozen contract-v2
    cache (no variant field) and are unaffected.  Returns the binding record
    for receipts."""
    allowed = plan.ARM_REQUIRED_CACHE_VARIANT.get(arm)
    if allowed is None:
        return None
    variant = meta.get("variant")
    if variant not in allowed:
        raise RuntimeError(
            f"arm {arm} requires one of the {sorted(allowed)} variant caches, "
            f"got variant={variant!r} at the passed --prepared-cache"
        )
    binding: dict[str, Any] = {
        "arm": arm, "variant": variant, "allowed_variants": sorted(allowed),
    }
    if variant in plan.PROTOCOL_BOUND_CACHE_VARIANTS:
        built_protocol = meta.get("estimator", {}).get("protocol")
        if built_protocol != protocol:
            raise RuntimeError(
                f"arm {arm} on protocol {protocol} requires the {variant} "
                f"cache built for that protocol, got estimator.protocol="
                f"{built_protocol!r}"
            )
        binding["protocol"] = protocol
    return binding


def filter_rows_by_protocol(
    rows: dict[str, dict[str, Any]], protocol: str
) -> dict[str, dict[str, Any]]:
    """Pure protocol filter: keep only the protocol's train + exam sessions
    and tag each kept row with ``protocol_role`` ("train" / "exam").

    Cache rows already exist per session, so this is pure filtering (no
    re-derivation).  Fail closed when any protocol session is missing.
    Inputs are not mutated."""
    sessions = plan.protocol_sessions(protocol)
    roles: dict[str, str] = {name: "train" for name in sessions["train"]}
    roles.update({name: "exam" for name in sessions["exam"]})
    kept: dict[str, dict[str, Any]] = {}
    for name, row in rows.items():
        if name not in roles:
            continue
        tagged = dict(row)
        tagged["protocol_role"] = roles[name]
        kept[name] = tagged
    plan.require(
        len(kept) == len(roles) == len(sessions["train"]) + len(sessions["exam"]),
        f"protocol {protocol} expects {len(roles)} sessions "
        f"({len(sessions['train'])} train + {len(sessions['exam'])} exam), "
        f"kept {len(kept)}",
    )
    return kept


def load_rows(
    cache_dir: Path, meta: dict[str, Any], protocol: str = plan.DEFAULT_PROTOCOL
) -> dict[str, dict[str, Any]]:
    """Load, re-hash and protocol-filter the prepared-cache sessions.

    Only the sessions of the active protocol are loaded (every array still
    re-hashed against the cache contract); the remaining cache sessions are
    simply not touched.  Note for exp1_narrow: its exam sessions carry
    split == "train" in the cache meta (they ARE manifest-train sessions,
    see plan.CROSS_PROTOCOL_DISCLOSURE) — the authoritative role is
    ``protocol_role``."""
    sessions = plan.protocol_sessions(protocol)
    wanted = sorted(set(sessions["train"]) | set(sessions["exam"]))
    rows: dict[str, dict[str, Any]] = {}
    for name in wanted:
        if name in plan.FORMAL_TEST_SESSIONS:
            raise eval_local.TestSplitForbiddenError(
                f"protocol {protocol} unexpectedly names formal-test session {name}"
            )
        info = meta.get("sessions", {}).get(name)
        if info is None:
            raise RuntimeError(f"protocol session {name} missing from prepared cache {cache_dir}")
        row = eval_local.load_session(cache_dir, name)
        row["split"] = info["split"]
        rows[name] = row
    return filter_rows_by_protocol(rows, protocol)


def swap_exam_banks_from_train(
    armed: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
    protocol: str,
) -> None:
    """z_vstate_srcbank law (user directive 2026-09-09, "完全不在新日期
    校准（bank 复用旧日期/source）"): after the per-session arm transform,
    replace every exam session's bank (carrier + E0 rows) with its frozen
    nearest-date train session's bank.  Row-for-row copy into the exam
    session's row space (train real rows [0:k] -> exam rows [0:k],
    k = min(exam rows, source real rows); exam rows beyond k stay zero --
    the honest degradation of carrying an old bank onto a bigger new-date
    array).  Mutates ``armed`` in place and records the mapping, the copy
    geometry and the digests before/after into ``records``."""
    mapping = plan.z_srcbank_map(protocol)
    for exam_name, src_name in sorted(mapping.items()):
        if exam_name not in armed:
            continue  # this exam session is not on the active protocol face
        dst, src = armed[exam_name], armed[src_name]
        if src["protocol_role"] != "train" or dst["protocol_role"] != "exam":
            raise RuntimeError(
                f"z-srcbank swap geometry broken: {src_name} must be a train "
                f"row and {exam_name} an exam row"
            )
        plan.require(
            plan.session_date(exam_name) > plan.session_date(src_name),
            f"z-srcbank leakage: exam {exam_name} is not strictly after "
            f"bank source {src_name}",
        )
        n_dst = int(dst["mask"].size)
        n_src_real = int(np.asarray(src["mask"]).sum())
        n_dst_real = int(np.asarray(dst["mask"]).sum())
        k = min(n_dst, n_src_real)
        record = records[exam_name]
        record["carrier_sha256_before_swap"] = record["carrier_sha256_after"]
        record["e0_sha256_before_swap"] = record["e0_sha256_after"]
        carrier = np.zeros_like(dst["carrier"])
        e0 = np.zeros_like(dst["e0"])
        carrier[:k] = src["carrier"][:k]
        e0[:k] = src["e0"][:k]
        dst["carrier"], dst["e0"] = carrier, e0
        record["z_srcbank"] = {
            "source_session": src_name,
            "rule": plan.Z_SRCBANK_RULE,
            "exam_rows": n_dst,
            "exam_real_rows": n_dst_real,
            "source_real_rows": n_src_real,
            "rows_copied": int(k),
            "rows_left_zero": n_dst_real - min(k, n_dst_real),
            "carrier_sha256_source": plan.array_digest(src["carrier"]),
            "e0_sha256_source": plan.array_digest(src["e0"]),
            "exam_date_after_source": True,
        }
        record["carrier_sha256_after"] = plan.array_digest(carrier)
        record["e0_sha256_after"] = plan.array_digest(e0)


def apply_arm_to_rows(
    arm: str, rows: dict[str, dict[str, Any]], seed: int = plan.SEED,
    protocol: str = plan.DEFAULT_PROTOCOL,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (arm rows, verify records); inputs are not mutated."""
    armed: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, row in sorted(rows.items()):
        transformed = arms_mod.apply_arm(arm, name, row["carrier"], row["mask"], seed)
        records[name] = arms_mod.verify_arm(
            arm, name, row["carrier"], row["mask"], transformed, seed
        )
        armed[name] = dict(row)
        armed[name]["carrier"] = transformed
        e0_action = arms_mod.arm_e0_action(arm)
        if e0_action == "zero":
            armed[name]["e0"] = np.zeros_like(row["e0"])
            plan.require(not bool(np.any(armed[name]["e0"])),
                         f"{arm} E0 must be exactly zero")
        records[name]["e0_action"] = e0_action
        records[name]["e0_sha256_after"] = plan.array_digest(armed[name]["e0"])
    if arm == "z_vstate_srcbank":
        swap_exam_banks_from_train(armed, records, protocol)
    return armed, records


# --------------------------------------------------------------------------
# model glue (imports the rift_v1 model definition; never copies it)
# --------------------------------------------------------------------------
def assert_cpu_discipline(torch) -> None:
    """Preflight-only: refuse if CUDA has already been initialized."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; preflight is CPU-only by contract")


def window_bins_for(model_override: str = "none") -> int:
    """Active raw-history window width of a model override (2026-09-10 axis).

    shortwin consumes only the LAST plan.SHORTWIN_WINDOW_BINS bins of each
    frozen 50-bin window; the query targets stay byte-identical to the W=50
    runs, so the condition isolates exactly the raw-history budget."""
    if model_override not in plan.MODEL_OVERRIDES:
        raise ValueError(
            f"unknown model_override {model_override!r}; expected one of "
            f"{plan.MODEL_OVERRIDES}"
        )
    return (plan.SHORTWIN_WINDOW_BINS if model_override == "shortwin"
            else plan.WINDOW_BINS)


def build_model(
    n_units: int, torch, device, fusion: str = "proj_add",
    model_override: str = "none",
):
    """Build the RIFT model for the arm's identity-fusion mode and the
    2026-09-10 model-geometry override.

    proj_add: the settled frontend (P: 50->16 added onto local, token_in 20).
    concat:  the matched concat frontend from btransform_unified_v2.concat_model
             ([local16 | E0_50 | carrier4], token_in 70; function-preserving
             fold of the proj_add init, same geometry/seed).
    model_override (user directive 2026-09-10, orthogonal to the arm axis;
    readings only): "shortwin" = context_bins 50->10 (W=10, 200 ms history);
    "shallow" = temporal depth 4->1 via the bench-local RiftShallowDecoder
    subclass (the rift_v1 main package is never modified); "none" = stock
    geometry (byte-identical to the pre-2026-09-10 behavior)."""
    from btransform_unified_v2 import RiftDecoder

    if fusion not in plan.FUSION_MODES:
        raise ValueError(f"unknown fusion {fusion!r}; expected one of {plan.FUSION_MODES}")
    if model_override not in plan.MODEL_OVERRIDES:
        raise ValueError(
            f"unknown model_override {model_override!r}; expected one of "
            f"{plan.MODEL_OVERRIDES}"
        )
    context_bins = window_bins_for(model_override)
    geometry = {"task": plan.MODEL_TASK, "units": n_units, "e0_dim": plan.E0_DIM,
                "carrier_dim": plan.CARRIER_DIM, "out_dim": plan.OUT_DIM}
    if fusion == "concat":
        from btransform_unified_v2.concat_model import RiftConcatDecoder

        model = RiftConcatDecoder(
            geometry, context_bins=context_bins, bias_mode="recency",
            seed=plan.SEED,
        ).to(device)
        if model.frontend.token_in != plan.CONCAT_TOKEN_IN:
            raise RuntimeError(
                f"concat frontend token_in must be {plan.CONCAT_TOKEN_IN} "
                f"(16 local + {plan.E0_DIM} E0 + {plan.CARRIER_DIM} carrier), "
                f"got {model.frontend.token_in}"
            )
    elif model_override == "shallow":
        from dandi688_bench_v1.model_variants import RiftShallowDecoder

        model = RiftShallowDecoder(
            geometry, context_bins=context_bins, bias_mode="recency",
            seed=plan.SEED, proj_dim=16,
            temporal_layers=plan.SHALLOW_TEMPORAL_LAYERS,
        ).to(device)
    else:
        model = RiftDecoder(
            geometry, context_bins=context_bins, bias_mode="recency",
            seed=plan.SEED, proj_dim=16,
        ).to(device)
    model.temporal.set_attention_backend("local")
    return model


def make_bank(session_id: str, row: dict[str, Any], torch, window_bins: int | None = None):
    from btransform_unified_v1.bank import TaskBank

    w = plan.WINDOW_BINS if window_bins is None else int(window_bins)
    e0_hash = plan.array_digest(row["e0"])
    return TaskBank(
        session_id, row["e0"], row["carrier"], row["mask"],
        np.zeros((1, w, len(row["mask"])), np.float32),
        np.zeros((1, plan.OUT_DIM), np.float32),
        np.zeros(1, np.int64),
        {"shape": tuple(row["e0"].shape), "trial_count": 30, "budget": 30,
         "estimator": "frozen B3S post_pool(M30) + profile-M10",
         "array_sha256": e0_hash, "e0_sha256": e0_hash,
         "carrier_sha256": plan.array_digest(row["carrier"])},
    )


def windows(row: dict[str, Any], starts: np.ndarray, window_bins: int | None = None):
    """Slice query windows out of a frozen cache row.

    Default (window_bins=None): the frozen W=50 law -- x = neural[s:s+50],
    y = behavior[s+49].  window_bins=w < 50 takes the LAST w bins of each
    frozen 50-bin window (x = neural[s+50-w:s+50]) with the SAME target
    y = behavior[s+49]: identical query times and targets, less raw history
    (the 2026-09-10 shortwin condition)."""
    w = plan.WINDOW_BINS if window_bins is None else int(window_bins)
    if not (5 <= w <= plan.WINDOW_BINS):
        raise ValueError(f"window_bins must be in [5, {plan.WINDOW_BINS}], got {w}")
    x = np.stack(
        [row["neural"][s + plan.WINDOW_BINS - w:s + plan.WINDOW_BINS] for s in starts]
    ).astype(np.float32)
    y = np.stack([row["behavior"][s + plan.WINDOW_BINS - 1] for s in starts]).astype(np.float32)
    v = np.ones((len(starts), w), bool)
    return x, y, v


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def heartbeat(dest: Path, *, status: str, event: str, **payload: Any) -> None:
    dump(dest / "heartbeat.json", {
        "status": status,
        "event": event,
        "pid": os.getpid(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "epoch": payload.pop("epoch", None),
        "global_step": payload.pop("global_step", None),
        "loss": payload.pop("loss", None),
        "updates_per_epoch": payload.pop("updates_per_epoch", None),
        "mean_loss": payload.pop("mean_loss", None),
        "elapsed_seconds": payload.pop("elapsed_seconds", None),
        **payload,
    })


def batches(rows: dict[str, dict[str, Any]], epoch: int, protocol: str, arm: str):
    names = sorted(s for s, r in rows.items() if r["protocol_role"] == "train")
    seed = int.from_bytes(
        hashlib.sha256(
            f"{plan.SCHEMA}:{protocol}:{arm}:{plan.SEED}:{epoch}".encode()
        ).digest()[:8],
        "little",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    for i in rng.permutation(len(names)):
        session = names[int(i)]
        starts = rows[session]["starts"][rng.permutation(len(rows[session]["starts"]))]
        for offset in range(0, len(starts), plan.BATCH):
            yield session, np.ascontiguousarray(starts[offset:offset + plan.BATCH])


def rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("resume requires CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def checkpoint_state(epoch: int, step: int, model, optimizer, contract_sha256: str) -> dict[str, Any]:
    return {
        "schema": plan.SCHEMA + "_checkpoint",
        "status": "FORMAL_PARTIAL",
        "smoke": False,
        "epoch_zero_based": epoch,
        "next_epoch_zero_based": epoch + 1,
        "global_step": step,
        "epochs": plan.EPOCHS,
        "contract_sha256": contract_sha256,
        "raw_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
    }


def validate_resume(state: dict[str, Any], contract_sha256: str, dest: Path, updates_per_epoch: int) -> None:
    required = {
        "schema": plan.SCHEMA + "_checkpoint",
        "status": "FORMAL_PARTIAL",
        "smoke": False,
        "epochs": plan.EPOCHS,
        "contract_sha256": contract_sha256,
    }
    if any(state.get(key) != value for key, value in required.items()):
        raise RuntimeError("resume checkpoint contract mismatch")
    epoch = state.get("epoch_zero_based")
    nxt = state.get("next_epoch_zero_based")
    if not isinstance(epoch, int) or nxt != epoch + 1 or not 0 <= epoch < plan.EPOCHS - 1:
        raise RuntimeError("resume checkpoint has invalid epoch")
    if not isinstance(state.get("global_step"), int) or state["global_step"] != (epoch + 1) * updates_per_epoch:
        raise RuntimeError("resume checkpoint global step does not exactly match completed epochs")
    if not all(key in state for key in ("raw_state_dict", "optimizer", "rng")):
        raise RuntimeError("resume checkpoint missing model/optimizer/RNG state")
    if (dest / "train_receipt.json").exists() or (dest / "average_e8_e11.pt").exists():
        raise RuntimeError("destination is already complete; resume refused")


def average_checkpoint(dest: Path, contract_sha256: str):
    import torch

    states = []
    for epoch in plan.AVG_EPOCHS_ZERO_BASED:
        path = dest / f"epoch_{epoch:03d}.pt"
        if not path.is_file():
            raise RuntimeError(f"missing fixed averaging checkpoint {path.name}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state.get("contract_sha256") != contract_sha256
                or state.get("epoch_zero_based") != epoch
                or state.get("smoke")):
            raise RuntimeError("averaging checkpoint contract mismatch")
        states.append(state["raw_state_dict"])
    avg = {
        key: (
            torch.stack([s[key].double() for s in states]).mean(0).to(states[0][key].dtype)
            if states[0][key].is_floating_point()
            else states[0][key]
        )
        for key in states[0]
    }
    path = dest / "average_e8_e11.pt"
    atomic_torch_save(path, {
        "schema": plan.SCHEMA + "_average",
        "average_epochs_zero_based": list(plan.AVG_EPOCHS_ZERO_BASED),
        "state_dict": avg,
        "contract_sha256": contract_sha256,
    })
    return path


def updates_per_epoch(armed: dict[str, dict[str, Any]]) -> int:
    return sum(
        (len(r["starts"]) + plan.BATCH - 1) // plan.BATCH
        for r in armed.values() if r["protocol_role"] == "train"
    )


def assert_updates_sane(protocol: str, cache: Path, updates: int) -> None:
    if protocol == "exp2_full":
        if cache.resolve() == plan.prepared_cache_path().resolve() and updates != UPDATES_PER_EPOCH_FROZEN:
            raise RuntimeError(f"Q50 update count drift against the frozen cache: {updates}")
    else:
        plan.require(
            0 < updates < UPDATES_PER_EPOCH_FROZEN,
            f"exp1_narrow updates_per_epoch sanity failed: {updates}",
        )


def assert_saved_contract(dest: Path, contract: dict[str, Any]) -> None:
    path = dest / "data_contract.json"
    if not path.is_file():
        raise RuntimeError("missing data_contract.json")
    if json.loads(path.read_text()) != contract:
        raise RuntimeError("source/data/config contract mismatch; refusing incompatible destination")


def smoke_forward_backward(
    row: dict[str, Any], session_id: str, batch: int = 4, fusion: str = "proj_add",
    model_override: str = "none",
) -> dict[str, Any]:
    """Small-batch CPU forward+backward smoke plus batch/stream parity."""
    import torch
    from torch import nn
    from btransform_unified_v2 import RiftStreamDecoder

    assert_cpu_discipline(torch)
    device = torch.device("cpu")
    w = window_bins_for(model_override)
    model = build_model(
        row["neural"].shape[1], torch, device, fusion=fusion,
        model_override=model_override,
    )
    x, y, v = windows(row, np.asarray(row["starts"][:batch]), window_bins=w)
    tx = torch.from_numpy(x).to(device)
    tv = torch.from_numpy(v).to(device)
    bank = make_bank(session_id, row, torch, window_bins=w)
    loss = nn.functional.mse_loss(model(tx, bank, input_valid_mask=tv), torch.from_numpy(y).to(device))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite smoke loss")
    loss.backward()
    grads_finite = all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
    if not grads_finite:
        raise FloatingPointError("nonfinite smoke gradient")

    model.eval()
    with torch.inference_mode():
        # one-window oracle (same pattern as the rift_v1 preflight): the batch
        # decoder output for a single window equals the stream-step output
        offline = model.forward_scores(tx[:1], bank, input_valid_mask=tv[:1])[0]
        stream = RiftStreamDecoder(model)
        online = torch.stack(
            [stream.stream_step(tx[:1, t], bank, [session_id], valid_mask=tv[:1, t])[0]
             for t in range(w)]
        )
    parity = float((offline - online).abs().max())
    if not math.isfinite(parity) or parity > 1e-5:
        raise RuntimeError(f"batch/stream parity failed: max_abs={parity}")
    model.train()
    return {
        "session": session_id,
        "fusion": fusion,
        "model_override": model_override,
        "window_bins": int(w),
        "batch": int(len(np.asarray(row["starts"][:batch]))),
        "loss": float(loss.detach()),
        "gradients_finite": grads_finite,
        "stream_parity": {"max_abs": parity, "bins": int(w)},
        "forward_finite": True,
    }


# --------------------------------------------------------------------------
# bench contract (train/score assertion framework)
# --------------------------------------------------------------------------
def make_bench_contract(
    arm: str,
    meta: dict[str, Any],
    armed_rows: dict[str, dict[str, Any]],
    verify_records: dict[str, dict[str, Any]],
    updates: int,
    protocol: str = plan.DEFAULT_PROTOCOL,
    status: str = "FORMAL_TRAIN",
    model_override: str = "none",
) -> dict[str, Any]:
    if model_override not in plan.MODEL_OVERRIDES:
        raise ValueError(
            f"unknown model_override {model_override!r}; expected one of "
            f"{plan.MODEL_OVERRIDES}"
        )
    n_units = next(iter(armed_rows.values()))["neural"].shape[1]
    fusion = arms_mod.arm_fusion(arm)
    context_bins = window_bins_for(model_override)
    model_files = [
        WS / "btransform_unified_v2/src/btransform_unified_v2/model.py",
        WS / "btransform_unified_v2/src/btransform_unified_v2/temporal.py",
        WS / "btransform_unified_v2/src/btransform_unified_v2/streaming.py",
        WS / "btransform_unified_v2/src/btransform_unified_v2/config.py",
        WS / "btransform_unified_v1/src/btransform_unified_v1/model.py",
        WS / "btransform_unified_v1/src/btransform_unified_v1/plan.py",
    ]
    if fusion == "concat":
        model_files.append(WS / "btransform_unified_v2/src/btransform_unified_v2/concat_model.py")
    if model_override == "shallow":
        # the bench-local subclass carries the geometry change; bind its bytes
        model_files.append(
            WS / "btransform_unified_v2/dandi688_bench_v1/src/dandi688_bench_v1/model_variants.py"
        )
    return {
        "schema": plan.SCHEMA + "_contract",
        "arm": arm,
        "status": status,
        "seed": plan.SEED,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "prepared_cache_relative": plan.PREPARED_CACHE_RELATIVE,
        "prepared_contract_sha256": plan.obj_sha256(meta),
        "prepared_cache_variant": meta.get("variant"),
        "split_counts": {"train": plan.SPLIT_COUNTS["train"], "val": plan.SPLIT_COUNTS["val"]},
        "protocol": plan.protocol_receipt_block(protocol),
        "formal_test_used": False,
        "gpu_policy": plan.GPU_POLICY,
        "model_override": model_override,
        "window_bins": plan.WINDOW_BINS,
        "input_window_bins": context_bins,
        "epochs": plan.EPOCHS,
        "fixed_average_epochs_zero_based": list(plan.AVG_EPOCHS_ZERO_BASED),
        "updates_per_epoch": updates,
        "model_config": {"task": plan.MODEL_TASK, "units": n_units,
                         "e0_dim": plan.E0_DIM, "carrier_dim": plan.CARRIER_DIM,
                         "out_dim": plan.OUT_DIM, "context_bins": context_bins,
                         "bias_mode": "recency", "seed": plan.SEED,
                         "proj_dim": 16, "attention_backend": "local",
                         "identity_mode": fusion,
                         **({"temporal_layers": plan.SHALLOW_TEMPORAL_LAYERS}
                            if model_override == "shallow" else {}),
                         **({"token_in": plan.CONCAT_TOKEN_IN} if fusion == "concat" else {})},
        "model_source_hashes": {
            str(f.relative_to(WS)): plan.file_sha256(f) for f in model_files
        },
        "arm_carrier_sha256": {
            name: rec["carrier_sha256_after"] for name, rec in sorted(verify_records.items())
        },
        "runner_sha256": plan.file_sha256(Path(__file__)),
    }


def bench_contract_digest(contract: dict[str, Any]) -> str:
    return plan.obj_sha256(contract)


def _comparable_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Contract comparison view: drop runner_sha256 (self-hash) and default
    the 2026-09-10 model-override fields so contracts written before that
    axis existed (model_override absent == "none", input_window_bins absent
    == plan.WINDOW_BINS) compare equal under --model-override none.  Keeps
    every pre-existing run scorable with the current runner."""
    view = {k: v for k, v in contract.items() if k != "runner_sha256"}
    view.setdefault("model_override", "none")
    view.setdefault("input_window_bins", plan.WINDOW_BINS)
    return view


def stage_preflight(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(args.cpu_threads)
    assert_cpu_discipline(torch)
    plan.verify_protocol_definitions()
    cache = Path(args.prepared_cache)
    meta = verify_prepared_cache(cache)
    variant_binding = assert_cache_variant_for_arm(meta, args.arm, args.protocol)
    rows = load_rows(cache, meta, args.protocol)
    armed, records = apply_arm_to_rows(args.arm, rows, protocol=args.protocol)
    updates = updates_per_epoch(armed)
    assert_updates_sane(args.protocol, cache, updates)

    first_name = next(n for n, r in sorted(armed.items()) if r["protocol_role"] == "train")
    first_train = armed[first_name]
    fusion = arms_mod.arm_fusion(args.arm)
    smoke = smoke_forward_backward(
        first_train, first_name, batch=args.smoke_batch, fusion=fusion,
        model_override=args.model_override,
    )

    contract = make_bench_contract(
        args.arm, meta, armed, records, updates, args.protocol,
        model_override=args.model_override,
    )
    payload = {
        "schema": plan.SCHEMA + "_preflight_receipt",
        "status": "PASSED",
        "arm": args.arm,
        "fusion": fusion,
        "model_override": args.model_override,
        "protocol": plan.protocol_receipt_block(args.protocol),
        "gpu_policy": plan.GPU_POLICY,
        "cuda_initialized": False,
        "device": "cpu",
        "cpu_threads": int(args.cpu_threads),
        "prepared_cache": {
            "path": str(cache),
            "schema": meta.get("schema"),
            "manifest_sha256": meta.get("manifest_sha256"),
            "split_counts": meta.get("split_counts"),
            "formal_test_used": meta.get("formal_test_used"),
            "variant": meta.get("variant"),
            "variant_binding": variant_binding,
            "sessions_verified": len(rows),
            "arrays_rehashed": True,
        },
        "arm_verify": records,
        "smoke": {"source": "prepared_cache", **smoke},
        "updates_per_epoch": updates,
        "contract_sha256": bench_contract_digest(contract),
        "formal_test_used": False,
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    digest = receipts.seal_json(args.dest / f"preflight_receipt_{args.arm}.json", payload)
    print(f"preflight PASSED arm={args.arm} receipt_sha256={digest} "
          f"contract_sha256={payload['contract_sha256']}")


def stage_train(args: argparse.Namespace) -> None:
    """12-epoch formal train mirrored from the frozen rift_v1 loop."""
    import torch
    from torch import nn
    from btransform_unified_v1 import plan as v1_plan
    from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
    from tfpd_exploration.src.m2_dual_track_v1.training import build_optimizer

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    random.seed(plan.SEED)
    np.random.seed(plan.SEED)
    torch.manual_seed(plan.SEED)
    plan.verify_protocol_definitions()
    cache = Path(args.prepared_cache)
    meta = verify_prepared_cache(cache)
    assert_cache_variant_for_arm(meta, args.arm, args.protocol)
    rows = load_rows(cache, meta, args.protocol)
    armed, records = apply_arm_to_rows(args.arm, rows, protocol=args.protocol)
    updates = updates_per_epoch(armed)
    assert_updates_sane(args.protocol, cache, updates)
    contract = make_bench_contract(
        args.arm, meta, armed, records, updates, args.protocol, status="FORMAL_TRAIN",
        model_override=args.model_override,
    )
    digest = bench_contract_digest(contract)

    if not args.resume:
        if args.dest.exists() and any(args.dest.iterdir()):
            raise FileExistsError("new training destination must be empty")
        args.dest.mkdir(parents=True, exist_ok=True)
        dump(args.dest / "data_contract.json", contract)
    else:
        assert_saved_contract(args.dest, contract)

    n_units = next(iter(armed.values()))["neural"].shape[1]
    window_bins = window_bins_for(args.model_override)
    model = build_model(
        n_units, torch, device, fusion=arms_mod.arm_fusion(args.arm),
        model_override=args.model_override,
    )
    opt = build_optimizer(model.named_parameters(), lr=v1_plan.LR_PEAK, weight_decay=v1_plan.WEIGHT_DECAY)
    begin = 0
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume(state, digest, args.dest, updates)
        model.load_state_dict(state["raw_state_dict"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        begin = state["next_epoch_zero_based"]
        step = state["global_step"]

    heartbeat(
        args.dest, status="TRAINING", event="start",
        epoch=begin, global_step=step, updates_per_epoch=updates,
        contract_sha256=digest,
    )
    for epoch in range(begin, plan.EPOCHS):
        done = 0
        losses: list[float] = []
        epoch_started = time.monotonic()
        for batch_index, (session, starts) in enumerate(batches(armed, epoch, args.protocol, args.arm)):
            x, y, v = windows(armed[session], starts, window_bins=window_bins)
            opt.zero_grad()
            keep = whole_unit_dropout(
                torch.from_numpy(armed[session]["mask"]),
                p=v1_plan.UNIT_DROPOUT,
                generator=torch.Generator().manual_seed(unit_dropout_seed(plan.SEED, epoch, batch_index)),
            ).to(device)
            loss = nn.functional.mse_loss(
                model(
                    torch.from_numpy(x).to(device),
                    make_bank(session, armed[session], torch, window_bins=window_bins),
                    dropout_keep=keep,
                    input_valid_mask=torch.from_numpy(v).to(device),
                ),
                torch.from_numpy(y).to(device),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss epoch={epoch} batch={batch_index}")
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), v1_plan.GRAD_CLIP, error_if_nonfinite=True))
            opt.step()
            step += 1
            done += 1
            losses.append(float(loss.detach()))
            if step == 1 or step % 100 == 0:
                heartbeat(
                    args.dest, status="TRAINING", event="step",
                    epoch=epoch, global_step=step, loss=losses[-1],
                    updates_per_epoch=updates, mean_loss=float(np.mean(losses)),
                    elapsed_seconds=time.monotonic() - epoch_started,
                    batch=batch_index, grad_norm=grad_norm,
                )
        if done != updates or step != (epoch + 1) * updates:
            raise RuntimeError("short or inconsistent formal epoch")
        state = checkpoint_state(epoch, step, model, opt, digest)
        atomic_torch_save(args.dest / f"epoch_{epoch:03d}.pt", state)
        heartbeat(
            args.dest, status="TRAINING", event="epoch_checkpoint",
            epoch=epoch, global_step=step, loss=losses[-1],
            updates_per_epoch=updates, mean_loss=float(np.mean(losses)),
            elapsed_seconds=time.monotonic() - epoch_started,
            checkpoint_sha256=plan.file_sha256(args.dest / f"epoch_{epoch:03d}.pt"),
        )

    avg_path = average_checkpoint(args.dest, digest)
    receipt = {
        "schema": plan.SCHEMA + "_train_receipt",
        "status": "TRAIN_COMPLETED",
        "epochs": plan.EPOCHS,
        "formal_test_used": False,
        "arm": args.arm,
        "protocol": args.protocol,
        "model_override": args.model_override,
        "updates_per_epoch": updates,
        "global_step": step,
        "contract_sha256": digest,
        "prepared_cache": str(cache),
        "device": str(device),
        "average_checkpoint_sha256": plan.file_sha256(avg_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipts.seal_json(args.dest / "train_receipt.json", receipt)
    heartbeat(
        args.dest, status="TRAIN_COMPLETED", event="complete",
        epoch=plan.EPOCHS - 1, global_step=step, updates_per_epoch=updates,
        **{key: value for key, value in receipt.items()
           if key not in {"status", "global_step", "updates_per_epoch"}},
    )
    print(f"train COMPLETED arm={args.arm} protocol={args.protocol} "
          f"contract_sha256={digest} global_step={step}")


def stage_score(args: argparse.Namespace) -> None:
    """Exam-face scoring framework (requires a completed formal train dest)."""
    import torch

    torch.set_num_threads(args.cpu_threads)
    plan.verify_protocol_definitions()
    receipt_path = args.dest / "train_receipt.json"
    avg_path = args.dest / "average_e8_e11.pt"
    if not receipt_path.is_file() or not avg_path.is_file():
        raise RuntimeError(
            "score requires a completed formal train destination "
            f"(train_receipt.json + average_e8_e11.pt under {args.dest})"
        )
    receipt = receipts.read_sealed(receipt_path)
    required = {
        "schema": plan.SCHEMA + "_train_receipt",
        "status": "TRAIN_COMPLETED",
        "epochs": plan.EPOCHS,
        "formal_test_used": False,
        "arm": args.arm,
        "protocol": args.protocol,
        "model_override": args.model_override,
    }
    for key, value in required.items():
        # receipts sealed before the 2026-09-10 override axis carry no
        # model_override field; absence == "none"
        got = receipt.get(key)
        if key == "model_override" and got is None:
            got = "none"
        if got != value:
            raise RuntimeError(f"train receipt contract mismatch on {key!r}")

    cache = Path(args.prepared_cache)
    meta = verify_prepared_cache(cache)
    assert_cache_variant_for_arm(meta, args.arm, args.protocol)
    rows = load_rows(cache, meta, args.protocol)
    armed, records = apply_arm_to_rows(args.arm, rows, protocol=args.protocol)
    contract = make_bench_contract(
        args.arm, meta, armed, records, receipt["updates_per_epoch"], args.protocol,
        status="FORMAL_TRAIN", model_override=args.model_override,
    )
    live_digest = bench_contract_digest(contract)
    if receipt.get("contract_sha256") != live_digest:
        sealed_path = args.dest / "data_contract.json"
        if not sealed_path.is_file():
            raise RuntimeError("train receipt does not match the recomputed bench contract")
        sealed = json.loads(sealed_path.read_text())
        if receipt.get("contract_sha256") != bench_contract_digest(sealed):
            raise RuntimeError("train receipt does not match dest/data_contract.json")
        if _comparable_contract(contract) != _comparable_contract(sealed):
            raise RuntimeError(
                "train contract drifted beyond runner_sha256; refusing to score"
            )
        contract = sealed

    device = torch.device(args.device)
    window_bins = window_bins_for(args.model_override)
    model = build_model(
        next(iter(armed.values()))["neural"].shape[1], torch, device,
        fusion=arms_mod.arm_fusion(args.arm), model_override=args.model_override,
    )
    avg = torch.load(avg_path, map_location=device, weights_only=False)
    if avg.get("schema") != plan.SCHEMA + "_average" or avg.get("contract_sha256") != receipt["contract_sha256"]:
        raise RuntimeError("formal average checkpoint contract mismatch")
    model.load_state_dict(avg["state_dict"])
    model.eval()

    exam_sessions = plan.protocol_sessions(args.protocol)["exam"]
    session_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for name, row in sorted(armed.items()):
            if row["protocol_role"] != "exam":
                continue
            preds, targets = [], []
            for off in range(0, len(row["starts"]), plan.BATCH):
                x, y, v = windows(row, row["starts"][off:off + plan.BATCH],
                                  window_bins=window_bins)
                p = model(
                    torch.from_numpy(x).to(device),
                    make_bank(name, row, torch, window_bins=window_bins),
                    input_valid_mask=torch.from_numpy(v).to(device),
                ).cpu().numpy()
                preds.append(p)
                targets.append(y)
            session_outputs[name] = (np.concatenate(preds), np.concatenate(targets))
    face = eval_local.evaluate_val_face(
        session_outputs, sessions=exam_sessions,
        face_role=plan.PROTOCOLS[args.protocol]["face_role"],
    )
    face.update({"arm": args.arm, "protocol": plan.protocol_receipt_block(args.protocol),
                 "model_override": args.model_override,
                 "contract_sha256": receipt["contract_sha256"],
                 "average_checkpoint_sha256": plan.file_sha256(avg_path)})
    receipts.seal_json(args.dest / f"score_receipt_{args.arm}.json", face)
    print(f"score arm={args.arm} protocol={args.protocol} "
          f"equal_session_mean_r2={face['equal_session_mean_r2']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("preflight", "train", "score"), required=True)
    parser.add_argument("--arm", choices=plan.ARMS, required=True)
    parser.add_argument("--protocol", choices=tuple(plan.PROTOCOLS),
                        default=plan.DEFAULT_PROTOCOL,
                        help="two-stage experiment protocol (ADDENDUM-TWO-STAGE); "
                             "default exp1_narrow = narrow span first, then full")
    parser.add_argument("--dest", type=Path, required=True,
                        help="destination dir (must live under btransform_unified_v2/dandi688_bench_v1/)")
    parser.add_argument("--prepared-cache", type=Path, default=plan.prepared_cache_path())
    parser.add_argument("--model-override", choices=plan.MODEL_OVERRIDES,
                        default="none",
                        help="model-geometry ablation axis (user directive "
                             "2026-09-10; readings only): shortwin = W50->10 "
                             "(model sees only the last 10 bins of each "
                             "window), shallow = temporal depth 4->1 "
                             "(bench-local subclass).  Must be passed "
                             "identically at train and score time.")
    parser.add_argument("--device", default="cuda:0",
                        help="torch device for train/score (default cuda:0); preflight stays CPU")
    parser.add_argument("--resume", type=Path, default=None,
                        help="optional epoch_XXX.pt inside --dest to resume train")
    parser.add_argument("--cpu-threads", type=int, default=plan.CPU_THREADS_DEFAULT)
    parser.add_argument("--smoke-batch", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bench_root = (WS / "btransform_unified_v2" / "dandi688_bench_v1").resolve()
    if bench_root not in args.dest.resolve().parents:
        raise RuntimeError(f"--dest must live under {bench_root}")
    if args.stage != "train" and args.resume:
        raise ValueError("--resume is valid only for train")
    if args.resume is not None and args.resume.parent.resolve() != args.dest.resolve():
        raise RuntimeError("resume checkpoint must belong to --dest")
    {"preflight": stage_preflight, "train": stage_train, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
