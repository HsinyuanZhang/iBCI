#!/usr/bin/env python3
"""Production-checkpoint, no-data preflight for the factorized T4 logit screen."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
SCRIPTS = ROOT / "sua_exploration" / "scripts"
sys.path.insert(0, str(SCE))
sys.path.insert(0, str(SCRIPTS))

from eval_adaptation_dandi688 import checkpoint_architecture_kwargs  # noqa: E402
from src.models.t4_logit_residual_module import T4LogitResidualLitModule  # noqa: E402


ANCHOR = ROOT / "sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt"
TEACHER = ROOT / "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
PROTOCOL = ROOT / "sua_exploration/docs/SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md"
OUT = ROOT / "sua_exploration/results/sua_t4_factorized_logit_residual_v1/preflight.json"
EXPECTED_ANCHOR = "cf533e7cd97801d53985383b37f3fa1ff72fb6c385eaeb8c81273c3fa128273d"
EXPECTED_TEACHER = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model(mode: str, interaction: str) -> T4LogitResidualLitModule:
    checkpoint = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    architecture = checkpoint_architecture_kwargs(checkpoint)
    torch.manual_seed(42)
    instance = T4LogitResidualLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path=str(TEACHER),
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=False,
        loss_mode="task_only",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode="calibrated",
        encoder_warmstart_path=str(ANCHOR),
        optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=0.0),
        scheduler=None,
        compile=False,
        **architecture,
        residual_mode=mode,
        interaction_mode=interaction,
        residual_rank=8,
        residual_permutation_seed=42 if mode == "shuffled" else None,
    )
    instance.setup("fit")
    instance.eval()
    return instance


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    if sha256(ANCHOR) != EXPECTED_ANCHOR or sha256(TEACHER) != EXPECTED_TEACHER:
        raise SystemExit("selected anchor or teacher SHA does not match frozen protocol")
    aligned = model("aligned", "attention_logit")
    shuffled = model("shuffled", "attention_logit")
    additive = model("aligned", "additive_control")
    instances = {"aligned_logit": aligned, "shuffled_logit": shuffled, "additive_control": additive}

    torch.manual_seed(20260802)
    neural = torch.randn(2, 50, 64)
    identity = torch.randn(1, 64, 50)
    t4 = torch.randn(1, 64, 4)
    zero_checks = {}
    with torch.no_grad():
        for name, instance in instances.items():
            student = instance.student
            baseline = student.decode_with_identity(neural, identity)
            state = student.derive_t4_logit_residual_state(t4)
            cached = student.decode_with_t4_logit_residual_state(neural, identity, state)
            direct = student.decode_with_t4_logit_residual(neural, identity, t4)
            zero_checks[name] = {
                "inherited_coupled_bit_equal": torch.equal(baseline, cached),
                "cached_on_the_fly_bit_equal": torch.equal(cached, direct),
                "zero_bias_nonzero_count": int(
                    torch.count_nonzero(student.t4_logit_residual.logit_bias(state.unit_factors))
                ),
            }
    if not all(
        row["inherited_coupled_bit_equal"]
        and row["cached_on_the_fly_bit_equal"]
        and row["zero_bias_nonzero_count"] == 0
        for row in zero_checks.values()
    ):
        raise RuntimeError(f"zero-init/cache contract failed: {zero_checks}")

    # The two attention arms have identical restored substrate/factor bytes;
    # only the residual input row attachment differs.
    aligned_state = aligned.student.state_dict()
    shuffled_state = shuffled.student.state_dict()
    if set(aligned_state) != set(shuffled_state) or any(
        not torch.equal(aligned_state[name], shuffled_state[name]) for name in aligned_state
    ):
        raise RuntimeError("aligned/shuffled initialized states differ")
    with torch.no_grad():
        aligned.student.t4_logit_residual.query_factors.fill_(0.1)
        shuffled.student.t4_logit_residual.query_factors.fill_(0.1)
        aligned_active = aligned.student.decode_with_t4_logit_residual(neural, identity, t4)
        shuffled_active = shuffled.student.decode_with_t4_logit_residual(neural, identity, t4)
    if torch.equal(aligned_active, shuffled_active):
        raise RuntimeError("active aligned/shuffled residuals did not separate")

    permutation = torch.tensor([*range(63, -1, -1)])
    with torch.no_grad():
        invariant_reference = aligned.student.decode_with_t4_logit_residual(neural, identity, t4)
        invariant_permuted = aligned.student.decode_with_t4_logit_residual(
            neural[:, :, permutation], identity[:, permutation], t4[:, permutation]
        )
    permutation_max_abs = float((invariant_reference - invariant_permuted).abs().max())
    if not torch.allclose(invariant_reference, invariant_permuted, atol=1.0e-5, rtol=1.0e-6):
        raise RuntimeError(f"joint unit permutation invariance failed: {permutation_max_abs}")

    trainable = {
        name: sorted(name for name, parameter in instance.student.named_parameters() if parameter.requires_grad)
        for name, instance in instances.items()
    }
    expected_trainable = [
        "t4_logit_residual.query_factors",
        "t4_logit_residual.unit_projection.weight",
    ]
    if any(names != expected_trainable for names in trainable.values()):
        raise RuntimeError(f"optimizer whitelist failed: {trainable}")
    counts = {
        name: sum(parameter.numel() for parameter in instance.student.parameters() if parameter.requires_grad)
        for name, instance in instances.items()
    }
    if set(counts.values()) != {48}:
        raise RuntimeError(f"parameter matching failed: {counts}")

    receipt = {
        "schema_version": "sua_t4_factorized_logit_preflight_v1",
        "no_dataset_opened": True,
        "no_formal_test_opened": True,
        "anchor": {"path": str(ANCHOR), "sha256": sha256(ANCHOR)},
        "teacher": {"path": str(TEACHER), "sha256": sha256(TEACHER)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "script_sha256": sha256(Path(__file__)),
        "zero_and_cache_checks": zero_checks,
        "aligned_shuffled_initialized_state_bit_equal": True,
        "active_aligned_shuffled_outputs_differ": True,
        "joint_unit_permutation_max_abs_delta": permutation_max_abs,
        "optimizer_trainable_names": trainable,
        "trainable_parameter_count": counts,
        "cost_receipt_n64": aligned.student.residual_cost_receipt(batch_size=1, num_units=64),
        "gate": "pass",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
