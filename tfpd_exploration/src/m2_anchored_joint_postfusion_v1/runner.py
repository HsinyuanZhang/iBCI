"""Post-attempt AJPF model preparation; never called by the inert CLI."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import adapter, plan, runtime


class RunnerError(RuntimeError):
    pass


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise RunnerError(message)


@dataclass
class PreparedJointArms:
    datamodule: Any
    modules: dict[str, Any]
    adapters: dict[str, Any]
    optimizers: dict[str, Any]
    strict_load_evidence: dict[str, Any]
    pit_prepare_calls: int
    decoder_dropout_original: dict[str, dict[str, Any]]


def _student_state_sha256(student: Any) -> str:
    """Use the reviewed scorer's digest law, only after a strict load."""
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    # The reviewed scorer primitive receives a lit module, whereas this
    # shared runner intentionally holds the student.  The holder preserves
    # that exact tensor framing without inventing a second digest law.
    holder = type("AJPFStrictStudentHolder", (), {"student": student})()
    return physical._student_state_sha(holder)


def _authorize_joint_trainable(module: Any, arm: str) -> tuple[list[Any], list[Any]]:
    """Unfreeze exactly id_encoder/decoder after inherited decoder unfreeze."""
    student = module.student
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    student._decoder_frozen = False
    student.train(True)
    _need(student.decoder.training is True and student._decoder_frozen is False,
          "AJPF decoder did not enter authorized train state")
    encoder_alpha: list[Any] = []
    decoder: list[Any] = []
    for name, parameter in student.named_parameters():
        if name.startswith("id_encoder."):
            parameter.requires_grad_(True); encoder_alpha.append(parameter)
        elif name.startswith("decoder."):
            parameter.requires_grad_(True); decoder.append(parameter)
    _need(encoder_alpha and decoder and not set(map(id, encoder_alpha)).intersection(map(id, decoder)),
          "AJPF trainable parameter topology drift")
    _need(all(not parameter.requires_grad for name, parameter in module.named_parameters()
              if not name.startswith("student.id_encoder.") and not name.startswith("student.decoder.")),
          "AJPF teacher/nonstudent parameter became trainable")
    return encoder_alpha, decoder


def _fresh_adam(*, torch: Any, encoder_alpha: list[Any], decoder: list[Any]) -> Any:
    optimizer = torch.optim.Adam([
        {"params": encoder_alpha, "lr": plan.ENCODER_ALPHA_LR},
        {"params": decoder, "lr": plan.DECODER_LR},
    ], betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS, weight_decay=0.0, amsgrad=False)
    runtime.validate_two_group_adam(optimizer=optimizer, encoder_alpha=encoder_alpha, decoder=decoder)
    return optimizer


def _disable_clone_internal_unit_dropout(module: Any) -> dict[str, Any]:
    """Reserve decoder unit dropout for AJPF's paired, route-owned hook.

    Decoder training must remain enabled.  We therefore only suppress the two
    decoder branches which would otherwise draw a separate mask in every arm;
    transformer dropout remains paired by restoration of the post-mask RNG.
    """
    decoder = module.student.decoder
    original = {
        "dynamic_dropout": bool(getattr(decoder, "dynamic_dropout", False)),
        "dropout_rate": float(getattr(decoder, "dropout_rate", 0.0)),
        "dynamic_dropout_low": float(getattr(decoder, "dynamic_dropout_low", 0.0)),
        "dynamic_dropout_high": float(getattr(decoder, "dynamic_dropout_high", 0.0)),
    }
    decoder.dynamic_dropout = False
    decoder.dropout_rate = 0.0
    _need(module.student.decoder.training is True, "AJPF disabled decoder training with internal dropout")
    return original


def prepare_joint_arms_after_attempt(*, repo_root: Path, device: str, launch_attestation: dict[str, Any]) -> PreparedJointArms:
    """Construct PIT exactly once, strict-load once, then clone all three arms.

    This function is deliberately post-attempt and is the future production
    entry only.  It neither reads held-out data nor chooses an arm/checkpoint.
    """
    import torch
    from tfpd_exploration.src.pit_m2_v1 import trainer as pit_trainer
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical

    _need(device == "cuda:0" and torch.cuda.is_initialized() and torch.cuda.current_device() == 0
          and torch.cuda.device_count() == 1 and launch_attestation.get("logical_device") == 0,
          "AJPF requires launch-attested single logical CUDA0")
    runner = pit_trainer.PitM2ArmedRunner(repo_root, "t0m", device=device)
    runner.prepare(attach_operator=False)
    _need(runner._litmodule is not None and runner._datamodule is not None, "AJPF single PIT source stack missing")
    # AJPF's route-owned causal source coordinator must not ask the PIT
    # DataLoader to reconstruct a randomized/static-prefix stream.  The only
    # permitted future consumer is the held APFG-coordinate materializer.
    _need(not hasattr(runner._datamodule, "train_dataloader") or callable(runner._datamodule.train_dataloader),
          "AJPF PIT DataModule topology unavailable")
    strict_native, evidence = physical._strict_sealed_pooled_clone(repo_root=repo_root, base_module=runner._litmodule)
    _need(_student_state_sha256(strict_native.student) == plan.STUDENT_STATE_SHA256, "AJPF selected strict state drift")
    native_encoder_authority = _student_state_sha256(strict_native.student.id_encoder)
    modules: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    optimizers: dict[str, Any] = {}
    dropout_original: dict[str, dict[str, Any]] = {}
    for arm in plan.ARM_ORDER:
        module = copy.deepcopy(strict_native)
        _need(module is not strict_native, "AJPF clone object alias")
        before = _student_state_sha256(module.student)
        if arm != "J-NATIVE":
            installed = adapter.install_after_strict_load(module.student, arm)
            adapters[arm] = installed
            if arm == "J-R1":
                adapter.require_positive_zero(installed.alpha)
        if arm == "J-NATIVE":
            _need(before == plan.STUDENT_STATE_SHA256, "AJPF native clone state drift")
        else:
            _need(_student_state_sha256(module.student.id_encoder.native) == native_encoder_authority,
                  "AJPF wrapper native substate drift")
        module.to(torch.device(device))
        enc, dec = _authorize_joint_trainable(module, arm)
        dropout_original[arm] = _disable_clone_internal_unit_dropout(module)
        modules[arm] = module
        optimizers[arm] = _fresh_adam(torch=torch, encoder_alpha=enc, decoder=dec)
    ids = [id(param) for module in modules.values() for param in module.parameters()]
    _need(len(ids) == len(set(ids)), "AJPF arms share Parameter objects")
    return PreparedJointArms(datamodule=runner._datamodule, modules=modules, adapters=adapters, optimizers=optimizers,
                             strict_load_evidence=dict(evidence), pit_prepare_calls=1,
                             decoder_dropout_original=dropout_original)
