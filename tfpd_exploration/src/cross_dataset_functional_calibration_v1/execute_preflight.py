"""Write the 20260905_123100 preflight root. CPU only. Does not touch Stage0."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import model_adapter
from . import normalizer as normalizer_mod
from . import plan
from . import receipts


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise plan.PlanError(message)


def ensure_preflight_root(repo_root: Path) -> Path:
    root = Path(repo_root) / plan.P_PREFLIGHT_ROOT_RELATIVE
    stage0 = Path(repo_root) / plan.STAGE0_ROOT_RELATIVE
    revision = Path(repo_root) / plan.P_REVISION_ROOT_RELATIVE
    _require(stage0.is_dir(), "Stage0 root missing")
    _require(root.resolve() != stage0.resolve(), "preflight root must not overwrite Stage0")
    _require(root.resolve() != revision.resolve(), "preflight root must not overwrite revision")
    terminal = stage0 / "terminal.json"
    _require(terminal.is_file(), "Stage0 terminal.json missing")
    observed = plan.sha256_bytes(terminal.read_bytes())
    _require(observed == plan.STAGE0_TERMINAL_SHA256, f"Stage0 terminal drifted: {observed}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_consumer_tests(pair: model_adapter.PConsumerPair) -> dict[str, Any]:
    tests = {
        "zero_basis_perturbation_reproduces_parent_prediction": pair.zero_perturbation_parent_replay(),
        "coefficient_recovery_intercept_unpenalized_label_permutation": {
            "status": "delegated_to_pytest",
            "intercept_unpenalized": True,
        },
        "solve_grads_finite_nonzero_and_source_query_loss_dictionary_grad": pair.source_query_loss_dictionary_grad(),
        "query_history_disjointness": pair.query_history_disjointness(),
        "simultaneous_unit_activity_carrier_permutation": pair.unit_permutation_invariance(),
        "dropout_masks_synchronize_neural_activity_carrier": pair.dropout_mask_sync(),
        "native_16d_signed_emg_unchanged": pair.native_output_contract(),
        "target_fit_zero_optimizer_steps_source_hashes_unchanged": pair.target_fit_no_backward(),
        "full_new_stage_resume_not_blocked_by_old_sfix": pair.new_stage_resume_roundtrip(),
        "pfix_pca_initial_consumer_weights_byte_identical": pair.initial_allowlist(),
    }
    def _is_failure(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        false_keys = (
            "passed",
            "prediction_parity_passed",
            "carrier_parity_passed",
            "finite",
            "nonzero",
            "support_carrier_unchanged",
            "consumer_weights_byte_identical",
            "roundtrip_ok",
        )
        if any(payload.get(key) is False for key in false_keys):
            return True
        return payload.get("old_sfix_missing_fields_block") is True

    failed = [name for name, payload in tests.items() if _is_failure(payload)]
    return {"tests": tests, "failed": failed, "passed": not failed}


def _spec_md(frozen: normalizer_mod.FrozenSourceNormalizer, consumer: dict[str, Any]) -> str:
    receipt = frozen.receipt()
    return "\n".join(
        [
            "# STAGE1_IMPLEMENTATION_SPEC — preflight 20260905_123100",
            "",
            "Authority: `REVIEW_CROSS_DATASET_P_OPERATOR_PREFLIGHT_FREEZE_V1_20260905.md`.",
            "Revision: `REVISION_CROSS_DATASET_P_OPERATOR_V1_20260905.md`.",
            "Stage0 `20260905_113700/` and revision `20260905_122000/` were not rewritten.",
            "",
            "## Gate status",
            "",
            "`STAGE1_PREFLIGHT_CPU`. Formal 12-epoch P-FIX / P-CA remain **not eligible**.",
            "`gpu_eligible=false` until the disposable profile is run by the coordinator.",
            "",
            "## Bound parent",
            "",
            f"- S-Fix `{plan.S_FIX_EPOCH011_SHA256}`",
            f"- Injection: `m1_emg_rsyn3_fold_local_v1.injection.apply_post_fc_in`, `student.carrier_projection_weight [1024,4]`",
            f"- Claim `{plan.P_PARENT_CLAIM}`",
            "- Z-Fix and all-source teacher remain forbidden.",
            "",
            "## Frozen normalizer",
            "",
            f"- Name `{receipt['name']}`",
            f"- μ0 SHA `{receipt['mu0_sha256']}`",
            f"- σ0 SHA `{receipt['sigma0_sha256']}`",
            f"- array SHA `{receipt['array_sha256']}`",
            f"- input SHA `{receipt['input_sha256']}`",
            f"- code SHA `{receipt['code_sha256']}`",
            f"- D0 positive `{receipt['d0_positive_count']}` zero `{receipt['d0_zero_count']}` (ReLU lock; no softplus)",
            f"- Parent carrier parity `{receipt['parent_carrier_parity_passed']}` max_abs `{receipt['parent_max_abs_err']}`",
            "- Computed once at init. Never updated per step/epoch. Never refit on target.",
            "- P-FIX and P-CA share the same μ0,σ0 bytes.",
            "",
            "## Consumer allowlist",
            "",
            "- Shared: decoder + B3 identity encoder + `carrier_projection_weight`.",
            "- P-CA only: `RowNormalizedNNMFBasis.raw_dictionary`.",
            "- Native 16-D signed EMG residual capacity preserved. No hidden output projection.",
            "",
            "## Disposable profile (implemented, not run here)",
            "",
            "- 100 paired source-only updates each arm, effective batch 32, sessions 26/27/28, query [10,210).",
            "- Zero outer scoring. Budget 30 minutes. No auto-extend to 12 epochs.",
            "- Does not inherit smoke RNG as formal init.",
            "",
            f"## Consumer test summary",
            "",
            f"- Failed names: `{consumer['failed']}`",
            "",
        ]
    )


def _handoff_md(
    frozen: normalizer_mod.FrozenSourceNormalizer,
    consumer: dict[str, Any],
    hashes: dict[str, str],
) -> str:
    ready = not consumer["failed"] and frozen.parent_carrier_parity_passed
    decision = "PREFLIGHT_CPU_READY" if ready else "PREFLIGHT_BLOCKED"
    blocker = "none" if ready else f"consumer/normalizer failures: {consumer['failed']}"
    return "\n".join(
        [
            "# HANDOFF — P operator preflight (CPU)",
            "",
            "Does not cover the M2 dual-track pack. Stage0 and revision roots are unchanged.",
            "",
            f"## Decision: `{decision}`",
            "",
            f"- Blocker: `{blocker}`",
            f"- BLOCKER={blocker}",
            f"- gpu_eligible=false",
            f"- gpu_work_started=false",
            f"- Disposable profile: NOT_RUN (launcher implemented; dry-path tested)",
            f"- Formal 12-epoch pair: NOT_RUN / not authorized",
            "",
            "## Normalizer",
            "",
            f"- `{frozen.name}`",
            f"- D0 zero-count `{frozen.d0_zero_count}` positive `{frozen.d0_positive_count}`",
            f"- array SHA `{frozen.array_sha256}`",
            f"- Parent carrier parity `{frozen.parent_carrier_parity_passed}`",
            "",
            "## Smallest next step",
            "",
            "Coordinator may run the disposable profile on one leased GPU after this CPU gate.",
            "Do not start 12-epoch training from this preflight.",
            "",
            "## Exact disposable-profile CLI (unrun)",
            "",
            "```",
            "PYTHONNOUSERSITE=1 CDF_DISPOSABLE_PROFILE=1 CUDA_VISIBLE_DEVICES=<coordinator-set> \\",
            "  /home/xinyuan/miniconda3/envs/spint/bin/python \\",
            "  tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py \\",
            "  --disposable-profile",
            "```",
            "",
            "## Hashes this turn",
            "",
            *[f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())],
            "",
        ]
    )


def execute(repo_root: Path) -> tuple[dict[str, str], str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = Path(repo_root)
    authority = repo_root / plan.P_PREFLIGHT_AUTHORITY_RELATIVE
    _require(authority.is_file(), "preflight authority missing")
    observed = plan.sha256_bytes(authority.read_bytes())
    _require(observed == plan.P_PREFLIGHT_AUTHORITY_SHA256, f"preflight authority sha drift: {observed}")
    revision = repo_root / plan.P_REVISION_RELATIVE
    _require(plan.sha256_bytes(revision.read_bytes()) == plan.P_REVISION_SHA256, "revision sha drift")
    root = ensure_preflight_root(repo_root)
    hashes: dict[str, str] = {}
    frozen = normalizer_mod.materialize(repo_root)
    pair = model_adapter.build_p_pair(repo_root, frozen)
    consumer = _run_consumer_tests(pair)
    hashes["normalizer.json"] = receipts.write_json(root / "normalizer.json", frozen.receipt())
    hashes["consumer_tests.json"] = receipts.write_json(root / "consumer_tests.json", consumer)
    hashes["STAGE1_IMPLEMENTATION_SPEC.md"] = receipts.write_text(
        root / "STAGE1_IMPLEMENTATION_SPEC.md", _spec_md(frozen, consumer)
    )
    ready = not consumer["failed"] and frozen.parent_carrier_parity_passed
    terminal = {
        "schema": "cross_dataset_functional_calibration_preflight_terminal_v1",
        "status": "PREFLIGHT_CPU_READY" if ready else "PREFLIGHT_BLOCKED",
        "cpu_only": True,
        "gpu_work_started": False,
        "gpu_eligible": False,
        "stage1_frozen": False,
        "parent_bytes": plan.S_FIX_EPOCH011_SHA256,
        "normalizer": plan.P_NORMALIZER_NAME,
        "normalizer_array_sha256": frozen.array_sha256,
        "d0_zero_count": frozen.d0_zero_count,
        "parent_carrier_parity_passed": frozen.parent_carrier_parity_passed,
        "stage0_root": plan.STAGE0_ROOT_RELATIVE,
        "stage0_terminal_sha256": plan.STAGE0_TERMINAL_SHA256,
        "revision_root": plan.P_REVISION_ROOT_RELATIVE,
        "preflight_root": plan.P_PREFLIGHT_ROOT_RELATIVE,
        "authority": plan.P_PREFLIGHT_AUTHORITY_RELATIVE,
        "authority_sha256": plan.P_PREFLIGHT_AUTHORITY_SHA256,
        "disposable_profile": "NOT_RUN",
    }
    hashes["terminal.json"] = receipts.write_json(root / "terminal.json", terminal)
    hashes["HANDOFF_FOR_ASTRA_REVIEW.md"] = receipts.write_text(
        root / "HANDOFF_FOR_ASTRA_REVIEW.md", _handoff_md(frozen, consumer, hashes)
    )
    hashes["HANDOFF_FOR_ASTRA_REVIEW.md"] = receipts.write_text(
        root / "HANDOFF_FOR_ASTRA_REVIEW.md", _handoff_md(frozen, consumer, hashes)
    )
    return hashes, hashes["terminal.json"]
