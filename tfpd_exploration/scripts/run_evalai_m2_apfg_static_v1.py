#!/usr/bin/env python3
"""Dry by default; reserve or execute the static-pool APFG EvalAI M2 probe.

Exploratory deployment of the frozen m2_spint_t4_mainline seed-42 checkpoint
``25d7bc72...``: the offline calibration law is byte-identical to the
officially scored ``dopt4_static_act30`` arm (greedy forward D-optimal k=4
within the first-30 finite-angle candidates, ridge lambda=0.1 T4 on the
selected support, label-free first-30 B3S activity pool), plus ONE anchored
post-fusion scalar gate applied offline at build time:
``E = native + tanh(alpha) * (post - native)``, ``alpha = -0.20759029686450958``
bound to the immutable V2 same-surface result graph.

Scientific status is pre-registered here: the local growing-pool external
effect (+0.004612, 4/6) did NOT pass the +0.010 promotion gate, and the
static-pool deployment is a lawful approximation of the requested UNCAPPED
configuration, not that configuration.  No official-score improvement may be
claimed from this cell alone.

Stages
------
- ``dry``       : print the plan, touch nothing.
- ``tests``     : run the package unit tests (no data, no CUDA).
- ``attempt``   : write the 0444+sidecar attempt receipt (pre-registration).
- ``export``    : build the official payload (offline calibration + gate).
- ``validate``  : payload audit + native-seal replay + ZERO==NATIVE sentinel
                  + contract assertions + host FalconEvaluator minival.
- ``build``     : docker build of the submission image (no push).
- ``terminal``  : write the 0444+sidecar terminal receipt.
- ``execute``   : tests -> attempt -> export -> validate -> build -> terminal.

No stage performs any network submission; the EvalAI push is operator-owned
and gated on explicit user confirmation (see submit_evalai_apfg_static.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
PACKAGE = TFPD_ROOT / "submissions/evalai_m2_apfg_static_v1"
RESULT_ROOT = TFPD_ROOT / "results/evalai_m2_apfg_static_v1"

CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
FROZEN_ALPHA = -0.20759029686450958
V2_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_postfusion_gate_v2_same_surface_control"
V2_TERMINAL_SHA256 = "9ea88d3a4e9048c484a4e67575a3b1c9a1323fbf0558d739acb065217f2a7b63"
V2_SCORE_SHA256 = "5dd7c3e911ca00709e028590af93c065a38f5ea1bd1369dc27fabda87b595d6e"
SEALED_SCREEN = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_AUDIT = "tfpd_exploration/results/cdm_p1_m2_v1/audit.json"
BASE_IMAGE = "spint-m2:e8-epoch027-76f0fb2"
PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_apfg_identity.pkl"
VALIDATION_NAME = "t4_m2_seed42_dopt4_act30_apfg_validation.pkl"

# Native-arm sealed anchors (the officially scored act30_dopt4 configuration).
ANCHOR_EXTERNAL = 0.2909923623168425
ANCHOR_WITHIN = 0.6772200181830803
# Officially scored static baselines for descriptive interpretation only.
OFFICIAL_ACT30_DOPT4_HO = 0.2897
OFFICIAL_ACT30_FULL_HO = 0.295

OWNED_PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"
OWNED_TESTS_RELATIVE = "tfpd_exploration/tests/test_evalai_m2_apfg_static_v1.py"
OWNED_DRIVER_RELATIVE = "tfpd_exploration/scripts/run_evalai_m2_apfg_static_v1.py"
OWNED_WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_APFG_STATIC_OFFICIAL_PROBE_V1_20260902.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{REPO_ROOT}:{PACKAGE}",
        "CUDA_VISIBLE_DEVICES": "",
    }


def _write_receipt(path: Path, payload: dict) -> dict:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    path.chmod(0o444)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    return {"receipt": str(path), "receipt_sha256": digest}


def _run(command: list[str], log_name: str, cwd: Path) -> str:
    log_path = TFPD_ROOT / f"evalai_m2_apfg_static_v1_{log_name}.log"
    completed = subprocess.run(
        command, cwd=str(cwd), env=_env(), capture_output=True, text=True
    )
    log_path.write_text(
        f"$ {' '.join(command)}\ncwd={cwd}\nrc={completed.returncode}\n\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1][:400] if completed.stderr.strip() else ""
        raise SystemExit(f"stage failed (rc={completed.returncode}), see {log_path}: {tail}")
    return completed.stdout


def owned_relative() -> list[str]:
    owned = [
        f"{OWNED_PACKAGE_RELATIVE}/{entry.name}"
        for entry in sorted(PACKAGE.iterdir())
        if entry.is_file()
    ]
    owned += [OWNED_TESTS_RELATIVE, OWNED_DRIVER_RELATIVE, OWNED_WORKORDER_RELATIVE]
    return sorted(set(owned))


def export_receipt() -> dict:
    return json.loads(
        (PACKAGE / "artifacts" / (Path(PAYLOAD_NAME).with_suffix(".receipt.json"))).read_text(
            encoding="utf-8"
        )
    )


def stage_tests() -> None:
    _run(
        [
            sys.executable, "-m", "pytest",
            "tfpd_exploration/tests/test_evalai_m2_apfg_static_v1.py",
            "-q",
        ],
        "tests",
        cwd=REPO_ROOT,
    )
    print(json.dumps({"stage": "tests", "status": "PASSED"}, sort_keys=True))


def stage_attempt() -> None:
    target = RESULT_ROOT / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": "evalai_m2_apfg_static_v1_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "EVALAI_M2_APFG_STATIC_V1_PACKAGING",
        "authority": (
            "user 2026-09-02 forward of the Luna-verified M2-APFG build request; "
            "interface ruling: official continual M2 provides no completed-trial "
            "boundary (sealed audit results/cdm_p1_m2_v1/audit.json) and the "
            "trial-free chunk alternative failed its pre-registered "
            "noninferiority gate (results/m2_a0_chunk_noninferiority_v2), so the "
            "user selected the static-pool approximation"
        ),
        "inference_only": True,
        "predecessor_package": "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1",
        "pre_registration": {
            "deployment": (
                "offline calibration byte-identical to dopt4_static_act30 plus one "
                "anchored post-fusion scalar gate applied at build time: "
                "E = native + tanh(alpha)*(post - native), alpha frozen"
            ),
            "frozen_alpha": FROZEN_ALPHA,
            "alpha_provenance": {
                "v2_result_root": V2_ROOT_RELATIVE,
                "v2_terminal_sha256": V2_TERMINAL_SHA256,
                "v2_score_sha256": V2_SCORE_SHA256,
            },
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "native_sealed_anchor": {
                "path": SEALED_SCREEN,
                "cell": "ridge_activity30_m4",
                "external_official_query_equal_session_mean": ANCHOR_EXTERNAL,
                "within_post30_equal_session_mean": ANCHOR_WITHIN,
            },
            "interpretation_rule": (
                "the official held-out score is a DESCRIPTIVE probe of a static "
                "approximation; it must be read against the officially scored "
                "act30_dopt4 (0.2897) and act30_full (0.295) baselines and cannot "
                "promote the APFG method, whose local external effect missed the "
                "pre-registered +0.010 gate"
            ),
            "contract": (
                "per the sealed cdm_p1_m2_v1 audit: continual m2 evaluator calls only "
                "reset(dataset_tags)+predict(neural_observations); no on_done; "
                "assert no trial metadata at predict, gate applied offline only, "
                "weights/identities byte-frozen across predicts"
            ),
            "sentinels": [
                "build time: APFG-ZERO identity == native identity bitwise, 13/13 sessions",
                "validate: decode-level ZERO run == NATIVE run exactly on prediction "
                "bytes/R2/target/starts/window count for all 13 sessions",
                "validate: native replay within 5e-3 of the sealed screen rows",
            ],
            "stages": ["tests", "export", "validate", "build", "terminal"],
            "submission": "NOT PERFORMED by this cell; the operator executes the documented push after user confirmation",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "no_user_site": True,
            "cuda_visible_devices": "",
        },
        "owned_sha256s": {item: sha256_file(REPO_ROOT / item) for item in owned_relative()},
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    print(json.dumps(_write_receipt(target, payload), indent=2, sort_keys=True))


def stage_export() -> None:
    stdout = _run(
        [sys.executable, str(PACKAGE / "export_apfg_static_payload.py"), "--execute"],
        "export",
        cwd=PACKAGE,
    )
    receipt = json.loads(stdout)
    print(json.dumps(
        {
            k: receipt[k]
            for k in (
                "payload_sha256",
                "session_count",
                "arm",
                "frozen_alpha",
                "zero_equals_native_sessions",
                "prior_deployed_identity_sha256_matches",
            )
        },
        sort_keys=True,
    ))


def stage_validate() -> None:
    _run(
        [
            sys.executable,
            str(PACKAGE / "validate_local.py"),
            "--execute",
            "--stages",
            "payload,native_seal,sentinel,contract,minival",
        ],
        "validate",
        cwd=PACKAGE,
    )


def stage_build() -> dict:
    receipt = export_receipt()
    tag = f"spint-t4-m2:apfg-static-s42-{receipt['payload_sha256'][:8]}"
    _run(
        [
            "docker", "build",
            "--build-arg", f"PAYLOAD_SHA256={receipt['payload_sha256']}",
            "-t", tag,
            "-f", "Dockerfile", ".",
        ],
        "build",
        cwd=PACKAGE,
    )
    inspect = json.loads(
        _run(["docker", "image", "inspect", tag], "image_inspect", cwd=REPO_ROOT)
    )[0]
    built = {
        "tag": tag,
        "id": inspect["Id"],
        "size": inspect.get("Size"),
        "labels": inspect["Config"].get("Labels", {}),
    }
    print(json.dumps({k: built[k] for k in ("tag", "id", "size")}, sort_keys=True))
    return built


def stage_terminal(built: dict) -> None:
    target = RESULT_ROOT / "terminal.json"
    if target.exists():
        raise SystemExit(f"terminal receipt already exists: {target}")
    receipt = export_receipt()
    validation = json.loads(
        (PACKAGE / "artifacts" / "local_validation_receipt.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema": "evalai_m2_apfg_static_v1_terminal_v1",
        "status": "TERMINAL",
        "cell": "EVALAI_M2_APFG_STATIC_V1_PACKAGING",
        "submission_performed": False,
        "docker": {"tag": built["tag"], "image_id": built["id"], "size": built["size"]},
        "artifact_tree": {
            "payload": {
                "path": f"{OWNED_PACKAGE_RELATIVE}/artifacts/{PAYLOAD_NAME}",
                "sha256": receipt["payload_sha256"],
            },
            "validation_payload": {
                "path": f"{OWNED_PACKAGE_RELATIVE}/artifacts/{VALIDATION_NAME}",
                "sha256": receipt["validation_sha256"],
            },
        },
        "frozen_alpha": receipt["frozen_alpha"],
        "alpha_provenance": receipt["alpha_provenance"],
        "zero_equals_native_sessions": receipt["zero_equals_native_sessions"],
        "prior_deployed_identity_sha256_matches": receipt["prior_deployed_identity_sha256_matches"],
        "local_validation": {
            "status": validation["status"],
            "sentinel": validation.get("zero_native_sentinel"),
            "contract": validation.get("contract"),
            "minival_metrics": validation.get("official_local_minival", {}).get("metrics"),
        },
        "interpretation": (
            "exploratory static-pool probe; official held-out score is descriptive "
            "against act30_dopt4 0.2897 / act30_full 0.295 and cannot promote APFG"
        ),
        "target_gradients": 0,
        "parameter_updates": 0,
    }
    print(json.dumps(_write_receipt(target, payload), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--stages",
        default="",
        help="comma list of tests,attempt,export,validate,build,terminal",
    )
    args = parser.parse_args()
    if args.plan or not args.stages:
        print(json.dumps({
            "schema": "evalai_m2_apfg_static_v1_driver_plan_v1",
            "status": "DRY_PLAN_ONLY",
            "stages": ["tests", "attempt", "export", "validate", "build", "terminal"],
            "package": OWNED_PACKAGE_RELATIVE,
            "result_root": str(RESULT_ROOT.relative_to(REPO_ROOT)),
            "frozen_alpha": FROZEN_ALPHA,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "native_anchor_external": ANCHOR_EXTERNAL,
            "official_baselines": {"act30_dopt4_ho": OFFICIAL_ACT30_DOPT4_HO, "act30_full_ho": OFFICIAL_ACT30_FULL_HO},
            "submission": "operator-owned push, gated on explicit user confirmation",
            "push_preflight": f"python {OWNED_PACKAGE_RELATIVE}/submit_evalai_apfg_static.py --plan",
        }, indent=2, sort_keys=True))
        return

    stage_order = ["tests", "attempt", "export", "validate", "build", "terminal"]
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in stage_order]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}")
    if "terminal" in requested and "build" not in requested:
        raise SystemExit("terminal requires the build stage in the same invocation")
    built = None
    for stage in requested:
        if stage == "tests":
            stage_tests()
        elif stage == "attempt":
            stage_attempt()
        elif stage == "export":
            stage_export()
        elif stage == "validate":
            stage_validate()
        elif stage == "build":
            built = stage_build()
        elif stage == "terminal":
            stage_terminal(built or {})


if __name__ == "__main__":
    main()
