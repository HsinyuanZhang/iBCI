"""Frozen non-network authorities for the AOF-S local package."""
from __future__ import annotations

from pathlib import Path
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v1/local_build"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_M2_AOF_SCALAR_OOF_CONFIRMATION_V1_20260903.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_AOF_SCALAR_STATIC_PACKAGE_V1_20260903.md"
RESULT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_ANCHORED_OUTPUT_MATRIX_FUSION_V1_20260903.md"
DESIGN_SHA256 = "69eaf228f0b9c11b2c00b24083ddea3b94e770b7c8e36674d94584ef383c9f4a"
WORKORDER_SHA256 = "15a49492c757a1da97b4fb9d4a723b81db715aa86d6d950eb81c21aa71805857"
RESULT_SHA256 = "b3b9e307c10dd8ae193bc1a5599d40ffb893abfc84c75eb87ad49a01ef67fd4a"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/artifacts/local_build_v1"

# Separate historical scored and local-reconstruction lineages; they are never
# SHA-equated merely because their calibration laws are scientifically matched.
OFFICIAL_581361_TERMINAL_RELATIVE = (
    "sua_exploration/evalai_t4_m2_activity_budget/artifacts/"
    "evalai_submission_581361_terminal_receipt_v1.json"
)
OFFICIAL_581361_TERMINAL_SHA256 = "68da5427e2d27169b8b6e1e08a487113b65b734a7e7644a5d1778f993febbb97"
OFFICIAL_581361_PUSH_RELATIVE = "sua_exploration/evalai_t4_m2_activity_budget/artifacts/evalai_push_state_m4_v1.json"
OFFICIAL_581361_PUSH_SHA256 = "015bdf9ccd3aa2c3959ff34b5b68346271ac59900f79edccb44505a518e33413"
OFFICIAL_581361_PAYLOAD_SHA256 = "c51b71167d81490927fee8a552785ee27ddff7e90085ed3ff4b18b857afbac40"
OFFICIAL_581361_IMAGE = "sha256:378bbd4868e515a3527418e098db2989e0fc8fbccc929ee029a237ce5e9f291b"
LOCAL_ACT30_PAYLOAD_SHA256 = "e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261"
LOCAL_ACT30_IMAGE = "sha256:79ac29ca71a84eb97be59411fc80d2a567127e95fd96b63dfa042d22b5a4421f"
LOCAL_DOCKER_BASE_TAG = "spint-m2:e8-epoch027-76f0fb2"
LOCAL_DOCKER_BASE_ID = "sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8"
STATIC_CLOSURE_RELATIVES = (
    DESIGN_RELATIVE,
    WORKORDER_RELATIVE,
    RESULT_RELATIVE,
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/export_aofs_static_payload.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/validate_local.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/driver.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/docker_local.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/decode.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/Dockerfile",
    "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/act30_dopt4_decoder.py",
    "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/export_act30_dopt4_payload.py",
    "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/laws.py",
    "tfpd_exploration/src/m2_anchored_output_fusion_v1/runner.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "sua_exploration/evalai_t4_m2/export_t4_payload.py",
    "sua_exploration/evalai_t4_m2/t4_spint_decoder.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "SPINT-main/third_party/__init__.py",
    "SPINT-main/third_party/falcon_challenge/__init__.py",
    "SPINT-main/third_party/falcon_challenge/filtering.py",
    "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v1.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    """Fail closed on any stale documentation authority before local admission."""
    expected = {
        DESIGN_RELATIVE: DESIGN_SHA256,
        WORKORDER_RELATIVE: WORKORDER_SHA256,
        RESULT_RELATIVE: RESULT_SHA256,
    }
    for relative, digest in expected.items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"AOF-S static authority drift: {relative}")
