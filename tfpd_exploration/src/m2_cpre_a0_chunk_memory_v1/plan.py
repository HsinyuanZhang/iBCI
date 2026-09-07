"""Frozen, data-free constants for M2 C-Pre / A0 V1."""

from __future__ import annotations

from dataclasses import dataclass

ROUTE_SCHEMA = "m2_cpre_a0_chunk_memory_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_CPRE_A0_CHUNK_MEMORY_V1_20260902.md"
WORKORDER_SHA256 = "5367edd6d6f3455c59fd892d5dd7113076191a1ca584fb4dfd3c975f02c7c64b"
DESIGN_REVIEW_RELATIVE = "tfpd_exploration/docs/DESIGN_REVIEW_CONTINUAL_CALIBRATION_NEXT_ROUTES_20260902.md"
DESIGN_REVIEW_SHA256 = "5a229071941a106733c30b0041816dbf5733f83af7dc481eea4c3ba4b2a36a40"

CPRE_ROOT_RELATIVE = "tfpd_exploration/results/m2_cpre_metadata_v1"
CPRE_V2_ROOT_RELATIVE = "tfpd_exploration/results/m2_cpre_metadata_v2"
CPRE_V2_REPAIR_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_CPRE_A0_CHUNK_MEMORY_V2_REPAIR_20260902.md"
CPRE_V2_REPAIR_SHA256 = "24b9ed27884d70b413f18aeadd827a0258d80eb96d717901643eec343510d055"
HISTORICAL_CPRE_BINDING_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_A0_HISTORICAL_CPRE_BINDING_V1_20260902.md"
HISTORICAL_CPRE_BINDING_SHA256 = "3374d874c948f9c89ec682b6977f52d07c5ed6d917c6335deec56273f41fdc79"
CPRE_V1_FAILURE_ATTEMPT_SHA256 = "47dabd4b1f364afb8e83402c1f7f098603e6fc3c6decbe2e1248f07699038e67"
CPRE_V1_FAILURE_SHA256 = "f91383045b4c7ee63b37272419b753d8779bbb456928d2ada081176e922f150a"
CPRE_V2_HISTORICAL_ATTEMPT_SHA256 = "64d594c44b1f9b7f4670caa00b3fc754db047a0c8ae592e5c45ee5018dbe4be8"
CPRE_V2_HISTORICAL_INVENTORY_SHA256 = "a0c76c912ca6aab10a427b3e28a50b1063dfbd68b7dc09d588f256bc1e5d7af2"
CPRE_V2_HISTORICAL_TERMINAL_SHA256 = "f98352a6aa95c35fe8fdc76bc5f9b47cdbac27a3c95aecfbf1f6f9a1692981b6"
CPRE_V2_HISTORICAL_CLOSURE_SHA256 = "c9d7851c97142c75cc3a2c0ac546ba156040e3decd26dffdec0d5558dbe59fe8"
A0_PARENT_RELATIVE = "tfpd_exploration/results/m2_a0_chunk_noninferiority_v1"
A0_ROOTS = {
    "external_post30_local": A0_PARENT_RELATIVE + "/external",
    "within_post30": A0_PARENT_RELATIVE + "/within",
    "aggregate": A0_PARENT_RELATIVE + "/aggregate",
}

PREDECESSOR_TERMINALS = {
    "memory_law": (
        "tfpd_exploration/results/m2_memory_law_scan_v1/terminal.json",
        "42462644b27eea8f9186b1990393b3472cca99779c4da64c1e28bc8c749e974b",
    ),
    "chrono4": (
        "tfpd_exploration/results/m2_chrono4_strict_v1/terminal.json",
        "77533ce4917047e42b32141fbe21632cea587cc6e6781f9de580decf4d29fc0a",
    ),
    "reblock10": (
        "tfpd_exploration/results/m2_reblock10_v1/terminal.json",
        "b77d8d370537d789879d83bbfa5676655884d1e6485d00dcf1c06d59b025993c",
    ),
}
ANCHOR_SCORE_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
ANCHOR_SCORE_SHA256 = "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce"
ANCHOR_CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
ANCHOR_NORMALIZER_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
ANCHOR_CELL = "ridge_activity30_m10"

CALIBRATION_TRIALS = 30
CARRIER_BUDGET = 10
M10_CHRONOLOGICAL_SELECTED_INDICES = tuple(range(10))
M10_CHRONOLOGICAL_SELECTED_INDICES_SHA256 = "9cc9e6b9e58fe00f5f6114b96eaed864e13bb732f14a2cd75d89930497e0b93f"
ACTIVITY_CAPACITY = 30
B3S_BINS = 100
WINDOW_BINS = 50
A0_FROZEN_DECODE_BATCH_SIZE = 32
CHUNK_LENGTH = 100
CHUNK_STRIDE = 100
PRIMARY_PHASE = 0
SENSITIVITY_PHASE = 50
EXPOSURE_GRID = (10, 30, 60, 120, "all-past")
K_GRID = (10, 30, 60, "infinity")
SURFACE_ORDER = ("external_post30_local", "within_post30")

EXTERNAL_GATE_MEAN = -0.005
EXTERNAL_GATE_WORST = -0.010
MATERIAL_BETTER_MEAN = 0.005

GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"


@dataclass(frozen=True)
class StaticDeviceProfile:
    physical_index: int = 0
    cuda_visible_devices: str = "0"
    logical_device: str = "cuda:0"
    uuid: str = GPU0_UUID
    gpu1_forbidden: bool = True


@dataclass(frozen=True)
class StaticSchedulerProfile:
    surface: str
    cpu_affinity: tuple[int, ...]
    workers: int = 0
    omp_threads: str = "1"
    mkl_threads: str = "1"
    openblas_threads: str = "1"
    numexpr_threads: str = "1"


SCHEDULER_PROFILES = {
    "external_post30_local": StaticSchedulerProfile(
        "external_post30_local", (0, 1, 2, 3, 16, 17, 18, 19)),
    "within_post30": StaticSchedulerProfile(
        "within_post30", (4, 5, 6, 7, 20, 21, 22, 23)),
}

# Explicit source closure only; no glob and no result paths.  Runtime-only
# sealed helpers are listed so a later capable execution cannot silently drift.
BOUND_PATTERNS = (
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/__init__.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/plan.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/inventory.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/chunk_memory.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/physical.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/receipts.py",
    "tfpd_exploration/scripts/run_m2_cpre_a0_chunk_memory_v1.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/physical.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "sua_exploration/evalai_t4_m2/export_t4_payload.py",
    "streaming_calibration_exp/src/models/falcon_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    WORKORDER_RELATIVE,
    CPRE_V2_REPAIR_RELATIVE,
    HISTORICAL_CPRE_BINDING_RELATIVE,
    DESIGN_REVIEW_RELATIVE,
)
