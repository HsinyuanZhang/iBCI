"""Frozen constants for the M2 A0 deterministic-cuBLAS V2 successor."""

from __future__ import annotations

ROUTE_SCHEMA = "m2_a0_deterministic_cublas_v2"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_A0_DETERMINISTIC_CUBLAS_SUCCESSOR_V2_20260902.md"
WORKORDER_SHA256 = "3e0c8b960d8a9c1226c1f93612c5adbffe085c8c4b437f220648c5a4b63f1574"
HISTORICAL_CPRE_BINDING_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_A0_HISTORICAL_CPRE_BINDING_V1_20260902.md"
HISTORICAL_CPRE_BINDING_SHA256 = "3374d874c948f9c89ec682b6977f52d07c5ed6d917c6335deec56273f41fdc79"

V1_EXTERNAL_ROOT_RELATIVE = "tfpd_exploration/results/m2_a0_chunk_noninferiority_v1/external"
V1_EXTERNAL_ATTEMPT_SHA256 = "a417a5d9b33e0d5edf7aa415c26b4012808e3a939202645aef2ff840ef6e2cd0"
V1_EXTERNAL_LAUNCH_SHA256 = "33942ddcf9d8f856926537fb7ced39f3511e3e569bffda415f5ef1f8623373fc"
V1_EXTERNAL_FAILURE_SHA256 = "ca1fd1b8077285c9010de4c7a772620051e64d35f4fdcd23f582faaf6e173093"
V1_HISTORICAL_CLOSURE_SHA256 = "35890d88e27de8809b87baaacfe43bb004a15775ac7ea99916771996c957b5d4"

V2_PARENT_RELATIVE = "tfpd_exploration/results/m2_a0_chunk_noninferiority_v2"
V2_ROOTS = {
    "external_post30_local": V2_PARENT_RELATIVE + "/external",
    "within_post30": V2_PARENT_RELATIVE + "/within",
    "aggregate": V2_PARENT_RELATIVE + "/aggregate",
}

GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
DETERMINISTIC_ENVIRONMENT = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

# This is deliberately explicit: no result paths, tests, or glob expansion.
BOUND_PATTERNS = (
    "tfpd_exploration/src/m2_a0_deterministic_cublas_v2/__init__.py",
    "tfpd_exploration/src/m2_a0_deterministic_cublas_v2/plan.py",
    "tfpd_exploration/src/m2_a0_deterministic_cublas_v2/binding.py",
    "tfpd_exploration/src/m2_a0_deterministic_cublas_v2/execution.py",
    "tfpd_exploration/src/m2_a0_deterministic_cublas_v2/lifecycle.py",
    "tfpd_exploration/scripts/run_m2_a0_deterministic_cublas_v2.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/__init__.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/plan.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/inventory.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/chunk_memory.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/physical.py",
    "tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/receipts.py",
    WORKORDER_RELATIVE,
    HISTORICAL_CPRE_BINDING_RELATIVE,
)
