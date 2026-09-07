"""Write score-free source fit wall/peak evidence for one owned Phase-C cell."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time

from lightning.pytorch import Callback
import torch


FOLDS = {
    0: "ses-2020-10-19-Run1", 1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1", 3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1", 5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}


class Post33SourceCostRuntimeV4(Callback):
    def __init__(
        self, *, output_path: str, resolved_config_path: str,
        cell_owner_path: str, owner_token: str, cost_supplement_path: str,
        arm: str, fold: int, seed: int,
    ) -> None:
        if arm not in {"spint", "t4"} or fold not in FOLDS or seed not in {42, 43, 44}:
            raise ValueError("source runtime cell identity invalid")
        self.output_path = Path(output_path).resolve()
        self.resolved_config_path = Path(resolved_config_path).resolve()
        self.cell_owner_path = Path(cell_owner_path).resolve(strict=True)
        self.owner_token = owner_token
        self.cost_supplement_path = Path(cost_supplement_path).resolve(strict=True)
        self.arm, self.fold, self.seed = arm, fold, seed
        self.started_ns: int | None = None
        self.train_batches = 0
        self.validation_batches = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        del pl_module
        owner = json.loads(self.cell_owner_path.read_text(encoding="utf-8"))
        expected_identity = (
            "m2_post33_phase_c_cell_ownership_v4", "PHASE_C_V4", self.arm,
            self.fold, self.seed, self.owner_token,
        )
        observed_identity = (
            owner.get("schema"), owner.get("phase_id"), owner.get("arm"),
            owner.get("fold"), owner.get("seed"), owner.get("owner_token"),
        )
        if observed_identity != expected_identity:
            raise PermissionError("source runtime callback ownership mismatch")
        cell = self.cell_owner_path.parent.parent.resolve(strict=True)
        if self.output_path != (cell / "run/source_cost_evidence.json").resolve():
            raise ValueError("source runtime output is outside canonical owned run")
        if self.resolved_config_path != (cell / "run/resolved_config.yaml").resolve():
            raise ValueError("source runtime resolved config is outside canonical owned run")
        if trainer.world_size != 1 or not torch.cuda.is_available():
            raise RuntimeError("Phase-C source runtime evidence requires one CUDA process")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started_ns = time.perf_counter_ns()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        del trainer, pl_module, batch, batch_idx
        if self.started_ns is not None:
            self.train_batches += 1

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        del pl_module, batch, batch_idx, dataloader_idx
        if self.started_ns is not None and not trainer.sanity_checking:
            self.validation_batches += 1

    def on_fit_end(self, trainer, pl_module) -> None:
        del trainer, pl_module
        if self.started_ns is None:
            raise RuntimeError("source runtime timer never started")
        torch.cuda.synchronize()
        wall = time.perf_counter_ns() - self.started_ns
        config = self.resolved_config_path.resolve(strict=True)
        import hashlib
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        stat = config.stat()
        sources = [session for fold, session in FOLDS.items() if fold != self.fold]
        supplement = json.loads(self.cost_supplement_path.read_text(encoding="utf-8"))
        if (
            supplement.get("schema") != "m2_post33_phase_c_cost_supplement_v4"
            or supplement.get("protocol_id") != "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
            or supplement.get("phase_id") != "PHASE_C_V4"
        ):
            raise ValueError("source runtime cost supplement mismatch")
        plan = supplement["source_plan_by_arm_fold_seed"][self.arm][str(self.fold)][str(self.seed)]
        if self.train_batches != plan["expected_train_batch_executions"]:
            raise ValueError("actual train batch executions differ from sealed cost plan")
        if self.validation_batches != plan["expected_validation_batch_executions_excluding_sanity"]:
            raise ValueError("actual validation batch executions differ from sealed cost plan")
        audit_path = Path(supplement["source_batch_audit"]["canonical_path"]).resolve(strict=True)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_row = audit["arms"][self.arm][str(self.fold)]
        audit_row_sha = hashlib.sha256(
            json.dumps(audit_row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        supplement_stat = self.cost_supplement_path.stat()
        supplement_sha = hashlib.sha256(self.cost_supplement_path.read_bytes()).hexdigest()
        payload = {
            "schema": "m2_post33_phase_c_source_cost_evidence_v4",
            "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
            "phase_id": "PHASE_C_V4",
            "arm": self.arm, "fold": self.fold, "seed": self.seed,
            "outer_session": FOLDS[self.fold], "source_sessions": sources,
            "scope": "fit_source_train_plus_exact_six_source_validation",
            "outer_scorer_calls": 0, "formal_data_accessed": False,
            "wall_time_ns": wall,
            "train_batch_executions": self.train_batches,
            "validation_batch_executions": self.validation_batches,
            "peak_memory_bytes": {
                "cuda_allocated": int(torch.cuda.max_memory_allocated()),
                "cuda_reserved": int(torch.cuda.max_memory_reserved()),
                "host_max_rss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            },
            "resolved_config": {
                "canonical_path": str(config), "size_bytes": stat.st_size, "sha256": digest,
            },
            "cost_supplement": {
                "canonical_path": str(self.cost_supplement_path),
                "size_bytes": supplement_stat.st_size, "sha256": supplement_sha,
            },
            "source_batch_audit_row_sha256": audit_row_sha,
        }
        data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
