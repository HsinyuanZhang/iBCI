"""V2-only scorer-compatible strict student state digest seam."""
from __future__ import annotations
from typing import Any
from . import plan
class RunnerError(RuntimeError): pass
class _StudentHolder:
    def __init__(self, student: Any) -> None: self.student=student
def strict_student_digest(student: Any) -> str:
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    digest=physical._student_state_sha(_StudentHolder(student))
    if digest != plan.SELECTED_STUDENT_STATE_SHA256: raise RunnerError("AJPF V2 strict selected student digest drift")
    return digest
def source_only_cpu_strict_load(*, repo_root: Any) -> dict[str,Any]:
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RunnerError("AJPF V2 source regression requires empty CVD")
    import torch
    if torch.cuda.is_initialized(): raise RunnerError("AJPF V2 source regression initialized CUDA")
    from tfpd_exploration.src.pit_m2_v1 import trainer
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    source=trainer.PitM2ArmedRunner(repo_root,"t0m",device="cpu");source.prepare(attach_operator=False)
    strict,evidence=physical._strict_sealed_pooled_clone(repo_root=repo_root,base_module=source._litmodule)
    digest=strict_student_digest(strict.student)
    return {"cpu_only":True,"cuda_initialized":False,"strict_load":True,"student_state_sha256":digest,"selected_t4_checkpoint_sha256":evidence["selected_t4_checkpoint_sha256"],"pit_prepare_calls":1}
