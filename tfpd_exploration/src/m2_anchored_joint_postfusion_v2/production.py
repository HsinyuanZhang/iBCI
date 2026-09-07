"""V2 outer composition over the unchanged nested V1 scientific executor."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path
from . import lifecycle, runner
def execute_preflight(*, repo_root: Path, root: Path) -> dict[str,str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("AJPF V2 preflight requires CPU-only CVD")
    reviewed=lifecycle.closure(repo_root); root.mkdir(mode=0o755)
    attempt=lifecycle.pair(root,"attempt.json",{"schema":"m2_anchored_joint_postfusion_v2_attempt","closure":reviewed,"target_access":False,"cuda_initialized":False})
    incident=lifecycle.hold_v1_failure(repo_root); incident_sha=lifecycle.pair(root,"v1_incident.json",incident)
    regression=runner.source_only_cpu_strict_load(repo_root=repo_root); regression_sha=lifecycle.pair(root,"strict_digest_regression.json",regression)
    terminal=lifecycle.pair(root,"terminal.json",{"schema":"m2_anchored_joint_postfusion_v2_terminal","attempt_sha256":attempt,"v1_incident_sha256":incident_sha,"strict_digest_regression_sha256":regression_sha,"terminal_xor_failure":True})
    return {"root":str(root),"terminal_sha256":terminal}


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_regression_subprocess(repo_root: Path) -> dict[str,object]:
    """Fresh CPU admission proof before the GPU executor is entered."""
    environment=dict(os.environ); environment.update({"CUDA_VISIBLE_DEVICES":"", "PYTHONPATH":str(repo_root), "PYTHONNOUSERSITE":"1"})
    code=("import json;from pathlib import Path;from tfpd_exploration.src.m2_anchored_joint_postfusion_v2.runner "
          "import source_only_cpu_strict_load;print(json.dumps(source_only_cpu_strict_load(repo_root=Path("+repr(str(repo_root))+"))))")
    child=subprocess.run([sys.executable,"-c",code],cwd=str(repo_root),env=environment,check=True,capture_output=True,text=True)
    return json.loads(child.stdout.splitlines()[-1])


def execute_composed(*, cap: lifecycle.Capability) -> dict[str,str]:
    """Root-only V2 route: outer V2 graph, exact nested V1 executor graphs."""
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1_lifecycle
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production as v1_production
    repo_root=cap.repo_root; outer_root,root_identity=lifecycle.consume(cap)
    combined=cap.closure; incident=lifecycle.hold_v1_failure(repo_root); published=[]
    try:
        attempt=lifecycle.pair(outer_root,"attempt.json",{"schema":"m2_anchored_joint_postfusion_v2_attempt","combined_closure":combined,"target_access":False});published.append("attempt.json")
        incident_sha=lifecycle.pair(outer_root,"v1_incident.json",incident);published.append("v1_incident.json")
        regression=_cpu_regression_subprocess(repo_root); regression_sha=lifecycle.pair(outer_root,"strict_digest_regression.json",regression);published.append("strict_digest_regression.json")
        v1_map=v1_lifecycle.closure_map(repo_root); training=outer_root/"training"
        train_cap=v1_lifecycle.issue_production_capability(repo_root=repo_root,root=training,reviewed=v1_map,digest=v1_lifecycle.closure_sha256(v1_map))
        train=v1_production.execute_training(cap=train_cap,repo_root=repo_root)
        v1_lifecycle.validate_training_topology(training,terminal=True)
        score=outer_root/"score"
        environment=dict(os.environ); environment.update({"CUDA_VISIBLE_DEVICES":"","CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONHASHSEED":"0","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","PYTHONPATH":str(repo_root)})
        code=("from pathlib import Path;from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle,production;"
              "r=Path("+repr(str(repo_root))+ ");m=lifecycle.closure_map(r);c=lifecycle.issue_production_capability(repo_root=r,root=Path("+repr(str(score))+"),reviewed=m,digest=lifecycle.closure_sha256(m));production.execute_cpu_score(cap=c,repo_root=r,training_root=Path("+repr(str(training))+"))")
        subprocess.run([sys.executable,"-c",code],cwd=str(repo_root),env=environment,check=True)
        v1_lifecycle.validate_score_topology(score,terminal=True)
        _need_outer(cap=cap,root_identity=root_identity,incident=incident,training=training,score=score,terminal=True)
        terminal=lifecycle.pair(outer_root,"terminal.json",{"schema":"m2_anchored_joint_postfusion_v2_terminal","attempt_sha256":attempt,"v1_incident_sha256":incident_sha,"strict_digest_regression_sha256":regression_sha,"nested_v1_training_terminal_sha256":_sha(training/"terminal.json"),"nested_v1_score_terminal_sha256":_sha(score/"terminal.json"),"nested_v1_training_manifest_sha256":_sha(training/"manifest.json"),"terminal_xor_failure":True});published.append("terminal.json")
        lifecycle.validate_outer(outer_root,terminal=True)
        return {"root":str(outer_root),"terminal_sha256":terminal,"training_root":str(training),"score_root":str(score)}
    except Exception as error:
        if outer_root.exists() and "terminal.json" not in published:
            if (outer_root/"training").is_dir() and (outer_root/"score").is_dir(): _need_outer(cap=cap,root_identity=root_identity,incident=incident,training=outer_root/"training",score=outer_root/"score",terminal=False)
            lifecycle.pair(outer_root,"failure.json",{"schema":"m2_anchored_joint_postfusion_v2_failure","published_prefix":published,"exception_class":type(error).__name__,"exception_message":str(error)[:400],"terminal_xor_failure":False})
        raise


def _need_outer(*,cap: lifecycle.Capability,root_identity: tuple[int,int],incident: dict[str,object],training: Path,score: Path,terminal: bool) -> None:
    info=cap.root.stat(follow_symlinks=False)
    if (int(info.st_dev),int(info.st_ino))!=root_identity or lifecycle.combined_closure(cap.repo_root)!=cap.closure: raise RuntimeError("AJPF V2 outer identity/closure drift")
    if lifecycle.hold_v1_failure(cap.repo_root)!=incident: raise RuntimeError("AJPF V2 incident drift")
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1
    if terminal:
        v1.validate_training_topology(training,terminal=True);v1.validate_score_topology(score,terminal=True)


def _execute_synthetic_composed_for_test(*, cap: lifecycle.Capability, fail_after_incident: bool = False) -> dict[str,str]:
    """Private typed no-CUDA exercise of the same outer receipt topology."""
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1_lifecycle
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production as v1_production
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import plan as v1_plan
    root, identity = lifecycle.consume(cap); incident = lifecycle.hold_v1_failure(cap.repo_root); published=[]
    training=root/"training"; score=root/"score"
    try:
        attempt=lifecycle.pair(root,"attempt.json",{"schema":"m2_anchored_joint_postfusion_v2_attempt","combined_closure":cap.closure,"synthetic":True});published.append("attempt.json")
        incident_sha=lifecycle.pair(root,"v1_incident.json",incident);published.append("v1_incident.json")
        if fail_after_incident: raise RuntimeError("synthetic inner admission failure")
        regression_sha=lifecycle.pair(root,"strict_digest_regression.json",{"cpu_only":True,"strict_load":True,"cuda_initialized":False,"student_state_sha256":"2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20"});published.append("strict_digest_regression.json")
        class Runtime:
            def source_authority(self): return {"synthetic":True}
            def smoke(self): return {"groups":12}
            def epochs(self): return tuple({"epoch":index} for index in range(1,13))
            def checkpoint_bytes(self,arm): return arm.encode("ascii")
        mapping=v1_lifecycle.closure_map(cap.repo_root)
        nested=v1_lifecycle._issue_for_test(repo_root=cap.repo_root,root=training,reviewed=mapping,digest=v1_lifecycle.closure_sha256(mapping))
        # The private V1 fixture publishes its exact checkpoint/manifest graph.
        v1_production._execute_synthetic_for_test(cap=nested,repo_root=cap.repo_root,runtime=Runtime())
        score.mkdir()
        for name in v1_lifecycle.SCORE_BODIES:
            body={"schema":"synthetic-v1-score","terminal_xor_failure":True} if name=="terminal.json" else {"name":name}
            v1_lifecycle._pair(score,name,body)
        v1_lifecycle.validate_training_topology(training,terminal=True);v1_lifecycle.validate_score_topology(score,terminal=True)
        _need_outer(cap=cap,root_identity=identity,incident=incident,training=training,score=score,terminal=True)
        terminal=lifecycle.pair(root,"terminal.json",{"schema":"m2_anchored_joint_postfusion_v2_terminal","attempt_sha256":attempt,"v1_incident_sha256":incident_sha,"strict_digest_regression_sha256":regression_sha,"nested_v1_training_terminal_sha256":_sha(training/"terminal.json"),"nested_v1_score_terminal_sha256":_sha(score/"terminal.json"),"nested_v1_training_manifest_sha256":_sha(training/"manifest.json"),"terminal_xor_failure":True});published.append("terminal.json")
        lifecycle.validate_outer(root,terminal=True);return {"root":str(root),"terminal_sha256":terminal}
    except Exception as error:
        if "terminal.json" not in published:
            training.mkdir(exist_ok=True);score.mkdir(exist_ok=True)
            lifecycle.pair(root,"failure.json",{"schema":"m2_anchored_joint_postfusion_v2_failure","published_prefix":published,"exception_class":type(error).__name__,"terminal_xor_failure":False})
            lifecycle.validate_outer(root,terminal=False,published=tuple(published))
        raise
