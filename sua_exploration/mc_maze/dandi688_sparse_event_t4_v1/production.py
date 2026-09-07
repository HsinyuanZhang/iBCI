"""Route-owned strict CPU preparation for the production Stage-1/2 executor.

No CUDA/Torch import is permitted at module import or during source-surface
preparation.  The GPU executor consumes this prepared contract only after its
immutable cell attempt has been published.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from . import plan
from .core import array_sha256, apply_reliability_mask, fit_source_normalizer, normalize_columns, require

if TYPE_CHECKING:
    from .descriptors import SparseEventMaterialization


def _attest_loads_before_cuda(repo_root: Path, seed: int, *, environ=None, check_output=None, required_physical_gpu: str | None = None):
    """Attempt-posted, pre-Torch/CUDA parent+teacher and physical GPU admission."""
    import os, subprocess
    from .core import sha256_file
    environ = os.environ if environ is None else environ
    check_output = subprocess.check_output if check_output is None else check_output
    required_environment = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": plan.PRODUCTION_PYTHONPATH,
    }
    require(
        all(environ.get(key) == value for key, value in required_environment.items()),
        "deterministic process environment drift",
    )
    loads={"teacher":sha256_file(Path(repo_root)/plan.TEACHER_RELATIVE),"parent":sha256_file(Path(repo_root)/plan.PARENT_RELATIVE[seed])}
    require(loads["teacher"]==plan.TEACHER_SHA256 and loads["parent"]==plan.PARENT_SHA256[seed],"teacher/parent SHA drift")
    physical=environ.get("CUDA_VISIBLE_DEVICES","")
    if required_physical_gpu is None:
        require(physical in {"0","1"},"requires physical CUDA_VISIBLE_DEVICES 0/1")
    else:
        require(required_physical_gpu == "0" and physical == required_physical_gpu, "requires physical CUDA_VISIBLE_DEVICES=0")
    uuid=check_output(["nvidia-smi","-i",physical,"--query-gpu=uuid","--format=csv,noheader,nounits"],text=True).strip()
    require(uuid in {"GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9","GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"},"unauthorized GPU UUID")
    return {"frozen_load_sha256":loads,"physical_gpu_index":physical,"gpu_uuid":uuid,"deterministic_environment":required_environment}


def _restore_stage2_full_student_trainability(model) -> dict[str, object]:
    """Undo the frozen-parent decoder policy for a matched Stage-2 clone."""
    model._decoder_frozen=False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    decoder=getattr(model,"decoder",None)
    decoupled=getattr(model,"decoupled_transformer",None)
    model.train(True)
    require(model._decoder_frozen is False,"Stage2 decoder frozen flag was not cleared")
    if decoder is not None:
        require(all(parameter.requires_grad for parameter in decoder.parameters()),"Stage2 decoder parameters remain frozen")
        require(decoder.training is True,"Stage2 decoder is not in train mode")
    if decoupled is not None:
        for parameter in decoupled.parameters():
            parameter.requires_grad_(True)
        decoupled.train(True)
        require(all(parameter.requires_grad for parameter in decoupled.parameters()),"Stage2 decoupled parameters remain frozen")
        require(decoupled.training is True,"Stage2 decoupled module is not in train mode")
    return {"decoder_frozen":False,"decoder_present":decoder is not None,"decoder_training":None if decoder is None else bool(decoder.training),"decoupled_present":decoupled is not None,"decoupled_training":None if decoupled is None else bool(decoupled.training)}


def run_stage2_estimator_coordinated(*, parent_student, batches, carrier_for_arm, predict_loss, score, seed: int, optimizer_factory=None):
    """Concrete two-arm paired complete-student loop; device-agnostic for CPU tests/GPU launch."""
    import copy, random, torch
    require(seed in plan.SEEDS, "unsupported paired seed")
    arms = {name: copy.deepcopy(parent_student) for name in ("WHOLE-T4", "POST700-T4")}
    trainability={name:_restore_stage2_full_student_trainability(model) for name,model in arms.items()}
    optimizer_factory = optimizer_factory or torch.optim.Adam
    optimizers = {name: optimizer_factory(model.parameters(), lr=plan.STAGE2_BASE_LEARNING_RATE) for name, model in arms.items()}
    states={name:[] for name in arms}; receipts=[]
    for epoch in range(plan.EPOCHS):
        losses={name:[] for name in arms}; steps={name:0 for name in arms}
        for batch in batches(epoch):
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
            cuda_states = None
            # Do not call torch.cuda.is_available(): that may initialize CUDA in
            # CPU-only tests.  Only touch CUDA RNG after a caller has already
            # initialized the CUDA runtime for a CUDA-resident student.
            if next(iter(next(iter(arms.values())).parameters())).is_cuda and torch.cuda.is_initialized():
                cuda_states = torch.cuda.get_rng_state_all()
            for name, model in arms.items():
                model.train(True)
                random.setstate(python_state); np.random.set_state(numpy_state); torch.random.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
                optimizers[name].zero_grad(set_to_none=True)
                loss=predict_loss(model,batch,carrier_for_arm(name,batch)); require(bool(torch.isfinite(loss)),f"{name}: nonfinite loss")
                loss.backward(); optimizers[name].step(); losses[name].append(float(loss.detach().cpu())); steps[name]+=1
        for name, model in arms.items(): states[name].append({key:value.detach().cpu().clone() for key,value in model.state_dict().items()})
        receipts.append({"epoch_zero_based":epoch,"loss_mean":{name:float(np.mean(losses[name])) for name in arms},"optimizer_steps":steps})
        require(all(value>0 for value in steps.values()) and len(set(steps.values()))==1,"paired estimator steps drift")
    averaged={}
    for name, model in arms.items():
        state={key:(torch.stack([item[key].to(torch.float64) for item in states[name][8:12]]).mean(0).to(value.dtype) if value.is_floating_point() else value) for key,value in states[name][0].items()}
        averaged[name]=copy.deepcopy(model); averaged[name].load_state_dict(state); averaged[name].eval()
    return {"models":arms,"averaged":averaged,"epoch_states":states,"epoch_receipts":receipts,"scores":{name:score(model) for name,model in averaged.items()},"student_trainability":trainability}


def run_stage2_film_coordinated(*, parent_student, film_factory, batches, predict_loss, seed: int, optimizer_factory=None, zero_anchor=None):
    """Device-agnostic five-arm Stage2 FiLM skeleton with full-state averaging."""
    import copy, random, torch
    from mc_maze.dandi688_cp_film_v1.runner import _state_sha
    require(seed in plan.SEEDS,"unsupported paired seed")
    names=("WHOLE-NATIVE","EMPTY","PHASE-R","SE-T4","ROW-SHUFFLE")
    models={name:copy.deepcopy(parent_student) for name in names}
    # Construct once, then clone the exact same zero-anchored initialization.
    # Calling the factory per arm would advance its RNG and silently destroy the
    # paired FiLM initialization contract before the first resident batch.
    head_template=film_factory()
    heads={"WHOLE-NATIVE":None, **{name:copy.deepcopy(head_template) for name in names if name!="WHOLE-NATIVE"}}
    initial_head_state_sha256={name:_state_sha(head) for name,head in heads.items() if head is not None}
    require(len(set(initial_head_state_sha256.values()))==1,"paired FiLM initial state drift")
    opt=optimizer_factory or torch.optim.Adam
    optimizers={}
    trainability={}
    for name,model in models.items():
        trainability[name]=_restore_stage2_full_student_trainability(model)
        groups=[{"params":model.parameters(),"lr":plan.STAGE2_BASE_LEARNING_RATE}]
        if heads[name] is not None: groups.append({"params":heads[name].parameters(),"lr":plan.FILM_HEAD_LEARNING_RATE})
        optimizers[name]=opt(groups)
    states={name:[] for name in names}; head_states={name:[] for name in names if heads[name] is not None}; receipts=[]
    zero_anchor_receipt=None
    for epoch in range(plan.EPOCHS):
        steps={name:0 for name in names}; losses={name:[] for name in names}
        resident_batches=iter(batches(epoch))
        try:
            first_batch=next(resident_batches)
        except StopIteration:
            require(False,"Stage2 FiLM epoch has no resident batches")
        if epoch==0 and zero_anchor is not None:
            # The sentinel must use the same first batch as training while
            # leaving all governing RNG streams and module modes untouched.
            saved_python,saved_numpy,saved_torch=random.getstate(),np.random.get_state(),torch.random.get_rng_state()
            saved_cuda=None
            if next(iter(next(iter(models.values())).parameters())).is_cuda and torch.cuda.is_initialized():
                saved_cuda=torch.cuda.get_rng_state_all()
            model_modes={name:model.training for name,model in models.items()}
            head_modes={name:head.training for name,head in heads.items() if head is not None}
            try:
                for model in models.values(): model.eval()
                for head in heads.values():
                    if head is not None: head.eval()
                with torch.no_grad():
                    zero_anchor_receipt=zero_anchor(models,heads,first_batch)
                require(isinstance(zero_anchor_receipt,dict) and zero_anchor_receipt.get("passed") is True,"Stage2 FiLM zero-anchor sentinel failed")
            finally:
                random.setstate(saved_python);np.random.set_state(saved_numpy);torch.random.set_rng_state(saved_torch)
                if saved_cuda is not None: torch.cuda.set_rng_state_all(saved_cuda)
                for name,model in models.items(): model.train(model_modes[name])
                for name,head in heads.items():
                    if head is not None: head.train(head_modes[name])
        for batch in __import__("itertools").chain((first_batch,),resident_batches):
            py,np_state,tr=random.getstate(),np.random.get_state(),torch.random.get_rng_state(); cuda=None
            if next(iter(next(iter(models.values())).parameters())).is_cuda and torch.cuda.is_initialized(): cuda=torch.cuda.get_rng_state_all()
            for name in names:
                random.setstate(py);np.random.set_state(np_state);torch.random.set_rng_state(tr)
                if cuda is not None:torch.cuda.set_rng_state_all(cuda)
                models[name].train(True)
                if heads[name] is not None: heads[name].train(True)
                optimizers[name].zero_grad(set_to_none=True); loss=predict_loss(models[name],heads[name],name,batch); require(bool(torch.isfinite(loss)),f"{name}: nonfinite"); loss.backward(); optimizers[name].step();steps[name]+=1;losses[name].append(float(loss.detach()))
        for name in names:
            states[name].append({k:v.detach().cpu().clone() for k,v in models[name].state_dict().items()})
            if heads[name] is not None:head_states[name].append({k:v.detach().cpu().clone() for k,v in heads[name].state_dict().items()})
        receipts.append({"epoch_zero_based":epoch,"optimizer_steps":steps,"loss_mean":{n:float(np.mean(losses[n])) for n in names}})
        require(all(value>0 for value in steps.values()) and len(set(steps.values()))==1,"paired FiLM steps drift")
    avg_models={};avg_heads={}
    for name in names:
        avg_models[name]=copy.deepcopy(models[name]); avg_models[name].load_state_dict({k:(torch.stack([s[k].to(torch.float64) for s in states[name][8:12]]).mean(0).to(v.dtype) if v.is_floating_point() else v) for k,v in states[name][0].items()}); avg_models[name].eval()
        if heads[name] is not None:
            avg_heads[name]=copy.deepcopy(heads[name]);avg_heads[name].load_state_dict({k:torch.stack([s[k].to(torch.float64) for s in head_states[name][8:12]]).mean(0).to(v.dtype) for k,v in head_states[name][0].items()}); avg_heads[name].eval()
    return {"models":models,"heads":heads,"averaged":avg_models,"averaged_heads":avg_heads,"epoch_states":states,"head_epoch_states":head_states,"epoch_receipts":receipts,"initial_head_state_sha256":initial_head_state_sha256,"head_template_factory_calls":1,"zero_anchor":zero_anchor_receipt,"student_trainability":trainability}


def _score_models_q50(surface, averaged_models, carriers, device):
    """Score averaged estimator students on all six validation Q50 surfaces."""
    import torch
    out={name:{} for name in averaged_models}
    for session,row in surface.sessions.items():
        if row.split != "val": continue
        target_parts=[]; predictions={name:[] for name in averaged_models}
        for offset in range(0,row.q50_starts.size,256):
            starts=row.q50_starts[offset:offset+256]; neural=torch.from_numpy(np.stack([row.record.neural[int(s):int(s)+50] for s in starts]).astype(np.float32)).to(device); target_parts.append(np.stack([row.record.behavior[int(s)+49] for s in starts]).astype(np.float32))
            for name,model in averaged_models.items():
                carrier=torch.from_numpy(carriers[name](row)).unsqueeze(0).to(device); cal=torch.from_numpy(row.record.calib_trials.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad(): mean=model.id_encoder.pre_pool(cal.permute(0,1,3,2)).mean(1); identity=model.id_encoder.post_pool(torch.cat((mean,carrier),-1)); predictions[name].append((model.decode_with_identity(neural,identity.expand(neural.shape[0],-1,-1))[:,-1]/5).cpu().numpy())
        target=np.concatenate(target_parts); total=np.square(target-target.mean(0)).sum()
        for name,parts in predictions.items():
            pred=np.concatenate(parts); out[name][session]={"r2":float(1-np.square(target-pred).sum()/total),"prediction_sha256":array_sha256(pred),"target_sha256":array_sha256(target),"query_sha256":array_sha256(row.q50_starts)}
    require(all(len(values)==6 for values in out.values()),"six validation sessions required")
    return out


def _score_stage2_film_q50(surface, averaged_models, averaged_heads, device, seed: int):
    """Score the five averaged Stage-2 FiLM arms on the exact validation Q50.

    This deliberately does not share a cached identity between arms.  Each
    complete student must regenerate its own M30 activity summary and its
    WHOLE-M10/parent-normalized carrier identity before the arm-specific FiLM
    profile is applied.  That makes the evidence valid even when the complete
    students have diverged during their paired base training.
    """
    import torch

    from .core import film_identity, phase_r_profile, row_shuffle_profile

    names = ("WHOLE-NATIVE", "EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
    require(set(averaged_models) == set(names), "Stage2 FiLM model arms drift")
    require(set(averaged_heads) == set(names[1:]), "Stage2 FiLM head arms drift")
    for name, model in averaged_models.items():
        model.eval()
        if name != "WHOLE-NATIVE":
            averaged_heads[name].eval()

    scores = {name: {} for name in names}
    for session_id, row in surface.sessions.items():
        if row.split != "val":
            continue
        carrier_np = np.asarray(row.whole_m10_parentnorm, dtype=np.float32)
        profile_np = np.asarray(row.profile_m10, dtype=np.float32)
        phase_np = phase_r_profile(profile_np, surface.reliability_mask)
        shuffled_np, _ = row_shuffle_profile(
            profile_np,
            session_id=session_id,
            view=surface.signal_view,
            training_seed=seed,
        )
        target_parts: list[np.ndarray] = []
        predictions = {name: [] for name in names}
        identities = {}
        # Recompute this state independently for every complete student.  Do
        # not reuse the Stage-1 parent cache: all Stage-2 base parameters are
        # trainable and potentially arm-specific after epoch zero.
        with torch.no_grad():
            for name, model in averaged_models.items():
                calibration = torch.from_numpy(np.asarray(row.record.calib_trials, dtype=np.float32)).unsqueeze(0).to(device)
                carrier = torch.from_numpy(carrier_np).unsqueeze(0).to(device)
                mean = model.id_encoder.pre_pool(calibration.permute(0, 1, 3, 2)).mean(1)
                if name == "WHOLE-NATIVE":
                    identities[name] = model.id_encoder.post_pool(torch.cat((mean, carrier), dim=-1))
                    continue
                profile = {
                    "EMPTY": np.zeros_like(profile_np),
                    "PHASE-R": phase_np,
                    "SE-T4": profile_np,
                    "ROW-SHUFFLE": shuffled_np,
                }[name]
                profile_tensor = torch.from_numpy(profile).unsqueeze(0).to(device=device, dtype=carrier.dtype)
                identities[name], _ = film_identity(
                    model.id_encoder,
                    mean,
                    carrier,
                    profile_tensor,
                    averaged_heads[name],
                )
            for offset in range(0, row.q50_starts.size, 256):
                starts = row.q50_starts[offset : offset + 256]
                neural = torch.from_numpy(
                    np.stack([row.record.neural[int(start) : int(start) + 50] for start in starts]).astype(np.float32)
                ).to(device)
                target_parts.append(
                    np.stack([row.record.behavior[int(start) + 49] for start in starts]).astype(np.float32)
                )
                for name, model in averaged_models.items():
                    prediction = model.decode_with_identity(
                        neural,
                        identities[name].expand(neural.shape[0], -1, -1),
                    )[:, -1] / 5.0
                    predictions[name].append(prediction.cpu().numpy())
        target = np.concatenate(target_parts)
        total = float(np.square(target - target.mean(axis=0)).sum())
        require(total > 0.0, f"{session_id}: degenerate validation target")
        for name, values in predictions.items():
            prediction = np.concatenate(values)
            scores[name][session_id] = {
                "r2": float(1.0 - np.square(target - prediction).sum() / total),
                "prediction_sha256": array_sha256(prediction),
                "target_sha256": array_sha256(target),
                "query_sha256": array_sha256(row.q50_starts),
                "window_count": int(target.shape[0]),
            }
    require(all(len(values) == 6 for values in scores.values()), "six validation sessions required")
    return scores


def _all_q50_training_batches(surface, seed: int, epoch: int):
    train=sorted(name for name,row in surface.sessions.items() if row.split=="train")
    require(len(train)==27,"requires exact 27 train sessions")
    rng=np.random.Generator(np.random.PCG64(int.from_bytes(__import__("hashlib").sha256(f"{plan.ROUTE_NAME}:stage2-q50:{seed}:{epoch}".encode()).digest()[:8],"little")))
    for index in rng.permutation(len(train)):
        name=train[int(index)]; row=surface.sessions[name]; starts=np.asarray(row.q50_starts,dtype=np.int64)[rng.permutation(len(row.q50_starts))]
        for offset in range(0,len(starts),32): yield name,row,np.ascontiguousarray(starts[offset:offset+32])


def _publish_stage2_full_state_artifacts(result_root: Path, result, seed: int, *, artifact_prefix: str = "stage2_estimator"):
    """Publish immutable two-arm full epoch-state and fixed-average torch artifacts."""
    import io, torch
    from mc_maze.dandi688_cp_film_v1.runner import _publish_bytes, _state_sha
    output={}
    for arm in ("WHOLE-T4","POST700-T4"):
        states=result["epoch_states"][arm]; require(len(states)==plan.EPOCHS,"requires 12 epoch states")
        buffer=io.BytesIO(); torch.save({"seed":seed,"arm":arm,"epoch_states":states,"average_epochs_zero_based":[8,9,10,11],"average_state_dict":result["averaged"][arm].state_dict()},buffer)
        sha=_publish_bytes(Path(result_root),f"{artifact_prefix}_{arm.lower()}.pt",buffer.getvalue())
        output[arm]={"artifact_sha256":sha,"per_epoch_state_sha256":[array_sha256(np.concatenate([value.detach().cpu().numpy().reshape(-1) for value in state.values() if value.is_floating_point()])) for state in states],"average_state_sha256":_state_sha(result["averaged"][arm])}
    return output


def _publish_stage2_film_full_state_artifacts(result_root: Path, result, seed: int, *, artifact_prefix: str = "stage2_film"):
    import io, torch
    from mc_maze.dandi688_cp_film_v1.runner import _publish_bytes, _state_sha
    output={}
    for arm, model in result["averaged"].items():
        base=result["epoch_states"][arm]; require(len(base)==12,"requires 12 base states")
        heads=result["head_epoch_states"].get(arm)
        buffer=io.BytesIO(); torch.save({"seed":seed,"arm":arm,"base_epoch_states":base,"base_average":model.state_dict(),"head_epoch_states":heads,"head_average":None if arm not in result["averaged_heads"] else result["averaged_heads"][arm].state_dict(),"average_epochs_zero_based":[8,9,10,11]},buffer)
        sha=_publish_bytes(Path(result_root),f"{artifact_prefix}_{arm.lower()}.pt",buffer.getvalue())
        output[arm]={"artifact_sha256":sha,"base_epoch_state_sha256":[array_sha256(np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in s.values() if v.is_floating_point()])) for s in base],"base_average_state_sha256":_state_sha(model),"head_epoch_state_sha256":None if heads is None else [array_sha256(np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in s.values() if v.is_floating_point()])) for s in heads],"head_average_state_sha256":None if arm not in result["averaged_heads"] else _state_sha(result["averaged_heads"][arm])}
    return output


def _admit_stage2_film_opening(repo_root: Path, *, stage0_admission=None) -> dict[str, object]:
    """Fail closed on the immutable three-seed Stage-1-v2 FiLM opening proof.

    The root-owned aggregate must be ``stage1_v3/aggregate.json`` with its
    immutable sidecar.  Its minimal opening schema is deliberately explicit:
    ``stage2_film_opening`` has ``status=OPEN``, strictly positive semantic and
    attachment means, and at least four positive semantic validation sessions.
    It must bind all three Stage-1-v2 terminal body digests by decimal seed.
    """
    import hashlib
    import json

    from .admission import _held_bytes, admit_stage0_graph

    repo_root=Path(repo_root).resolve()
    stage0=dict(admit_stage0_graph(repo_root) if stage0_admission is None else stage0_admission)
    require(stage0.get("film_route")=="OPEN","Stage-0 FiLM admission closed")
    root=repo_root / plan.STAGE1_SUCCESSOR_RELATIVE
    aggregate_name="aggregate.json"
    aggregate_bytes=_held_bytes(root / aggregate_name)
    aggregate_sha=hashlib.sha256(aggregate_bytes).hexdigest()
    sidecar=_held_bytes(root / f"{aggregate_name}.sha256").decode("ascii").strip().split()
    require(sidecar==[aggregate_sha,aggregate_name],"Stage1-v3 aggregate sidecar drift")
    aggregate=json.loads(aggregate_bytes)
    require(isinstance(aggregate,dict),"Stage1-v3 aggregate is not an object")
    require(aggregate.get("status")=="PASS" and aggregate.get("route")==plan.ROUTE_NAME,"Stage1-v3 aggregate status/route drift")
    require(aggregate.get("stage")=="stage1_v3_aggregate","Stage1-v3 aggregate stage drift")
    require(aggregate.get("seeds")==list(plan.SEEDS),"Stage1-v3 aggregate seed roster drift")
    opening=aggregate.get("stage2_film_opening")
    require(isinstance(opening,dict),"Stage1-v3 FiLM opening missing")
    require(opening.get("status")=="OPEN","Stage1-v3 FiLM opening is closed")
    require(float(opening.get("semantic_mean",float("nan")))>0.0,"Stage1-v3 semantic mean does not open FiLM")
    require(float(opening.get("attachment_mean",float("nan")))>0.0,"Stage1-v3 attachment mean does not open FiLM")
    require(int(opening.get("semantic_positive_sessions",-1))>=4,"Stage1-v3 semantic sign count does not open FiLM")
    declared=aggregate.get("stage1_terminal_sha256")
    require(isinstance(declared,dict) and set(declared)=={str(seed) for seed in plan.SEEDS},"Stage1-v3 terminal digest roster drift")
    terminals={}
    for seed in plan.SEEDS:
        leaf=root / "sua" / f"seed{seed}" / "terminal.json"
        body=_held_bytes(leaf)
        digest=hashlib.sha256(body).hexdigest()
        terminal_sidecar=_held_bytes(leaf.with_name("terminal.json.sha256")).decode("ascii").strip().split()
        require(terminal_sidecar==[digest,"terminal.json"],f"Stage1-v3 seed{seed} terminal sidecar drift")
        terminal=json.loads(body)
        require(isinstance(terminal,dict) and terminal.get("status")=="PASS",f"Stage1-v3 seed{seed} did not pass")
        require(declared[str(seed)]==digest,f"Stage1-v3 seed{seed} terminal binding drift")
        terminals[str(seed)]=digest
    return {"stage0_admission":stage0,"stage1_v3_aggregate_relative":plan.STAGE1_SUCCESSOR_RELATIVE + "/aggregate.json","stage1_v3_aggregate_sha256":aggregate_sha,"stage1_terminal_sha256":terminals,"stage2_film_opening":opening}


@dataclass(frozen=True)
class PreparedSession:
    session_id: str
    split: str
    record: object
    q50_starts: np.ndarray
    parent_m30_t4: np.ndarray
    whole_m10_raw: np.ndarray
    post700_m10_raw: np.ndarray
    whole_m10_parentnorm: np.ndarray
    whole_m10_t4: np.ndarray
    post700_m10_t4: np.ndarray
    profile_m10: np.ndarray
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class PreparedSurface:
    sessions: Mapping[str, PreparedSession]
    normalizers: Mapping[str, Mapping[str, np.ndarray]]
    parent_m30_normalizer: Mapping[str, np.ndarray]
    reliability_mask: tuple[bool, bool, bool, bool]
    signal_view: str


def _q50_starts(trials) -> np.ndarray:
    require(len(trials) > plan.QUERY_START_TRIAL, "insufficient rewarded trials for Q50")
    starts = np.concatenate([np.arange(int(row["start"]), int(row["stop"]) - plan.WINDOW_SIZE_BINS + 1, dtype=np.int64) for row in trials[plan.QUERY_START_TRIAL:]])
    require(starts.size > 0, "empty Q50 surface")
    return np.ascontiguousarray(starts)


def prepare_source_surface(
    repo_root: Path,
    *,
    signal_view: str,
    reliability_mask: tuple[bool, bool, bool, bool],
    datamodule_factory: Callable[[Path, str], object] | None = None,
    descriptor_materializer: Callable[..., "SparseEventMaterialization"] | None = None,
    trial_lister=None,
) -> PreparedSurface:
    """Build M10 carrier/profile rows and Q50 starts on exact 27/6 source roster.

    `record.side_features` is retained as the parent M30-normalized T4 only;
    new descriptors are route-owned and never call the dense CP materializer.
    """
    require(signal_view in {"sua", "pseudo_mua"}, "unsupported view")
    if datamodule_factory is None:
        if signal_view == "sua":
            from mc_maze.dandi688_cp_film_v1.data import prepare_datamodule
            datamodule_factory = lambda root, _view: prepare_datamodule(root)
        else:
            # The CP helper freezes ``signal_view='sua'``.  Stage3 must build
            # the same strict 27/6 source contract at electrode-pooled pseudo-
            # MUA resolution instead of silently reusing SUA carrier records.
            from mc_maze.dandi688_cp_film_v1 import plan as cp_plan
            from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
            def datamodule_factory(root, view):
                dm = Dandi688MultiSessionDataModule(
                    data_dir=str(root / cp_plan.DATA_RELATIVE), task="CO",
                    split_counts=(cp_plan.TRAIN_SESSIONS, cp_plan.VALIDATION_SESSIONS, cp_plan.FORMAL_TEST_SESSIONS),
                    batch_size=cp_plan.BATCH_SIZE, window_size=cp_plan.WINDOW,
                    calibration_n_trials=cp_plan.ACTIVITY_SUPPORT, max_trial_length=cp_plan.TRIAL_LENGTH,
                    bin_size_ms=cp_plan.BIN_MS, num_workers=0, pin_memory=False,
                    random_calibration=False, seed=42, max_units_exclusive=100,
                    cache_dir=str(root / cp_plan.CACHE_RELATIVE), signal_view=view,
                    side_feature_group="t4", side_feature_pool_size=cp_plan.T4_SUPPORT,
                    train_val_manifest_path=str(root / cp_plan.MANIFEST_RELATIVE),
                )
                dm.setup("fit")
                return dm
    if descriptor_materializer is None:
        from .descriptors import materialize_sparse_event_t4

        descriptor_materializer = materialize_sparse_event_t4
    dm = datamodule_factory(Path(repo_root), signal_view)
    require(getattr(dm, "test_dataset", None) is None, "formal-test dataset was constructed")
    splits = getattr(dm, "session_splits")
    require(len(splits["train"]) == 27 and len(splits["val"]) == 6 and len(splits["test"]) == 6, "strict 27/6/6 roster drift")
    parent_stats = getattr(dm, "_side_feature_stats", None)
    require(parent_stats is not None and len(parent_stats) == 2, "parent M30 normalizer unavailable")
    parent_m30_normalizer = {"mean": np.asarray(parent_stats[0], dtype=np.float64), "scale": np.asarray(parent_stats[1], dtype=np.float64)}
    raw: dict[str, tuple[str, object, np.ndarray, Any]] = {}
    if trial_lister is None:
        from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
        trial_lister = list_datamodule_rewarded_trials
    train_available = set(dm.train_dataset.window_indices)
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset)):
        require(dataset is not None, f"missing {split} dataset")
        for session_id, record in dataset.sessions.items():
            path = Path(repo_root) / plan.DATA_RELATIVE / f"{session_id}_behavior+ecephys.nwb"
            descriptor = descriptor_materializer(path, signal_view=signal_view, namespace="candidate")
            trials = trial_lister(path, bin_size_ms=plan.BIN_SIZE_MS, window_size=plan.WINDOW_SIZE_BINS, trial_result_filter=plan.REWARDED_RESULT)
            q50 = _q50_starts(trials)
            available = train_available if split == "train" else {(session_id, int(start)) for start in record.valid_starts}
            require(all((session_id, int(start)) in available for start in q50), f"{session_id}: Q50 unavailable in strict dataset")
            raw[session_id] = (split, record, q50, descriptor)
    require(sum(row[0] == "train" for row in raw.values()) == 27 and sum(row[0] == "val" for row in raw.values()) == 6, "prepared roster drift")
    train = [row[3] for row in raw.values() if row[0] == "train"]
    normalizers = {
        "whole_m10": fit_source_normalizer([row.whole_t4 for row in train]),
        "post700_m10": fit_source_normalizer([row.post700_t4 for row in train]),
        "profile_m10": fit_source_normalizer([row.raw_profile for row in train]),
    }
    prepared: dict[str, PreparedSession] = {}
    for session_id, (split, record, q50, descriptor) in raw.items():
        profile = apply_reliability_mask(normalize_columns(descriptor.raw_profile, **normalizers["profile_m10"]), reliability_mask)
        whole_parent = normalize_columns(descriptor.whole_t4, **parent_m30_normalizer)
        prepared[session_id] = PreparedSession(session_id, split, record, q50, np.ascontiguousarray(record.side_features, dtype=np.float32), descriptor.whole_t4.copy(), descriptor.post700_t4.copy(), whole_parent, normalize_columns(descriptor.whole_t4, **normalizers["whole_m10"]), normalize_columns(descriptor.post700_t4, **normalizers["post700_m10"]), profile, {**descriptor.receipt, "q50_starts_sha256": array_sha256(q50), "parent_m30_t4_sha256": array_sha256(record.side_features), "whole_m10_parentnorm_sha256": array_sha256(whole_parent), "whole_m10_t4_sha256": array_sha256(normalize_columns(descriptor.whole_t4, **normalizers["whole_m10"])), "post700_m10_t4_sha256": array_sha256(normalize_columns(descriptor.post700_t4, **normalizers["post700_m10"])), "profile_m10_sha256": array_sha256(profile)})
    return PreparedSurface(prepared, normalizers, parent_m30_normalizer, tuple(bool(value) for value in reliability_mask), signal_view)


def execute_gpu_cell(repo_root: Path, cell, result_root: Path):
    """Compatibility alias: one physical root is coordinated per stage/seed."""
    return execute_seed_stage(repo_root, stage=cell.stage, seed=cell.seed, view=cell.signal_view, result_root=result_root)


def _execute_stage2_film(repo_root: Path, *, seed: int, view: str, result_root: Path, stage_label: str = "stage2", branch_admission=None) -> dict[str, object]:
    """Run a conditional five-arm matched FiLM matrix in one seed root."""
    from .lifecycle import begin_attempt, close_failure, close_terminal, publish_immutable_json

    if branch_admission is None:
        opening_admission=_admit_stage2_film_opening(repo_root)
        mask=tuple(bool(value) for value in opening_admission["stage0_admission"]["reliability_mask"])
        admission_field="stage1_v2_opening_admission"
    else:
        opening_admission=dict(branch_admission)
        mask=tuple(bool(value) for value in opening_admission["branch_gate"]["reliability_mask"])
        admission_field="stage2_aggregate_admission"
    attempt=begin_attempt(result_root,{
        "route":plan.ROUTE_NAME,
        "stage":f"{stage_label}_film",
        "seed":seed,
        "view":view,
        admission_field:opening_admission,
        "gpu_initialized":False,
        "decoder_opened":False,
        "formal_test_opened":False,
    })
    try:
        # This load/GPU attestation is intentionally before every torch.cuda
        # call, model/checkpoint read, DataModule construction, or NWB open.
        launch_receipt = (
            _attest_loads_before_cuda(repo_root, seed, required_physical_gpu="0")
            if stage_label in {"stage2_v3", "stage3"}
            else _attest_loads_before_cuda(repo_root, seed)
        )
        import torch
        from mc_maze.dandi688_cp_film_v1 import runner as base_runner
        from .core import build_film, film_identity, phase_r_profile, row_shuffle_profile

        surface=prepare_source_surface(repo_root,signal_view=view,reliability_mask=mask)
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"requires exactly one isolated CUDA device")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        device=torch.device("cuda:0")
        parent=base_runner._prepare_student(repo_root,seed,device)
        for parameter in parent.parameters():
            parameter.requires_grad_(True)

        def batches(epoch):
            yield from _all_q50_training_batches(surface,seed,epoch)

        def predict_loss(model,head,arm,batch):
            session_id,row,starts=batch
            neural=torch.from_numpy(np.stack([row.record.neural[int(start):int(start)+50] for start in starts]).astype(np.float32)).to(device)
            target=torch.from_numpy(np.stack([row.record.behavior[int(start)+49] for start in starts]).astype(np.float32)).to(device)
            calibration=torch.from_numpy(np.asarray(row.record.calib_trials,dtype=np.float32)).unsqueeze(0).to(device)
            carrier=torch.from_numpy(np.asarray(row.whole_m10_parentnorm,dtype=np.float32)).unsqueeze(0).to(device)
            mean=model.id_encoder.pre_pool(calibration.permute(0,1,3,2)).mean(1)
            if arm=="WHOLE-NATIVE":
                identity=model.id_encoder.post_pool(torch.cat((mean,carrier),dim=-1))
            else:
                profile={
                    "EMPTY":np.zeros_like(row.profile_m10),
                    "PHASE-R":phase_r_profile(row.profile_m10,surface.reliability_mask),
                    "SE-T4":row.profile_m10,
                    "ROW-SHUFFLE":row_shuffle_profile(row.profile_m10,session_id=session_id,view=view,training_seed=seed)[0],
                }[arm]
                identity,_=film_identity(
                    model.id_encoder,
                    mean,
                    carrier,
                    torch.from_numpy(np.asarray(profile,dtype=np.float32)).unsqueeze(0).to(device),
                    head,
                    direct_zero=False,
                )
            prediction=model.decode_with_identity(neural,identity.expand(neural.shape[0],-1,-1))[:,-1]/5.0
            return torch.mean((prediction-target)**2)

        def zero_anchor(models,heads,batch):
            """Real pre-step parity check on the first governing resident batch."""
            session_id,row,starts=batch
            neural=torch.from_numpy(np.stack([row.record.neural[int(start):int(start)+50] for start in starts]).astype(np.float32)).to(device)
            calibration=torch.from_numpy(np.asarray(row.record.calib_trials,dtype=np.float32)).unsqueeze(0).to(device)
            carrier=torch.from_numpy(np.asarray(row.whole_m10_parentnorm,dtype=np.float32)).unsqueeze(0).to(device)
            native_model=models["WHOLE-NATIVE"]
            native_mean=native_model.id_encoder.pre_pool(calibration.permute(0,1,3,2)).mean(1)
            native_identity=native_model.id_encoder.post_pool(torch.cat((native_mean,carrier),dim=-1))
            native_prediction=native_model.decode_with_identity(neural,native_identity.expand(neural.shape[0],-1,-1))[:,-1]/5.0
            profiles={
                "EMPTY":np.zeros_like(row.profile_m10),
                "PHASE-R":phase_r_profile(row.profile_m10,surface.reliability_mask),
                "SE-T4":row.profile_m10,
                "ROW-SHUFFLE":row_shuffle_profile(row.profile_m10,session_id=session_id,view=view,training_seed=seed)[0],
            }
            identity_bitwise={}; prediction_bitwise={}; direct_native_branch={}
            for arm,profile in profiles.items():
                model=models[arm]
                mean=model.id_encoder.pre_pool(calibration.permute(0,1,3,2)).mean(1)
                identity,direct=film_identity(
                    model.id_encoder,
                    mean,
                    carrier,
                    torch.from_numpy(np.asarray(profile,dtype=np.float32)).unsqueeze(0).to(device),
                    heads[arm],
                    direct_zero=True,
                )
                prediction=model.decode_with_identity(neural,identity.expand(neural.shape[0],-1,-1))[:,-1]/5.0
                direct_native_branch[arm]=bool(direct["direct_native_branch"])
                identity_bitwise[arm]=bool(torch.equal(identity,native_identity))
                prediction_bitwise[arm]=bool(torch.equal(prediction,native_prediction))
            passed=bool(
                all(direct_native_branch.values())
                and all(identity_bitwise.values())
                and all(prediction_bitwise.values())
            )
            return {
                "passed":passed,
                "session_id":session_id,
                "query_start_sha256":array_sha256(np.asarray(starts,dtype=np.int64)),
                "window_count":int(len(starts)),
                "identity_bitwise":identity_bitwise,
                "prediction_bitwise":prediction_bitwise,
                "direct_native_branch":direct_native_branch,
            }

        result=run_stage2_film_coordinated(
            parent_student=parent,
            film_factory=lambda:build_film().to(device),
            batches=batches,
            predict_loss=predict_loss,
            seed=seed,
            zero_anchor=zero_anchor,
        )
        scores=_score_stage2_film_q50(surface,result["averaged"],result["averaged_heads"],device,seed)
        artifacts=_publish_stage2_film_full_state_artifacts(result_root,result,seed,artifact_prefix=f"{stage_label}_film")
        training_sha=publish_immutable_json(result_root,"training.json",{
            "attempt_sha256":attempt,
            "launch_receipt":launch_receipt,
            admission_field:opening_admission,
            "epochs":plan.EPOCHS,
            "arms":list(plan.STAGE2_FILM_ARMS),
            "average_epochs_zero_based":list(plan.AVERAGE_EPOCHS_ZERO_BASED),
            "epoch_receipts":result["epoch_receipts"],
            "initial_head_state_sha256":result["initial_head_state_sha256"],
            "initial_head_states_identical":True,
            "head_template_factory_calls":result["head_template_factory_calls"],
            "zero_anchor":result["zero_anchor"],
            "student_trainability":result["student_trainability"],
            "base_trainable_parameter_count":{name:sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) for name,model in result["models"].items()},
            "full_state_artifacts":artifacts,
        })
        score_sha=publish_immutable_json(result_root,"score.json",{
            "attempt_sha256":attempt,
            "launch_receipt":launch_receipt,
            admission_field:opening_admission,
            "per_session":scores,
        })
        close_terminal(result_root,attempt_sha256=attempt,payload={
            "training_sha256":training_sha,
            "score_sha256":score_sha,
            "full_state_artifacts":artifacts,
            admission_field:opening_admission,
            "launch_receipt":launch_receipt,
            "zero_anchor":result["zero_anchor"],
            "status_detail":f"{stage_label.upper()}_FILM_SEED_COMPLETE__THREE_SEED_AGGREGATE_OPEN",
        })
        return {"attempt_sha256":attempt,"training_sha256":training_sha,"score_sha256":score_sha}
    except BaseException as error:
        close_failure(result_root,attempt_sha256=attempt,error=error)
        raise


def _execute_stage3_estimator(repo_root: Path, *, seed: int, view: str, result_root: Path, stage2_admission, stage_label: str = "stage3") -> dict[str, object]:
    """Run the held-admitted pseudo-MUA two-arm matched estimator matrix."""
    from .lifecycle import begin_attempt, close_failure, close_terminal, publish_immutable_json

    admission=dict(stage2_admission)
    mask=tuple(bool(value) for value in admission["branch_gate"]["reliability_mask"])
    attempt=begin_attempt(result_root,{
        "route":plan.ROUTE_NAME,"stage":f"{stage_label}_estimator","seed":seed,"view":view,
        "stage2_aggregate_admission":admission,"gpu_initialized":False,"decoder_opened":False,"formal_test_opened":False,
    })
    try:
        launch_receipt = (
            _attest_loads_before_cuda(repo_root, seed, required_physical_gpu="0")
            if stage_label in {"stage2_v3", "stage3"}
            else _attest_loads_before_cuda(repo_root, seed)
        )
        import torch
        from mc_maze.dandi688_cp_film_v1 import runner as base

        surface=prepare_source_surface(repo_root,signal_view=view,reliability_mask=mask)
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"requires exactly one isolated CUDA device")
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
        device=torch.device("cuda:0")
        parent=base._prepare_student(repo_root,seed,device)

        def batches(epoch):
            yield from _all_q50_training_batches(surface,seed,epoch)

        def carrier(arm,batch):
            row=batch[1]
            value=row.whole_m10_t4 if arm=="WHOLE-T4" else row.post700_m10_t4
            return torch.from_numpy(value).unsqueeze(0).to(device)

        def loss(model,batch,carrier_value):
            _session,row,starts=batch
            neural=torch.from_numpy(np.stack([row.record.neural[int(start):int(start)+50] for start in starts]).astype(np.float32)).to(device)
            target=torch.from_numpy(np.stack([row.record.behavior[int(start)+49] for start in starts]).astype(np.float32)).to(device)
            calibration=torch.from_numpy(np.asarray(row.record.calib_trials,dtype=np.float32)).unsqueeze(0).to(device)
            mean=model.id_encoder.pre_pool(calibration.permute(0,1,3,2)).mean(1)
            identity=model.id_encoder.post_pool(torch.cat((mean,carrier_value),dim=-1))
            prediction=model.decode_with_identity(neural,identity.expand(neural.shape[0],-1,-1))[:,-1]/5.0
            return torch.mean((prediction-target)**2)

        result=run_stage2_estimator_coordinated(parent_student=parent,batches=batches,carrier_for_arm=carrier,predict_loss=loss,score=lambda _model:None,seed=seed)
        scores=_score_models_q50(surface,result["averaged"],{"WHOLE-T4":lambda row:row.whole_m10_t4,"POST700-T4":lambda row:row.post700_m10_t4},device)
        artifacts=_publish_stage2_full_state_artifacts(result_root,result,seed,artifact_prefix=f"{stage_label}_estimator")
        training_sha=publish_immutable_json(result_root,"training.json",{
            "attempt_sha256":attempt,"launch_receipt":launch_receipt,"stage2_aggregate_admission":admission,
            "epochs":plan.EPOCHS,"average_epochs_zero_based":list(plan.AVERAGE_EPOCHS_ZERO_BASED),
            "epoch_receipts":result["epoch_receipts"],"student_trainability":result["student_trainability"],"full_state_artifacts":artifacts,
        })
        score_sha=publish_immutable_json(result_root,"score.json",{
            "attempt_sha256":attempt,"launch_receipt":launch_receipt,"stage2_aggregate_admission":admission,"per_session":scores,
        })
        close_terminal(result_root,attempt_sha256=attempt,payload={
            "training_sha256":training_sha,"score_sha256":score_sha,"full_state_artifacts":artifacts,
            "stage2_aggregate_admission":admission,"launch_receipt":launch_receipt,
            "status_detail":f"{stage_label.upper()}_ESTIMATOR_SEED_COMPLETE__THREE_SEED_AGGREGATE_OPEN",
        })
        return {"attempt_sha256":attempt,"training_sha256":training_sha,"score_sha256":score_sha}
    except BaseException as error:
        close_failure(result_root,attempt_sha256=attempt,error=error)
        raise


def execute_seed_stage(repo_root: Path, *, stage: int, seed: int, view: str, result_root: Path) -> dict[str, object]:
    """Concrete coordinated Stage-1 SUA executor (all matched arms in one root).

    Stage-2 is deliberately rejected until its decoder-trainable loop is added;
    it cannot silently reuse the frozen-parent Stage-1 loop.
    """
    if stage == 3:
        require(view == "pseudo_mua" and seed in plan.SEEDS, "Stage-3 supports pseudo-MUA paired seeds only")
        from .stage2_aggregate import admit_stage3_branch
        repo_root = Path(repo_root).resolve()
        result_root = Path(result_root).resolve()
        estimator_root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage3" / "estimator" / view / f"seed{seed}"
        film_root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage3" / "film" / view / f"seed{seed}"
        if result_root == estimator_root:
            admission = admit_stage3_branch(repo_root, "estimator")
            return _execute_stage3_estimator(repo_root,seed=seed,view=view,result_root=result_root,stage2_admission=admission)
        if result_root == film_root:
            admission = admit_stage3_branch(repo_root, "film")
            return _execute_stage2_film(repo_root,seed=seed,view=view,result_root=result_root,stage_label="stage3",branch_admission=admission)
        require(False, "Stage3 result root is neither canonical estimator nor canonical FiLM root")
    if stage == 2:
        require(view == "sua" and seed in plan.SEEDS, "Stage-2 estimator supports SUA paired seeds")
        repo_root = Path(repo_root).resolve()
        result_root = Path(result_root).resolve()
        from .stage2_aggregate import admit_stage2_v2_successor, admit_stage2_v3_successor
        v3_estimator_root = repo_root / plan.STAGE2_V3_RELATIVE / "estimator" / view / f"seed{seed}"
        v2_estimator_root = repo_root / plan.STAGE2_V2_RELATIVE / "estimator" / view / f"seed{seed}"
        v2_film_root = repo_root / plan.STAGE2_V2_RELATIVE / "film" / view / f"seed{seed}"
        if result_root == v3_estimator_root:
            require(seed == 44, "Stage2-v3 reruns only interrupted estimator seed44")
            successor = admit_stage2_v3_successor(repo_root)
            from .admission import admit_stage0_graph
            successor["branch_gate"] = {"reliability_mask": list(admit_stage0_graph(repo_root)["reliability_mask"])}
            return _execute_stage3_estimator(
                repo_root, seed=seed, view=view, result_root=result_root,
                stage2_admission=successor, stage_label="stage2_v3",
            )
        if result_root == v2_estimator_root:
            require(seed == 44, "Stage2-v2 estimator reruns only interrupted seed44")
            successor = admit_stage2_v2_successor(repo_root, "estimator")
            from .admission import admit_stage0_graph
            successor["branch_gate"] = {"reliability_mask": list(admit_stage0_graph(repo_root)["reliability_mask"])}
            return _execute_stage3_estimator(repo_root,seed=seed,view=view,result_root=result_root,stage2_admission=successor,stage_label="stage2_v2")
        if result_root == v2_film_root:
            successor = admit_stage2_v2_successor(repo_root, "film")
            from .admission import admit_stage0_graph
            successor["branch_gate"] = {"reliability_mask": list(admit_stage0_graph(repo_root)["reliability_mask"])}
            return _execute_stage2_film(repo_root,seed=seed,view=view,result_root=result_root,stage_label="stage2_v2",branch_admission=successor)
        estimator_root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage2" / "estimator" / view / f"seed{seed}"
        film_root = repo_root / plan.RESULT_PARENT_RELATIVE / "stage2" / "film" / view / f"seed{seed}"
        if result_root == film_root:
            return _execute_stage2_film(repo_root,seed=seed,view=view,result_root=result_root)
        require(result_root == estimator_root, "Stage2 result root is neither canonical estimator nor canonical FiLM root")
        from .admission import admit_stage0_graph
        from .lifecycle import begin_attempt, close_failure, close_terminal, publish_immutable_json
        admission=dict(admit_stage0_graph(repo_root)); mask=tuple(bool(x) for x in admission["reliability_mask"])
        require(admission["estimator_route"] == "OPEN", "Stage2 estimator admission closed")
        attempt=begin_attempt(result_root,{"route":plan.ROUTE_NAME,"stage":"stage2_estimator","seed":seed,"view":view,"stage0_admission":admission,"gpu_initialized":False})
        try:
            launch_receipt=_attest_loads_before_cuda(repo_root,seed)
            import torch, io
            from mc_maze.dandi688_cp_film_v1 import runner as base
            surface=prepare_source_surface(repo_root,signal_view=view,reliability_mask=mask)
            require(torch.cuda.is_available() and torch.cuda.device_count()==1,"requires exactly one isolated CUDA device")
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
            device=torch.device("cuda:0"); parent=base._prepare_student(repo_root,seed,device)
            for p in parent.parameters(): p.requires_grad_(True)
            train=sorted(x for x,r in surface.sessions.items() if r.split=="train")
            def batches(epoch):
                yield from _all_q50_training_batches(surface,seed,epoch)
            def carrier(arm,b):
                return torch.from_numpy(surface.sessions[b[0]].whole_m10_t4 if arm=="WHOLE-T4" else surface.sessions[b[0]].post700_m10_t4).unsqueeze(0).to(device)
            def loss(model,b,c):
                name,r,starts=b; neural=torch.from_numpy(np.stack([r.record.neural[int(s):int(s)+50] for s in starts]).astype(np.float32)).to(device); target=torch.from_numpy(np.stack([r.record.behavior[int(s)+49] for s in starts]).astype(np.float32)).to(device); cal=torch.from_numpy(r.record.calib_trials.astype(np.float32)).unsqueeze(0).to(device); mean=model.id_encoder.pre_pool(cal.permute(0,1,3,2)).mean(1); identity=model.id_encoder.post_pool(torch.cat((mean,c),-1)); return torch.mean((model.decode_with_identity(neural,identity.expand(neural.shape[0],-1,-1))[:,-1]/5-target)**2)
            result=run_stage2_estimator_coordinated(parent_student=parent,batches=batches,carrier_for_arm=carrier,predict_loss=loss,score=lambda m:None,seed=seed)
            q50_score=_score_models_q50(surface,result["averaged"],{"WHOLE-T4":lambda row:row.whole_m10_t4,"POST700-T4":lambda row:row.post700_m10_t4},device)
            artifacts=_publish_stage2_full_state_artifacts(result_root,result,seed)
            body=publish_immutable_json(result_root,"training.json",{"attempt_sha256":attempt,"launch_receipt":launch_receipt,"epoch_receipts":result["epoch_receipts"],"student_trainability":result["student_trainability"],"full_state_artifacts":artifacts,"average_epochs_zero_based":[8,9,10,11]})
            score_sha=publish_immutable_json(result_root,"score.json",{"attempt_sha256":attempt,"launch_receipt":launch_receipt,"per_session":q50_score})
            close_terminal(result_root,attempt_sha256=attempt,payload={"training_sha256":body,"score_sha256":score_sha,"full_state_artifacts":artifacts,"launch_receipt":launch_receipt,"status_detail":"STAGE2_ESTIMATOR_SEED_COMPLETE"}); return {"attempt_sha256":attempt,"training_sha256":body,"score_sha256":score_sha}
        except BaseException as error:
            close_failure(result_root,attempt_sha256=attempt,error=error); raise
    require(stage == 1, "unsupported stage")
    require(view == "sua" and seed in plan.SEEDS, "Stage-1 currently supports SUA paired seeds only")
    from .admission import admit_stage1_successor
    from .lifecycle import begin_attempt, close_failure, close_terminal, publish_immutable_json
    repo_root = Path(repo_root).resolve()
    result_root = Path(result_root).resolve()
    expected_root = repo_root / plan.STAGE1_SUCCESSOR_RELATIVE / view / f"seed{seed}"
    require(result_root == expected_root, "Stage1 result root is not canonical")
    successor_admission = dict(admit_stage1_successor(repo_root))
    admission = dict(successor_admission["stage0_admission"])
    mask = tuple(bool(value) for value in admission["reliability_mask"])
    require(admission["film_route"] == "OPEN", "Stage1 FiLM admission closed")
    attempt_sha = begin_attempt(result_root, {"route": plan.ROUTE_NAME, "stage": "stage1_v3", "seed": seed, "view": view, "stage0_admission": admission, "successor_admission": successor_admission, "gpu_initialized": False, "decoder_opened": False, "formal_test_opened": False})
    try:
        # All expensive imports/data/CUDA work occurs strictly after attempt.
        launch_receipt = _attest_loads_before_cuda(repo_root, seed)
        import copy, io, torch
        from mc_maze.dandi688_cp_film_v1 import runner as base_runner
        surface = prepare_source_surface(repo_root, signal_view=view, reliability_mask=mask)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "requires exactly one isolated CUDA device")
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
        device = torch.device("cuda:0")
        student = base_runner._prepare_student(repo_root, seed, device)
        parent_state_before = base_runner._state_sha(student)
        build_film = __import__(
            "mc_maze.dandi688_sparse_event_t4_v1.core", fromlist=["build_film"]
        ).build_film
        film_template = build_film().to(device)
        films = {
            arm: copy.deepcopy(film_template).to(device)
            for arm in ("EMPTY", "PHASE-R", "SE-T4", "ROW-SHUFFLE")
        }
        initial_head_state_sha256 = {
            arm: base_runner._state_sha(film) for arm, film in films.items()
        }
        require(
            len(set(initial_head_state_sha256.values())) == 1,
            "paired FiLM initial state drift",
        )
        # Materialized cache uses route M10 carrier under parent M30 normalizer.
        cache = {}
        for name, row in surface.sessions.items():
            cal = torch.from_numpy(np.asarray(row.record.calib_trials, dtype=np.float32)).unsqueeze(0).to(device)
            carrier = torch.from_numpy(row.whole_m10_parentnorm).unsqueeze(0).to(device)
            with torch.no_grad():
                mean = student.id_encoder.pre_pool(cal.permute(0, 1, 3, 2)).mean(1)
                native = student.id_encoder.post_pool(torch.cat((mean, carrier), -1))
            full = torch.from_numpy(row.profile_m10).unsqueeze(0).to(device)
            phase = torch.from_numpy(np.column_stack((row.profile_m10[:, :3], np.zeros(row.profile_m10.shape[0], dtype=np.float32)))).unsqueeze(0).to(device)
            perm = __import__("mc_maze.dandi688_sparse_event_t4_v1.core", fromlist=["row_shuffle_profile"]).row_shuffle_profile(row.profile_m10, session_id=name, view=view, training_seed=seed)[0]
            cache[name] = (mean, carrier, native, {"EMPTY": torch.zeros_like(full), "PHASE-R": phase, "SE-T4": full, "ROW-SHUFFLE": torch.from_numpy(perm).unsqueeze(0).to(device)})
        optimizers = {name: torch.optim.Adam(film.parameters(), lr=plan.FILM_HEAD_LEARNING_RATE) for name, film in films.items()}
        epochs = {name: [] for name in films}
        epoch_receipts = []
        train_names = sorted(name for name, row in surface.sessions.items() if row.split == "train")
        sentinel = {"identity_bitwise": {}, "prediction_bitwise": {}, "passed": False}
        for epoch in range(plan.EPOCHS):
            losses = {name: [] for name in films}
            for session in train_names:
                row, (mean, carrier, _native, profiles) = surface.sessions[session], cache[session]
                sample_count = min(1024, row.q50_starts.size)
                rng = np.random.Generator(np.random.PCG64(int.from_bytes(__import__("hashlib").sha256(f"{plan.ROUTE_NAME}:stage1:{seed}:{epoch}:{session}".encode()).digest()[:8], "little")))
                starts = np.sort(rng.choice(row.q50_starts, size=sample_count, replace=False))
                for offset in range(0, starts.size, 32):
                    neural = torch.from_numpy(np.stack([row.record.neural[int(s):int(s)+50] for s in starts[offset:offset+32]]).astype(np.float32)).to(device)
                    target = torch.from_numpy(np.stack([row.record.behavior[int(s)+49] for s in starts[offset:offset+32]]).astype(np.float32)).to(device)
                    if not sentinel["passed"]:
                        with torch.no_grad():
                            native_prediction = student.decode_with_identity(neural, _native.expand(neural.shape[0],-1,-1))[:, -1] / 5.0
                            for arm, film in films.items():
                                identity, direct = __import__("mc_maze.dandi688_sparse_event_t4_v1.core", fromlist=["film_identity"]).film_identity(student.id_encoder, mean, carrier, profiles[arm], film)
                                prediction = student.decode_with_identity(neural, identity.expand(neural.shape[0],-1,-1))[:, -1] / 5.0
                                sentinel["identity_bitwise"][arm] = bool(direct["direct_native_branch"] and torch.equal(identity, _native))
                                sentinel["prediction_bitwise"][arm] = bool(torch.equal(prediction, native_prediction))
                        sentinel["passed"] = bool(all(sentinel["identity_bitwise"].values()) and all(sentinel["prediction_bitwise"].values()))
                        require(sentinel["passed"], "pre-first-step zero-anchor parity failure")
                    for arm, film in films.items():
                        optimizers[arm].zero_grad(set_to_none=True); identity, _ = __import__("mc_maze.dandi688_sparse_event_t4_v1.core", fromlist=["film_identity"]).film_identity(student.id_encoder, mean, carrier, profiles[arm], film, direct_zero=False); pred = student.decode_with_identity(neural, identity.expand(neural.shape[0], -1, -1))[:, -1] / 5.0; loss = torch.mean((pred-target)**2); require(bool(torch.isfinite(loss)), f"{arm}: loss nonfinite"); loss.backward(); optimizers[arm].step(); losses[arm].append(float(loss.detach().cpu()))
            for arm in films: epochs[arm].append({key: value.detach().cpu().clone() for key, value in films[arm].state_dict().items()})
            epoch_receipts.append({"epoch_zero_based": epoch, "loss_mean": {arm: float(np.mean(losses[arm])) for arm in films}, "optimizer_steps": {arm: len(losses[arm]) for arm in films}, "state_sha256": {arm: __import__("mc_maze.dandi688_cp_film_v1.runner", fromlist=["_state_sha"])._state_sha(films[arm]) for arm in films}})
        averaged = {}
        for arm, states in epochs.items():
            averaged[arm] = copy.deepcopy(films[arm])
            state = {}
            for key, value in states[0].items():
                state[key] = torch.stack([item[key].to(torch.float64) for item in states[8:12]]).mean(0).to(value.dtype) if value.is_floating_point() else value
            averaged[arm].load_state_dict(state)
        parent_state_after_training = base_runner._state_sha(student)
        require(parent_state_after_training == parent_state_before, "frozen Stage1 parent state changed during training")
        checkpoints = {}
        for arm, film in films.items():
            buf = io.BytesIO(); torch.save({"seed": seed, "arm": arm, "epoch_states": epochs[arm], "average_epochs_zero_based": [8,9,10,11], "average_state_dict": averaged[arm].state_dict()}, buf)
            from mc_maze.dandi688_cp_film_v1.runner import _publish_bytes
            checkpoints[arm] = _publish_bytes(result_root, f"stage1_{arm.lower()}.pt", buf.getvalue())
        train_sha = publish_immutable_json(result_root, "training.json", {"attempt_sha256": attempt_sha, "launch_receipt": launch_receipt, "epochs": 12, "arms": list(films), "epoch_state_count": {name: len(values) for name, values in epochs.items()}, "epoch_receipts": epoch_receipts, "average_epochs_zero_based": [8,9,10,11], "checkpoints": checkpoints, "gpu_uuid": launch_receipt["gpu_uuid"], "zero_anchor": sentinel,"initial_head_state_sha256":initial_head_state_sha256,"initial_head_states_identical":True,"head_template_factory_calls":1,"parent_state_before":parent_state_before,"parent_state_after_training":parent_state_after_training})
        scores = {name: {} for name in ("WHOLE-NATIVE", "WHOLE-PARENTNORM", "WHOLE-M10NORM", "POST700-M10NORM", *films)}
        score_evidence = {name: {} for name in scores}
        for session, row in surface.sessions.items():
            if row.split != "val": continue
            mean, carrier, native, profiles = cache[session]; target_parts=[]; pred={key: [] for key in scores}
            post = torch.from_numpy(row.post700_m10_t4).unsqueeze(0).to(device); whole_m10 = torch.from_numpy(row.whole_m10_t4).unsqueeze(0).to(device)
            with torch.no_grad():
                ids={"WHOLE-NATIVE": native, "WHOLE-PARENTNORM": native, "WHOLE-M10NORM": student.id_encoder.post_pool(torch.cat((mean, whole_m10),-1)), "POST700-M10NORM": student.id_encoder.post_pool(torch.cat((mean, post),-1))}
                ids.update({arm: __import__("mc_maze.dandi688_sparse_event_t4_v1.core", fromlist=["film_identity"]).film_identity(student.id_encoder, mean, carrier, profiles[arm], averaged[arm])[0] for arm in films})
                for offset in range(0,row.q50_starts.size,256):
                    starts=row.q50_starts[offset:offset+256]; neural=torch.from_numpy(np.stack([row.record.neural[int(s):int(s)+50] for s in starts]).astype(np.float32)).to(device); target_parts.append(np.stack([row.record.behavior[int(s)+49] for s in starts]).astype(np.float32))
                    for name, identity in ids.items(): pred[name].append((student.decode_with_identity(neural,identity.expand(neural.shape[0],-1,-1))[:,-1]/5.0).cpu().numpy())
            target=np.concatenate(target_parts); total=np.square(target-target.mean(0)).sum();
            for name, values in pred.items():
                prediction = np.concatenate(values)
                r2 = float(1-np.square(target-prediction).sum()/total)
                scores[name][session] = r2
                score_evidence[name][session] = {
                    "r2": r2,
                    "prediction_sha256": array_sha256(prediction),
                    "target_sha256": array_sha256(target),
                    "query_sha256": array_sha256(row.q50_starts),
                    "window_count": int(target.shape[0]),
                }
        parent_state_after_score = base_runner._state_sha(student)
        require(parent_state_after_score == parent_state_before, "frozen Stage1 parent state changed during scoring")
        sufficient = {name: {"equal_session_mean": float(np.mean(list(values.values()))), "session_count": len(values), "prediction_surface": "Q50"} for name, values in scores.items()}
        score_sha = publish_immutable_json(result_root, "score.json", {"attempt_sha256":attempt_sha,"per_session_r2":scores,"per_session_evidence":score_evidence,"sufficient_statistics": sufficient,"gpu_uuid":launch_receipt["gpu_uuid"],"parent_state_before":parent_state_before,"parent_state_after_training":parent_state_after_training,"parent_state_after_score":parent_state_after_score})
        close_terminal(result_root, attempt_sha256=attempt_sha, payload={"training_sha256": train_sha, "score_sha256": score_sha, "launch_receipt":launch_receipt,"parent_state_sha256":parent_state_before,"status_detail": "STAGE1_SEED_COMPLETE__THREE_SEED_OPENING_NOT_DECIDED"})
        return {"attempt_sha256": attempt_sha, "training_sha256": train_sha, "score_sha256": score_sha}
    except BaseException as error:
        close_failure(result_root, attempt_sha256=attempt_sha, error=error); raise
