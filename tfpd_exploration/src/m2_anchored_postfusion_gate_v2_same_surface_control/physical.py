"""Post-attempt CPU-only APFG V2 coordinator; no module-level Torch import.

The route intentionally owns only orchestration.  Native POOLED decoding is
delegated to ``g_replay.rollout_g00m`` and APFG decoding to the frozen V1
``rollout_apfg_raw_pool`` primitive.  Thus V2 changes the numerical surface
(same-process CPU native control), not the activity law, decoder, metric, or
causal commit law.
"""
from __future__ import annotations
import copy
import hashlib
from typing import Any, Mapping, Sequence
from . import plan
class PhysicalError(RuntimeError): pass
def _need(ok,msg):
    if not ok: raise PhysicalError(msg)
def make_cpu_views(*, strict_loaded_module: Any, refit_alpha: float, torch: Any):
    """Create native, +0 adapter, and sealed-alpha adapter from one state."""
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.adapter import freeze_and_install, exact_positive_zero
    native=copy.deepcopy(strict_loaded_module).to(torch.device('cpu')).eval()
    zero=copy.deepcopy(strict_loaded_module).to(torch.device('cpu')).eval(); za=freeze_and_install(zero.student); za.set_alpha_training(False)
    learned=copy.deepcopy(strict_loaded_module).to(torch.device('cpu')).eval(); la=freeze_and_install(learned.student)
    with torch.no_grad(): la.alpha.fill_(float(refit_alpha))
    la.set_alpha_training(False)
    for module in (native,zero,learned):
        for parameter in module.parameters(): parameter.requires_grad_(False); parameter.grad=None
        module.eval(); _need(not module.training and all(not p.requires_grad for p in module.parameters()),'V2 target model freeze drift')
    _need(exact_positive_zero(za.alpha),'V2 zero alpha lost IEEE +0.0')
    _need(float(la.alpha.detach().cpu())==plan.REFIT_ALPHA,'V2 sealed refit alpha drift')
    return {"NATIVE-POOLED":native,"APFG-ZERO":zero,"APFG-LEARNED":learned}
def assert_cpu_only(torch: Any):
    _need(not torch.cuda.is_initialized(),'V2 CPU scorer must not initialize CUDA')


def prepare_selected_cpu_once(*, repo_root: Any) -> dict[str, Any]:
    """Prepare one PIT CPU stack and strict-load the exact selected-T4 state.

    This is callable only after V2's immutable attempt.  It never creates a
    second DataModule and never initializes CUDA.  The selected checkpoint is
    read through the reviewed selected-T4 strict loader rather than via a new
    path in this successor.
    """
    import os
    _need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'V2 CPU scorer requires empty CVD')
    import torch
    assert_cpu_only(torch)
    from tfpd_exploration.src.pit_m2_v1 import trainer
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as scorer_physical
    runner=trainer.PitM2ArmedRunner(repo_root, 't0m', device='cpu')
    runner.prepare(attach_operator=False)
    base=runner._litmodule
    _need(base is not None and runner._datamodule is not None, 'V2 PIT CPU stack absent')
    base.eval()
    for p in base.parameters(): p.requires_grad_(False); p.grad=None
    selected,evidence=scorer_physical._strict_sealed_pooled_clone(repo_root=repo_root, base_module=base)
    _need(evidence.get('selected_t4_checkpoint_sha256') == plan.SELECTED_CHECKPOINT_SHA256,
          'V2 selected checkpoint literal drift')
    _need(evidence.get('student_state_after_load_sha256') == plan.SELECTED_STUDENT_STATE_SHA256,
          'V2 selected student state literal drift')
    views=make_cpu_views(strict_loaded_module=selected, refit_alpha=plan.REFIT_ALPHA, torch=torch)
    _need(not torch.cuda.is_initialized(), 'V2 PIT/strict load initialized CUDA')
    return {'datamodule':runner._datamodule, 'base_module':base, 'selected_module':selected,
            'views':views, 'selected_evidence':evidence, 'pit_prepare_calls':1,
            'cuda_initialized':False}


def materialize_13_same_process_inputs(*, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize each input once and generate its CPU native control once."""
    import torch
    assert_cpu_only(torch)
    from tfpd_exploration.src.cdm_p1_m2_local_v1 import replay as g_replay
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as scorer_physical
    native=prepared.get('views',{}).get('NATIVE-POOLED')
    datamodule=prepared.get('datamodule')
    _need(native is not None and datamodule is not None, 'V2 native/datamodule missing')
    _need(scorer_physical._student_state_sha(native) == plan.SELECTED_STUDENT_STATE_SHA256,
          'V2 native selected state drift before target materialization')
    for name in ('APFG-ZERO','APFG-LEARNED'):
        _need(_selected_native_substate_sha(model=prepared['views'][name]) == plan.SELECTED_STUDENT_STATE_SHA256,
              f'V2 {name} inherited native substate drift before target materialization')
    comparators: dict[str, dict[str, Any]]={}
    state=scorer_physical._student_state_sha(native)
    def native_comparator(record: Mapping[str, Any]) -> Mapping[str, Any]:
        runtime=record['_runtime']; dataset=runtime['dataset']
        before=scorer_physical._student_state_sha(native)
        result=g_replay.rollout_g00m(torch=torch, model=native, ds=dataset, session=str(record['session']),
                                     views=runtime['views'], support=runtime['support'], query_rows=runtime['query_rows'],
                                     side_mean=__import__('numpy').asarray(dataset.side_feature_mean, dtype='float32'),
                                     side_std=__import__('numpy').asarray(dataset.side_feature_std, dtype='float32'),
                                     device=torch.device('cpu'), batch_size=plan.CPU_DECODE_BATCH_SIZE)
        after=scorer_physical._student_state_sha(native)
        _need(before == after == state, f"{record['key']}: V2 native scorer mutated selected state")
        out={'policy':'NATIVE-POOLED-SAME-PROCESS-CPU','r2':float(result['r2']),
             'window_count':int(result['window_count']),'query_starts_sha256':str(result['query_starts_sha256']),
             'target_sha256':str(result['target_sha256']),'prediction_sha256':str(result['prediction_sha256']),
             'model_state_before_sha256':before,'model_state_after_sha256':after,
             'parameter_updates':0,'target_updates':0}
        comparators[str(record['key'])]=out
        return out
    materialized=scorer_physical.materialize_13_inputs_and_pooled_comparators(
        prepared={'datamodule':datamodule, 'base_module':prepared['base_module']}, pooled_comparator=native_comparator)
    _need(len(comparators)==13 and len(materialized.get('records',{}))==13, 'V2 exact same-process materialization drift')
    materialized['native_comparators']=comparators
    materialized['native_same_process_state_sha256']=state
    return materialized


def _native_row(*, record: Mapping[str, Any], comparator: Mapping[str, Any]) -> dict[str, Any]:
    return {'surface':str(record['surface']), 'session':str(record['session']), 'system':'NATIVE-POOLED',
            'memory_law':'FIXED30', 'input_authority_key':str(record['key']), 'r2':float(comparator['r2']),
            'prediction_sha256':str(comparator['prediction_sha256']), 'target_sha256':str(comparator['target_sha256']),
            'query_starts_sha256':str(comparator['query_starts_sha256']), 'window_count':int(comparator['window_count']),
            'model_state_before_sha256':str(comparator['model_state_before_sha256']),
            'model_state_after_sha256':str(comparator['model_state_after_sha256']),
            'native_substate_sha256':str(comparator['model_state_before_sha256']),
            'alpha_exact_positive_zero':None,
            'parameter_updates':0, 'target_updates':0, 'activity_authority':'pooled_g00m_linear'}


def _selected_native_substate_sha(*, model: Any) -> str:
    """Digest a wrapped APFG student in the native selected-T4 key space.

    Installing the scalar wrapper changes ``state_dict`` key names and adds
    alpha, so whole-model hashes are correctly different.  The native
    substate remapping proves instead that every inherited selected-T4 tensor
    stayed byte-identical to the strict-loaded checkpoint.
    """
    digest=hashlib.sha256()
    for name,value in sorted(model.student.state_dict().items()):
        if name=='id_encoder.alpha':
            continue
        mapped=('id_encoder.'+name[len('id_encoder.native.'):]
                if name.startswith('id_encoder.native.') else name)
        array=value.detach().cpu().contiguous().numpy()
        digest.update(plan.canonical_json({'name':mapped,'shape':list(array.shape),'dtype':str(array.dtype)}))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _score_one_record(*, torch: Any, views: Mapping[str, Any], record: Mapping[str, Any],
                      comparator: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use the reviewed V1 per-record rollout twice per law; no copied decoder.

    The local row assembly is only the V2 five-system receipt projection.  All
    activity pool construction, decode-before-commit, neural windowing, T4,
    batch chunking, and R2 calculation remain inside reviewed primitives.
    """
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import scoring as v1_scoring
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.source_replay import material_from_g00m_runtime_record, rollout_apfg_raw_pool
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as scorer_physical
    material=material_from_g00m_runtime_record(record); dataset=record['_runtime']['dataset']
    zero_rows: dict[str, dict[str, Any]]={}; learned_rows: dict[str, dict[str, Any]]={}
    for law in plan.LAWS:
        zero=views['APFG-ZERO']; learned=views['APFG-LEARNED']
        zstate=scorer_physical._student_state_sha(zero)
        z=rollout_apfg_raw_pool(torch=torch, model=zero, material=material, dataset=dataset, law=law,
                                device=torch.device('cpu'), batch_size=plan.CPU_DECODE_BATCH_SIZE)
        _need(zstate==scorer_physical._student_state_sha(zero), f"{record['key']}: zero scoring mutated model")
        zrow=v1_scoring._row(system='APFG-ZERO', law=law, record=record, result=z, state=zstate)
        zrow['native_substate_sha256']=_selected_native_substate_sha(model=zero)
        zrow['alpha_exact_positive_zero']=True
        if law=='FIXED30': v1_scoring._strict_zero_match(zrow, comparator)
        lstate=scorer_physical._student_state_sha(learned)
        lv=rollout_apfg_raw_pool(torch=torch, model=learned, material=material, dataset=dataset, law=law,
                                 device=torch.device('cpu'), batch_size=plan.CPU_DECODE_BATCH_SIZE)
        _need(lstate==scorer_physical._student_state_sha(learned), f"{record['key']}: learned scoring mutated model")
        lrow=v1_scoring._row(system='APFG-LEARNED', law=law, record=record, result=lv, state=lstate)
        lrow['native_substate_sha256']=_selected_native_substate_sha(model=learned)
        lrow['alpha_exact_positive_zero']=False
        # The target-only successor must never silently score a different
        # refit scalar than the one sealed by the held V1 alpha-selection
        # receipt.  Persist it per learned row as well as in final receipts.
        lrow['learned_refit_alpha']=float(plan.REFIT_ALPHA)
        zero_rows[law]=zrow; learned_rows[law]=lrow
    # V2's receipt order is system-major.  Do not use the natural V1
    # per-law interleaving here: the latter is correct for V1's 52-row codec
    # but would fail V2's 65-row canonical sequence after all target work.
    return [_native_row(record=record, comparator=comparator), zero_rows['FIXED30'], zero_rows['UNCAPPED'],
            learned_rows['FIXED30'], learned_rows['UNCAPPED']]


def score_65_rows_from_materialized(*, prepared: Mapping[str, Any], materialized: Mapping[str, Any],
                                    record_keys: Sequence[str] | None=None) -> list[dict[str, Any]]:
    """Produce canonical V2 rows, optionally for the one-session smoke only."""
    import torch
    assert_cpu_only(torch)
    records=materialized.get('records'); comparators=materialized.get('native_comparators'); views=prepared.get('views')
    _need(isinstance(records, Mapping) and isinstance(comparators, Mapping) and isinstance(views, Mapping), 'V2 material authority missing')
    chosen=None if record_keys is None else {str(key) for key in record_keys}
    if chosen is not None: _need(chosen and chosen <= set(records), 'V2 selected record key drift')
    rows=[]
    for surface in plan.SURFACES:
        group=sorted((record for record in records.values() if record.get('surface')==surface), key=lambda r:str(r['session']))
        _need(len(group)==plan.ROSTER_SIZES[surface], 'V2 surface roster drift')
        for record in group:
            if chosen is not None and str(record['key']) not in chosen: continue
            rows.extend(_score_one_record(torch=torch, views=views, record=record, comparator=comparators[str(record['key'])]))
    if chosen is None:
        from . import laws
        laws.validate_rows(rows)
    else:
        _need(len(rows)==5*len(chosen), 'V2 smoke row cardinality drift')
    return rows
