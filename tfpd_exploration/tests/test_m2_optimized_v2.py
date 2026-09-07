import torch
import pytest

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_optimized_v2.decoder import M2ExactOptimizedDecoder
from tfpd_exploration.src.two_mainlines_long_v1.latency_opt_v2 import FrontendWindowCache
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import load_s1_ema
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import load_pick


def test_m2_front_cache_and_last_query_match_oracle_on_shift():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, _ = load_s1_ema(device="cpu")
    bank = data.load_session_bank("ext4", plan.EXT4_SESSIONS[0], device="cpu")
    x = torch.tensor(bank.X_store[:54], dtype=torch.float32)
    optimized = M2ExactOptimizedDecoder(model, [bank])
    with torch.inference_mode():
        got0 = optimized.rebuild(x[:50].unsqueeze(0))
        ref0 = model.forward_last(x[:50].unsqueeze(0), bank, bank.unit_mask)
        got1 = optimized.advance(x[50:51].unsqueeze(0))
        ref1 = model.forward_last(x[1:51].unsqueeze(0), bank, bank.unit_mask)
    assert torch.allclose(got0, ref0, atol=1e-5, rtol=1e-5)
    assert torch.allclose(got1, ref1, atol=1e-5, rtol=1e-5)


def test_m2_cache_auto_invalidates_on_inplace_bank_or_weight_change():
    model, _ = load_s1_ema(device="cpu")
    bank = data.load_session_bank("ext4", plan.EXT4_SESSIONS[0], device="cpu")
    x = torch.tensor(bank.X_store[:52], dtype=torch.float32)
    optimized = M2ExactOptimizedDecoder(model, [bank])
    with torch.inference_mode():
        optimized.rebuild(x[:50].unsqueeze(0))
        bank.T.add_(0.0)  # Version changes even for a numerically neutral mutation.
        with pytest.raises(RuntimeError, match="advance requires rebuild"):
            optimized.advance(x[50:51].unsqueeze(0))
        optimized.rebuild(x[1:51].unsqueeze(0))
        next(model.parameters()).add_(0.0)
        with pytest.raises(RuntimeError, match="advance requires rebuild"):
            optimized.advance(x[51:52].unsqueeze(0))


def test_generic_frontend_cache_reports_repaired_positions_and_rejects_kernel_one():
    frontend = lambda x: x.sum(dim=-1, keepdim=True)
    with pytest.raises(ValueError, match="kernel >= 2"):
        FrontendWindowCache(frontend, window=5, kernel=1)
    cache = FrontendWindowCache(frontend, window=5, kernel=5)
    raw = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
    cache.rebuild(raw)
    cache.advance(torch.ones(1, 1, 2))
    assert cache.changed_indices == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("kind",("small","large"))
def test_m2_b7_distinct_bank_startup_rollover_permutation_mask_and_numeric_mutation(kind):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, _ = load_pick(kind,device="cpu")
    sessions=plan.HELDIN_SESSIONS[:7]
    banks=[data.load_session_bank("source_minival", s, device="cpu") for s in sessions]
    traces=[torch.tensor(b.X_store[:64],dtype=torch.float32) for b in banks]
    optimized=M2ExactOptimizedDecoder(model,banks)
    raw=torch.zeros(7,50,96)
    with torch.inference_mode():
        # True decoder startup: first rebuild is all-zero, then every source
        # bin advances the cache; selected steps cover both boundary and rollover.
        optimized.rebuild(raw)
        for t in range(57):
            step=torch.stack([x[t] for x in traces]).unsqueeze(1)
            got=optimized.advance(step)
            raw=torch.cat((raw[:,1:],step),1)
            if t in {0,1,3,4,5,24,49,50}:
                ref=torch.cat([model.forward_last(raw[i:i+1],b,b.unit_mask) for i,b in enumerate(banks)])
                assert torch.allclose(got,ref,atol=1e-5,rtol=1e-5)
        for t in range(57,60):
            step=torch.stack([x[t] for x in traces]).unsqueeze(1)
            got=optimized.advance(step)
            raw=torch.cat((raw[:,1:],step),1)
            ref=torch.cat([model.forward_last(raw[i:i+1],b,b.unit_mask) for i,b in enumerate(banks)])
            assert torch.allclose(got,ref,atol=1e-5,rtol=1e-5)
        # A semantic, not merely version-neutral, bank mutation forces rebuild.
        banks[3].T[0,0].add_(0.125)
        with pytest.raises(RuntimeError,match="advance requires rebuild"):
            optimized.advance(raw[:,-1:])
        got=optimized.rebuild(raw); ref=torch.cat([model.forward_last(raw[i:i+1],b,b.unit_mask) for i,b in enumerate(banks)])
        assert torch.allclose(got,ref,atol=1e-5,rtol=1e-5)
        # Mask semantics are part of cache identity; preserve at least one unit.
        banks[2].unit_mask[1]=False
        with pytest.raises(RuntimeError,match="advance requires rebuild"):
            optimized.advance(raw[:,-1:])
        got=optimized.rebuild(raw); ref=torch.cat([model.forward_last(raw[i:i+1],b,b.unit_mask) for i,b in enumerate(banks)])
        assert torch.allclose(got,ref,atol=1e-5,rtol=1e-5)
        # Independent session row permutation must travel with raw row and bank.
        rows=torch.tensor([6,4,2,0,5,1,3])
        row_banks=[banks[int(i)] for i in rows]
        row_raw=raw[rows]
        row_opt=M2ExactOptimizedDecoder(model,row_banks)
        got_rows=row_opt.rebuild(row_raw)
        ref_rows=torch.cat([model.forward_last(row_raw[i:i+1],b,b.unit_mask) for i,b in enumerate(row_banks)])
        assert torch.allclose(got_rows,ref_rows,atol=1e-5,rtol=1e-5)
        # Unit order is bank/mask/raw synchronized; independently reorder each
        # session and retain its own output row.
        from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.container_decoder import SessionBank
        perms=[torch.roll(torch.arange(96),i) for i in range(7)]
        with torch.inference_mode(False):
            permuted=[SessionBank(b.E0[q],b.T[q],b.unit_mask[q]) for b,q in zip(banks,perms)]
            raw_p=torch.stack([raw[i,:,q] for i,q in enumerate(perms)])
            reordered=M2ExactOptimizedDecoder(model,permuted)
            got_p=reordered.rebuild(raw_p)
            ref_p=torch.cat([model.forward_last(raw_p[i:i+1],b,b.unit_mask) for i,b in enumerate(permuted)])
        assert torch.allclose(got_p,ref_p,atol=1e-5,rtol=1e-5)
        # Same numerical assertion for a changed model weight.
        next(model.parameters()).view(-1)[0].add_(1e-4)
        with pytest.raises(RuntimeError,match="advance requires rebuild"):
            optimized.advance(raw[:,-1:])
        got=optimized.rebuild(raw); ref=torch.cat([model.forward_last(raw[i:i+1],b,b.unit_mask) for i,b in enumerate(banks)])
        assert torch.allclose(got,ref,atol=1e-5,rtol=1e-5)


def test_m2_dtype_migration_requires_rebuild_and_inference_bank_is_rejected():
    model,_=load_s1_ema(device="cpu")
    bank=data.load_session_bank("source_minival",plan.HELDIN_SESSIONS[0],device="cpu")
    raw=torch.zeros(1,50,96); opt=M2ExactOptimizedDecoder(model,[bank]); opt.rebuild(raw)
    model.double()
    with pytest.raises(RuntimeError,match="advance requires rebuild"):
        opt.advance(raw[:,-1:])
    # Rebuild after dtype migration uses the migrated model/bank compilation.
    got=opt.rebuild(raw.double())
    with torch.inference_mode():
        ref=model.forward_last(raw.double(),bank,bank.unit_mask)
    assert torch.allclose(got,ref,atol=1e-5,rtol=1e-5)
    from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.container_decoder import SessionBank
    with torch.inference_mode():
        bad=SessionBank(bank.E0.clone(),bank.T.clone(),bank.unit_mask.clone())
    with pytest.raises(RuntimeError,match="outside torch.inference_mode"):
        M2ExactOptimizedDecoder(model,[bad])
