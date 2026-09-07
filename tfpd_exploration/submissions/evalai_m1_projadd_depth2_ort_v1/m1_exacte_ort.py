"""ONNX Runtime CPU backend for the fast exact-E M1 proj_add runtime.

Candidate B of P1-B (workorder section 3): the per-bin advance path
(set-attention over the 5 changed positions -> z window assembly ->
temporal blocks 1-3 -> last-query last block -> readout) and the per-wave
rebuild path (full frontend + same temporal tail) are exported to ONNX and
run with onnxruntime CPUExecutionProvider. Same weights, same operator
semantics (SDPA causal/last-query, windowed sinusoidal PE, LayerNorm, k=5
boundary recompute); no cross-window temporal KV reuse.

Static per-wave bank tensors (P(E0), carrier T4) are graph INPUTS so one
graph serves any bank composition; model weights, slots and PE are baked as
constants. Graphs are exported OFFLINE (export_ort_graphs) and shipped as
files; runtime session creation is the only cold cost (disclosed).

Mask policy: exported unmasked; requires all-true bank unit masks (verified
for the 7 public tags); otherwise the engine refuses and the caller falls
back to the eager fast path.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_exacte_fast as fast  # noqa: E402


class _OrtAdvanceCore(nn.Module):
    """advance-path graph: local_sel + z_prev -> prediction."""

    def __init__(self, model) -> None:
        super().__init__()
        fe = model.frontend
        self.token_norm = fe.token_norm
        self.token_mlp = fe.token_mlp
        self.mha = fe.mha
        self.slot_ffn_norm = fe.slot_ffn_norm
        self.slot_ffn = fe.slot_ffn
        self.slot_proj = fe.slot_proj
        self.slot_norm = fe.slot_norm
        self.slots = fe.slots
        self.blocks = nn.ModuleList(list(model.temporal.blocks))
        self.final_norm = model.final_norm
        self.readout = model.readout
        self.register_buffer("pe", model.temporal.pe[:100].clone().unsqueeze(0), persistent=False)

    def forward(self, local_sel: torch.Tensor, z_prev: torch.Tensor, proj_static: torch.Tensor, t4_static: torch.Tensor) -> torch.Tensor:
        cfg_slots = self.slots.shape[0]
        set_dim = self.slots.shape[1]
        batch, P, n_units, conv_ch = local_sel.shape
        proj_out = conv_ch * proj_static.shape[-2]
        fused = (local_sel.unsqueeze(-2) + proj_static).reshape(batch, P, n_units, proj_out)
        tokens = self.token_norm(self.token_mlp(torch.cat([fused, t4_static.expand(batch, P, n_units, t4_static.shape[-1])], dim=-1)))
        slots = self.slot_norm(self.slots).view(1, 1, cfg_slots, set_dim).expand(batch, P, cfg_slots, set_dim)
        q = slots.reshape(batch * P, cfg_slots, set_dim)
        k = tokens.reshape(batch * P, n_units, set_dim)
        attn_out, _ = self.mha(q, k, k, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        z5 = self.slot_proj(slots_out.reshape(batch, P, cfg_slots * set_dim))
        z = torch.cat((z5[:, :4], z_prev[:, 5:], z5[:, 4:5]), dim=1)
        hidden = z + self.pe
        for block in self.blocks[:-1]:
            hidden = block(hidden)
        block = self.blocks[-1]
        normed = block.norm1(hidden)
        attn = block.attn
        b_, w_, d_ = normed.shape
        qkv = attn.qkv(normed).view(b_, w_, 3, attn.n_heads, attn.head_dim)
        qq, kk, vv = qkv.unbind(dim=2)
        qq = qq[:, -1:].transpose(1, 2)
        kk = kk.transpose(1, 2)
        vv = vv.transpose(1, 2)
        ao = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=0.0, is_causal=False)
        ao = attn.proj(ao.transpose(1, 2).contiguous().view(b_, 1, d_))
        last = hidden[:, -1:] + ao
        last = last + block.ffn(block.norm2(last))
        return self.readout(self.final_norm(last))[:, 0, :], z


class _OrtRebuildCore(nn.Module):
    """rebuild-path graph: full window -> prediction (once per wave)."""

    def __init__(self, model) -> None:
        super().__init__()
        fe = model.frontend
        self.local_conv = fe.local_conv
        self.token_norm = fe.token_norm
        self.token_mlp = fe.token_mlp
        self.mha = fe.mha
        self.slot_ffn_norm = fe.slot_ffn_norm
        self.slot_ffn = fe.slot_ffn
        self.slot_proj = fe.slot_proj
        self.slot_norm = fe.slot_norm
        self.slots = fe.slots
        self.blocks = nn.ModuleList(list(model.temporal.blocks))
        self.final_norm = model.final_norm
        self.readout = model.readout
        self.register_buffer("pe", model.temporal.pe[:100].clone().unsqueeze(0), persistent=False)

    def forward(self, window: torch.Tensor, proj_static: torch.Tensor, t4_static: torch.Tensor) -> torch.Tensor:
        cfg_slots = self.slots.shape[0]
        set_dim = self.slots.shape[1]
        batch, width, n_units = window.shape
        local = self.local_conv(window)
        conv_ch = local.shape[-1]
        proj_out = conv_ch * proj_static.shape[-2]
        fused = (local.unsqueeze(-2) + proj_static).reshape(batch, width, n_units, proj_out)
        tokens = self.token_norm(self.token_mlp(torch.cat([fused, t4_static.expand(batch, width, n_units, t4_static.shape[-1])], dim=-1)))
        slots = self.slot_norm(self.slots).view(1, 1, cfg_slots, set_dim).expand(batch, width, cfg_slots, set_dim)
        q = slots.reshape(batch * width, cfg_slots, set_dim)
        k = tokens.reshape(batch * width, n_units, set_dim)
        attn_out, _ = self.mha(q, k, k, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        z = self.slot_proj(slots_out.reshape(batch, width, cfg_slots * set_dim))
        hidden = z + self.pe
        for block in self.blocks[:-1]:
            hidden = block(hidden)
        block = self.blocks[-1]
        normed = block.norm1(hidden)
        attn = block.attn
        b_, w_, d_ = normed.shape
        qkv = attn.qkv(normed).view(b_, w_, 3, attn.n_heads, attn.head_dim)
        qq, kk, vv = qkv.unbind(dim=2)
        qq = qq[:, -1:].transpose(1, 2)
        kk = kk.transpose(1, 2)
        vv = vv.transpose(1, 2)
        ao = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=0.0, is_causal=False)
        ao = attn.proj(ao.transpose(1, 2).contiguous().view(b_, 1, d_))
        last = hidden[:, -1:] + ao
        last = last + block.ffn(block.norm2(last))
        return self.readout(self.final_norm(last))[:, 0, :], z


def export_ort_graphs(model, out_dir: Path, batches=(1, 2, 3, 4), n_units: int = 64) -> dict:
    """Export per-batch ONNX graphs with the frozen EMA weights baked in."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for B in batches:
        adv = _OrtAdvanceCore(model).eval()
        reb = _OrtRebuildCore(model).eval()
        local_sel = torch.randn(B, 5, n_units, 16)
        z_prev = torch.randn(B, 100, 256)
        proj_static = torch.randn(B, 1, n_units, 1, 16)
        t4_static = torch.randn(B, 1, n_units, 4)
        window = torch.randn(B, 100, n_units)
        p_adv = out_dir / f"ort_adv_b{B}.onnx"
        p_reb = out_dir / f"ort_rebuild_b{B}.onnx"
        torch.onnx.export(
            adv, (local_sel, z_prev, proj_static, t4_static), str(p_adv),
            opset_version=17, input_names=["local_sel", "z_prev", "proj_static", "t4_static"],
            output_names=["out", "z_out"], dynamo=False,
        )
        torch.onnx.export(
            reb, (window, proj_static, t4_static), str(p_reb),
            opset_version=17, input_names=["window", "proj_static", "t4_static"],
            output_names=["out", "z_out"], dynamo=False,
        )
        rows[f"B{B}"] = {"advance": str(p_adv), "rebuild": str(p_reb)}
    return rows


class OrtExactEEngine(fast.FastExactEEngine):
    """Fast engine with the advance/rebuild cores on ONNX Runtime."""

    def __init__(self, model, banks, graph_dir: Path, intra_op: int = 2, inter_op: int = 1) -> None:
        super().__init__(model, banks)
        import onnxruntime as ort

        if not self.keep_all_true:
            raise ValueError("ORT path requires all-true bank unit masks (public M1 banks satisfy this)")
        self._ort = ort
        self._graph_dir = Path(graph_dir)
        self._so = ort.SessionOptions()
        self._so.intra_op_num_threads = intra_op
        self._so.inter_op_num_threads = inter_op
        self.sess_adv = {}
        self.sess_reb = {}
        self.session_init_s = 0.0

    def _session(self, kind: str, B: int):
        cache = self.sess_adv if kind == "adv" else self.sess_reb
        if B not in cache:
            path = self._graph_dir / f"ort_{kind}_b{B}.onnx"
            if not path.is_file():
                return None
            import time as _t

            t0 = _t.perf_counter()
            cache[B] = self._ort.InferenceSession(str(path), self._so, providers=["CPUExecutionProvider"])
            self.session_init_s += _t.perf_counter() - t0
        return cache[B]

    def rebuild(self, window: torch.Tensor) -> torch.Tensor:
        B = window.shape[0]
        self.raw = window.detach().clone()
        sess = self._session("rebuild", B)
        if sess is not None:
            out, z = sess.run(None, {
                "window": window.numpy(),
                "proj_static": self.proj_static.numpy(),
                "t4_static": self.t4_static.numpy(),
            })
            self.z = torch.from_numpy(z)
            return torch.from_numpy(out)
        with torch.inference_mode():
            self.z = self._frontend_full(self.raw).detach().clone()
            return self._temporal_last(self.z)

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        raw = self.raw
        z_prev = self.z
        conv = self.model.frontend.local_conv
        k = self.kernel
        with torch.inference_mode():
            raw.copy_(torch.roll(raw, shifts=-1, dims=1))
            raw[:, -1] = next_bin[:, 0]
            left = conv(raw[:, : k - 1])
            right = conv(raw[:, -k:])[:, -1:]
            local_sel = torch.cat((left, right), dim=1)
        B = raw.shape[0]
        sess = self._session("adv", B)
        if sess is not None:
            out, z_new = sess.run(None, {
                "local_sel": local_sel.numpy(),
                "z_prev": z_prev.numpy(),
                "proj_static": self.proj_static.numpy(),
                "t4_static": self.t4_static.numpy(),
            })
            self.z = torch.from_numpy(z_new)
            return torch.from_numpy(out)
        with torch.inference_mode():
            z5 = self._set_attention(local_sel)
            z = torch.empty_like(z_prev)
            z[:, :4] = z5[:, :4]
            z[:, 4:-1] = z_prev[:, 5:]
            z[:, -1] = z5[:, 4]
            self.z = z
            return self._temporal_last(z)


class OrtTrfFalconDecoder(fast.FastTrfFalconDecoder):
    """Protocol-identical decoder with the ORT-backed engine."""

    def __init__(self, *args, graph_dir=None, intra_op: int = 2, inter_op: int = 1, **kw):
        self._graph_dir = Path(graph_dir)
        self._intra_op = intra_op
        self._inter_op = inter_op
        super().__init__(*args, **kw)

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        if not self.local_banks:
            raise RuntimeError("reset(dataset_tags) must be called before predict")
        self.observe(neural_observations)
        n_active = len(self.local_banks)
        decoder_input = torch.as_tensor(
            np.ascontiguousarray(self.observation_buffer[:, :n_active, :].transpose(1, 0, 2)),
            dtype=torch.float32,
            device=self.device,
        )
        if self._engine is None:
            self._engine = OrtExactEEngine(
                self.decoder, self.local_banks, self._graph_dir,
                intra_op=self._intra_op, inter_op=self._inter_op,
            )
            with torch.inference_mode():
                prediction = self._engine.rebuild(decoder_input)
        else:
            with torch.inference_mode():
                prediction = self._engine.advance(decoder_input[:, -1:, :])
        native = prediction.detach().cpu().numpy() / self.behavior_scaling_factor
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite Transformer prediction")
        if n_active < self.batch_size:
            pad = np.zeros((self.batch_size - n_active, native.shape[1]), dtype=np.float32)
            native = np.concatenate([native, pad], axis=0)
        self._n_predicts += 1
        if self._n_predicts % 2000 == 0:
            print(f"ort-e predict_steps={self._n_predicts} n_active={n_active}", flush=True)
        return native.astype(np.float32, copy=False)


def smoke(payload_path: str, window_path: str, graph_dir: str) -> None:
    import json

    from falcon_challenge.config import FalconConfig, FalconTask

    bundle = np.load(window_path)
    config = FalconConfig(task=FalconTask.m1)
    decoder = OrtTrfFalconDecoder(
        task_config=config,
        model_path=payload_path,
        batch_size=1,
        graph_dir=graph_dir,
        intra_op=2,
        inter_op=1,
    )
    decoder.reset(dataset_tags=[str(bundle["tag_stem"])])
    window = np.asarray(bundle["window"], dtype=np.float32)
    if window.ndim != 2:
        raise ValueError("smoke window must be [100, N]")
    pred = None
    for row in window:
        pred = decoder.predict(row.reshape(1, -1))
    expected = np.asarray(bundle["expected"], dtype=np.float32)
    if pred is None or pred.shape != expected.shape or not np.allclose(pred, expected, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError(f"container smoke mismatch {pred} vs {expected}")
    if not np.isfinite(pred).all():
        raise RuntimeError("container smoke non-finite")
    print(json.dumps({"status": "CONTAINER_SMOKE_PASS", "pred": pred.tolist(), "kind": "proj_add_ort_d2"}, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-payload", default="")
    parser.add_argument("--smoke-window", default="")
    parser.add_argument("--graph-dir", default="")
    args = parser.parse_args()
    if args.smoke_payload and args.smoke_window and args.graph_dir:
        smoke(args.smoke_payload, args.smoke_window, args.graph_dir)
    else:
        raise SystemExit("use --smoke-payload --smoke-window --graph-dir")
