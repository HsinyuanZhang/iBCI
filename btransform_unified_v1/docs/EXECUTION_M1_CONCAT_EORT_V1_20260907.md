# M1 concat D4 → BT-EORT 骨架

日期：2026-09-07。授权：准备骨架，GPU 空闲后自动开训。不抢 RIFT 卡，不提交 EvalAI。系列名 **BT-EORT**（旧称 E-ORT 只指推理合同）；见 [命名锁定](NAMING_BT_EORT_RIFT_20260907.md)。脚本/目录仍用历史 `*_eort*` 字面。

## 唯一变量

相对 582045（P16 proj_add D2 BT-EORT，官方 HO 0.533）：**identity = concat**。

- token = `cat(local16, E0 100, T4)`，`token_in=120`，无 `e0_proj`
- temporal **D4**（统一默认；D2 已官方丢分）
- 同一 fullsession 4 日、seed 42、AdamW 1e-4、EMA 0.9995、24 ep
- 选模：可见 HO-calib trio，ext6 e18–24。`register:false`

不是 581982 V3 current-query。不是 CONCAT_W100 旧 dest（无 HO-calib pick，不覆盖）。

## 阶段

| 阶段 | 状态 |
|---|---|
| CPU probe | 完成：`fullsession_20260907T0844Z`，213336 windows，6665 upd/ep，params 3557616 |
| train / score / pick | launcher 等空闲 GPU（不抢 RIFT；双空闲时优先 GPU1）。2026-09-07 17:15 授权：10 分钟检查见空卡则立刻开训 |
| BT-EORT pack | pick 之后；`pack_m1_concat_eort_v1.py` 目前只核 pick，不建镜像 |
| EvalAI | 禁止，除非另授权 |

## 命令

```bash
ROOT=btransform_unified_v1/results/m1_concat_eort/fullsession_20260907T0844Z
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  python btransform_unified_v1/scripts/m1_concat_eort_series.py --stage probe --face fullsession --root "$ROOT"
python btransform_unified_v1/scripts/launch_m1_concat_eort_when_free.py "$ROOT"
```
