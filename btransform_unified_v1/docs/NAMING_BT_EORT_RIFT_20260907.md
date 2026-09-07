# 系列命名：BT-EORT / RIFT

日期：2026-09-07。用户锁定。此后口头与新文档用这两名；不改写已封存路径、官方 submission、或正在跑的脚本标识。

| 名称 | 是什么 | 不是什么 |
|---|---|---|
| **BT-EORT** | B-transformer 模型函数 + fast exact-E + ONNX。窗内正弦 PE，无跨窗 KV。H1/M1/M2 的层数、W、concat/proj_add 是实例参数。 | 新 temporal 结构。不是 581982 V3 current-query。 |
| **RIFT** | Recency-biased Incremental Finite-context Transformer。新模型函数（稳定 token、近时偏置、有限 KV）。代码族 `rift_v1`，根在 `btransform_unified_v2/`。 | BT-EORT 的加速补丁。上 ORT 时写 `architecture=RIFT, backend=ORT`，不叫 BT-EORT。 |

旧称 **E-ORT** 只指推理合同。系列名从本日起用 **BT-EORT**。历史收据、EvalAI ID（582044/045/047）、目录 `*_eort*`、环境变量 `*_EORT_*` 保持原字，引用时注明即 BT-EORT。

H1 系统锚点仍是 582044（BT-EORT B2 P16 L200）。它是 incumbent 参照，不是 RIFT 的单因素对照；RIFT 的对照是同初始化的 flat（R0）。
