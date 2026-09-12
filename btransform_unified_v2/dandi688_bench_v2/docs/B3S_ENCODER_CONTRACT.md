# B3S encoder contract

本文件保留 B3S class/schema 与迁移前 CPU smoke 的历史定义，不再是 DANDI688 当前实验线的名称。当前 encoder 合同为 [CONCAT encoder contract](CONCAT_ENCODER_CONTRACT.md)：它使用相同的 `pre_pool → trial mean → post_pool` 骨架，并将 Full 的 MOVE-T4 融合统一命名为 encoder-input concat。`concat` 不指 M2 RIFT decoder 的 P16/proj_add frontend。

这是 DANDI688 后续神经训练唯一允许使用的 encoder 结构名称。它与 M1 同系列，固定为：

```text
support[M33, T100, N_pad=100]
→ pre_pool: Linear(T100, h64) + ReLU
→ trial mean over M33
→ post_pool: 3-layer affine/ReLU stack → E0[N_pad, 50]
```

Full 在 trial mean 后、`post_pool` 前拼接同一表示、同一目标 session 的 MOVE-T4 carrier side；Full 的 carrier 为每 unit 4 维。Full 的 `post_pool[0].weight[:, hidden_dim:]` side columns **必须零初始化**，使空/零 side 在初始化时严格回到 pooled B3S 路径。ACT 只执行 pooled activity 路径，不读取 carrier。Raw-set 没有 encoder，不读取 M33 support 或 carrier，而是维持既定 source-trained global `g[16]` 输入。

side columns 在 source 预训练中保持可训练；零初始化只保证初始 E0 不受 side 影响，不表示整个 Full decoder 不读取直接 T4，也不自动保证与独立初始化的 ACT 参数逐字节一致。

该顺序和 zero-init 条件以 M1 的真实实现为准：[SideFeatureEarlyPoolEncoder B3S](../../../streaming_calibration_exp/src/models/components/streaming_encoders.py)（`variant=B3S`）及 [encode_b3s wrapper](../../../btransform_unified_v1/src/btransform_unified_v1/m1_b3s_joint.py)。M1 的 pretrained B3 权重不能直接迁移到 DANDI，因为 support T/E 语义不同；DANDI 的 SUA/PMUA 必须各自在新的 18-source surface 上独立预训练。

除 encoder interface 外，M2 RIFT decoder、R50、P16/proj_add、learned temporal module、session-balanced sampling、batch32、24×3165 updates、EMA、dropout 和 source/development/final 协议不因该命名统一而改变。

无 FiLM B3S Python 实现已完成，并通过 [two-source CPU smoke](../results/smoke_b3s_2015_m33_v1/receipt.json)；该产物严格标为 `SMOKE`，没有正式训练、选择、开发评分或 final 评分资格。当前旧 Full 的历史实现含 empty-contrast FiLM，且其 carrier path 不满足上述 B3S side-column zero-init 条件。仅删除 FiLM 并沿用既有权重不满足新合同；实现还须对齐 side columns 的零初始化，并从 DANDI source 重新预训练。该旧实现启动的 SUA/PMUA pretrain 都在 4 segments（12,660 updates）后由用户终止；其权重没有 B3S 资格，不得恢复或用于选择、开发评分或 final 评分。见 [终止收据](../results/formal_campaign_2015_m33_allseeds/TERMINATED_BY_USER.json)。
