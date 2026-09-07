# Result: H1 Cross-Record Anchored Post-Pool Screen V1

日期：2026-09-03  
状态：`COMPLETE_STOP_DIRECT_IDENTITY_FUSION`

## 1. 结论

在五个 source-grouped date-LODO C1 checkpoint 上，从 held-in-calib 的 M3
身份冷启动到对应 held-in-minival recording，冻结权重的 post-MLP pooling
没有产生可提交的正结果。预注册选择的五折均值为 `-0.001379 R²`，4/5
日期非负，worst `-0.006895`；未达到 `+0.005 / 4-of-5 / worst>=-0.010`
的通过门。

四折因 source-only 选择直接回到 `RP-G000`，唯一选择非零记忆的日期为
19250115：source 选择 `RP-G050`，outer-date 反而为 `-0.006895`。因此本结果
不支持把冻结 C1 的 post-MLP activity identity 直接部署到新 recording。

## 2. 描述性固定臂

这些臂在 outer date 上只是诊断曲线，不能替代 source-selected primary：

| arm | mean delta | positive dates | worst date | post-commit mean delta |
|---|---:|---:|---:|---:|
| RN-G020 | +0.002086 | 3/5 | -0.003061 | +0.004221 |
| RN-G050 | **+0.003181** | 3/5 | -0.009245 | **+0.007119** |
| RN-G100 | -0.000734 | 3/5 | -0.023354 | +0.001886 |
| RP-G005 | -0.000381 | 2/5 | -0.002203 | -0.000632 |
| RP-G010 | -0.000846 | 1/5 | -0.004515 | -0.001409 |
| RP-G050 | -0.008063 | 1/5 | -0.027439 | -0.013814 |
| RP-G100 | -0.027568 | 0/5 | -0.066426 | -0.047992 |

RN-G050 表明新 recording 的 activity 不是完全无信息；但它只在原生
early-pool 算子中产生小幅、非一致收益。RP 随 gate 增大单调恶化，符合
“为 early-pool 训练的非线性被直接换成 post-pool 后发生算子离群”的解释。

## 3. 正确边界

- 可以说：同 recording 真 trial 的 `+0.078573` headroom 没有直接迁移到
  calibration-recording→new-recording 的 frozen post-pool deployment。
- 不可以说：MLP 后 pooling 从头联合训练一定无效。本格只检验冻结 C1 的
  identity-space 正 gate。
- 不按同一设计追加 12 epoch RP 重训；原设计已经预注册 RP 负时关闭该路线。
- 允许一个科学上不同、事前冻结的新 successor：保留完整 post-pool候选，
  在 source dates 上闭式拟合 signed scalar output residual，然后严格 LODO。
  它检验 post-pool 是否含有方向相反但可校正的互补信号，不修改网络权重。

## 4. 权威收据

- result root：`tfpd_exploration/h1_series_20260830/results/h1_cross_record_anchored_postpool_v1/`
- attempt SHA256：`6537d85255dcdf077fea2e1c290d6f84fe250d58553ad0c9db41588fb282dc66`
- score SHA256：`62ebd18db6ce504c01414bd392a84a617c2b2aa2049af1e68b90e34bfc238d83`
- terminal SHA256：`ec86e26c513c3ccb9ab4ca5f8d0f0ce244e6b162b507c635f581078559a1804a`
- target optimizer/backward/model updates：全部 0；formal held-out/EvalAI 未打开。

