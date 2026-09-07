# RESULT_FABLE_TKD_M2_V1 — TKD 在 M2 上的结构性负结果与结案

Date: 2026-09-05
Type: 结案文档（预注册判定 ADDENDUM-5/6 降级条款执行）
Workorder: `tfpd_exploration/docs/WORKORDER_FABLE_TKD_M2_V1_20260904.md`（含 ADDENDUM-1..6）
Result root: `tfpd_exploration/results/fable_tkd_m2_v1/`（60+ receipts，全部 0444+sha256 sidecar）
Status: **M2_LINE_CLOSED_BY_PREREGISTERED_DOWNGRADE**（全格 16 runs 与 Wave3 未执行——在 ≈0 信号上测门无意义）

---

## 0. 一句话结论

**在 M2 上，"T4 键控的 0.2M 流式 decoder" 无法跨 session 迁移**：蒸馏后 within 面可达 champion 水平（0.52–0.65 vs 0.689），但外部 6 session 的 R² 在任何 epoch、任何选择律、任何 λ_E 下都不超过 **+0.074**（champion 0.295）。G1 非劣门结构性不可达，G3 identity 门（SHUF/POOL）失去可测信号。预注册降级条款触发，程序按计划转向 M1。

## 1. 证据链（全部 sealed，按时间序）

| # | 证据 | 数字 | receipt |
|---|---|---|---|
| 1 | A1 PV 锚（数学） | 最终 min corr **0.9975** 纯净 PASS（β=28/uniform，D12 值域后） | `stage0_r4/a1_pv_anchor.json`、`stage0_r6/` |
| 2 | Stage A 基线：校准后经典 PV | within **0.0317** / external **0.0278** | `stage0_r5/baselines.json` |
| 3 | Stage A 基线：ridge(rate10→y×5) | within **0.2438** / external **−0.0260** | 同上 |
| 4 | from-scratch pilots r4–r8（D8–D13 逐项修复后） | best: within 0.161 / external 0.036 | `runs/A2_s42_a{3,4,5}/`、`stage1_failure_r{2..6}.json` |
| 5 | 蒸馏 r9 A2-SSM（D15 全有界化） | 仍有 −11.9 破坏性 episode → **SSM 证伪** | `runs/A2_s42_a6/` |
| 6 | 蒸馏 r9 A2-GRU | within **0.65**（≈champion 0.689）/ external **−0.053** | `runs/A2_GRU_s42/` |
| 7 | r10 公平选择律（D16 minival 80/20）+ 双 λ_E | λ=0.1: sel-ep16 external **−0.070**（曲线峰 −0.006）；λ=1.0: sel-ep25 external **+0.059**（曲线峰 **+0.074**） | `runs/A2_GRU_s42_le{1,10}/`、`stage1_failure_r8.json` |
| 8 | REF-SHUF 重放（G3 校准） | p0 复现误差 0.0；ts4 均值 0.20746；**Δ_ref=0.0878** | `ref_shuf_replay.json` |
| 9 | A4 成本（唯一幸存正结果） | TKD 流式 409,088 MAC/bin vs champion 84.0M/bin（**205×**）/ 最优缓存下界 55.8M（**137×**） | `stage0_r2/a4_mac.json` |

## 2. 判读

1. **优化问题已被蒸馏+GRU 解决**（r9/r10 曲线干净、单调、MSE 远低于零解地板）——失败不在训练技术。
2. **选择律公平化后数字更差**（r10 vs r9）——r9 的 −0.053 里没有被选择压掉的信号。
3. **跨 session 水平是绑定约束**：外部曲线全程 ≤ +0.074，任何选择律无法到达 0.10。λ_E=1.0 把外部从负拉到 ≈0.06（教师保真度主导），但逐 session 双峰（3 家 +0.15~+0.22，3 家 ≈0/负）。
4. **champion 的 0.295 来自大容量预训练 SPINT decoder 的跨 session 不变性**，而非 identity 选择（T4 0.2952 vs 活动 identity 0.2991 本就平手）；蒸馏能传递"函数"，传不了这种不变性。
5. **闭式基线独立佐证**：PV（0.03）与线性 ridge（external −0.03）说明 M2 跨 session 迁移对"每 unit 线性读入 + 闭式 identity"这一整个类都不成立，不是 TKD 特有的失败。

## 3. 幸存的正结果与可复用资产

1. **成本主张数字**（A4，G2 双口径 205×/137×）——架构级事实，独立于 R² 水平。
2. **A1 锚方法论**：PV 锚构造（仿射 Φ_k 旁路、Σdir(ψ)=0 共模精确抵消、σ 池化常数律、输出尺度校准）经三轮修复后达到 0.9975，可移植到任何"identity 生成读入"的架构断言。
3. **Δ_ref=0.0878**：champion 自身的 T4 内容增量在 M2 外部面仅 ~0.09——修正了"T4 增量量级"预期（688 是 +0.47~0.71），是论文 M2 段的重要 context。
4. **负结果本身**：SSM(对角线性) 在该配置下不稳定（D15 全有界化仍爆）；from-scratch 小 decoder + 纯 T4 适配在 M2 的天花板（external ~0.04）；蒸馏可恢复 within 不可恢复 external。
5. 全套工程资产：runner（preflight/receipt/receipt 律）、teacher 蒸馏缓存管线、minival 选择律实现、16 项单测。

## 4. 偏差与修复全录（D1–D16）

D1 σ 线性化；D2 Φ_k 隐层旁路；D3 champion MAC 维度来源；D5 MAC 口径约定；D6 A1 参照改经典深度加权 PV（原双重 1/m 是规格错误）；D7 β/mass 经 CLI 授权调优；D8 σ floor（被 D12 取代）；D9 SSM a=0.9 + GLU bias 2.0；D10 P 打破 ε 鞍点；D11 ψ 尾列播种；D12 值域池化常数化；D13 ×5 损失 + init 输出尺度校准 + γ=tanh；D14 蒸馏主臂；D15 SSM 状态有界化；D16 minival 选择律。全部记于 `plan.DEVIATIONS` 与各 receipt。

## 5. 对论文的影响（review §4/§5 对照）

- review 的两个设计点（T4 表示；紧凑消费者+零初始化槽）**不受本结果伤害**：M2 本就是"无缺口→无补丁"行；TKD M2 失败的是"独立小 decoder"主张，不是 T4 主张。
- 论文可写的诚实段落：*"我们尝试了 0.2M 参数的调谐键控流式 decoder：PV 锚精确（corr 0.998）但在 M2 近乎空（R²≈0.03）；蒸馏可达到 held-in 平价（0.52–0.65）但跨 session 迁移在所有配置下 ≤0.07。SPINT 级外部成绩依赖于大容量 decoder 的跨 session 不变性，无法经输出蒸馏传给 100× 更小的学生。"* 与 §6 decoder 无关性实验（在 X 上测 T4 增量）互补而非冲突。
- **流程教训**（写进 M1 工单）：minival 选择律、×5 目标尺度、值域池化常数、GRU 时间模型、λ_E=1.0 蒸馏——全部作为 M1 的 Day-1 默认。

## 6. 边界

- 结论限于：M2 外部 6 session 面、0.2M 级容量、闭式 T4/ρ 适配、本调度（lr 3e-4/30ep/batch32，frozen）。未测：更大容量（0.5–2M）、更丰富闭式 identity（如附加校准块活动统计列——仍零梯度）、更长调度、正则化（dropout/早停于固定 epoch）。
- 未执行：全格 16 runs、Wave3（drop/延迟）——降级条款下无信息量；A4 成本数字以解析口径幸存。
- M1 线承接判定（kill-gate 见 M1 工单）；若 M1 亦无 identity 可测面，向用户提出 688 建议（唯一已知 T4 大增量数据集）。

## 7. 证据路径索引

- 工单+6 附录：`tfpd_exploration/docs/WORKORDER_FABLE_TKD_M2_V1_20260904.md`
- 实现规格：`SPEC_FABLE_TKD_M2_V1_IMPL/WAVE2/WAVE3_*.md`
- 代码：`tfpd_exploration/src/fable_tkd_m2_v1/`、`tfpd_exploration/scripts/run_fable_tkd_m2_v1.py` 等
- Receipts：`tfpd_exploration/results/fable_tkd_m2_v1/`（stage0_r1..r6、teacher_cache、runs/A2_s42_a2..a6、A2_GRU_s42_le{1,10}、ref_shuf_replay.json、stage1_failure_r1..r8）
- 测试：`tfpd_exploration/tests/test_fable_tkd_m2_v1.py`（16 项，最终全绿）

## 8. 终局完整性审计披露（2026-09-05 收线时）

全 root 审计：67 个 JSON receipt 的 file-bytes sidecar 全部通过；**teacher_cache 的 7 个 .npy sidecar 使用的是数组 tobytes 摘要而非文件字节摘要**（manifest 字段名 `sha256_bytes` 有误导）——载入数组的内容摘要与 manifest/sidecar 逐一致，内容完整且被绑定，但标准 `sha256sum -c` 会对此 7 个文件误报；复现者应以 `np.load` 后的数组摘要为准（或对照 manifest `array_sha256`/`sha256_bytes` 两口径）。此为约定偏差披露，不涉完整性问题。另：收线时 GPU0/GPU1 均空闲、无遗留进程；M2 线负结果不改变 review §4/§5 的论文主张（见 §5）。
