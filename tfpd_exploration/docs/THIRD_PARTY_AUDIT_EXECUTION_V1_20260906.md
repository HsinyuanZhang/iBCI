# THIRD-PARTY AUDIT — EXECUTION REPORT V1

- 执行者：GLM5.3（用户指定的第三方审核者），2026-09-06T13:0xZ。
- 依据清单：`THIRD_PARTY_AUDIT_CHECKLIST_20260906.md`（含附录 A）。
- 约束遵守：全程只读 + CPU（H1 formal12 双卡训练未受干扰，清单 §交付要求）；未重选 checkpoint、未覆盖任何失败记录、未提交任何官方任务。
- 方法：不采信任何 receipt 的 PASS/哈希声明，关键数字从逐点预测 npz 独立重算，payload 从镜像内只读实测。

## 逐项判定

### P0-1 H1 低 R² 完整计算对齐 — **仅局部支持（本轮部分完成）**

已独立证实：
- ORIGINAL 参照逐位复现：`original_h1_minival_native_float64.npz`（20,325×7，13 sessions）我的公式重算
  pooled 0.960784 / equal 0.960439 / worst 0.9532 = receipt 完全一致。
- 家族 RAW endpoint12 逐点重算：FLAT pooled 0.462044 / ROUTE 0.394599（`h1_fixed_endpoint12_raw_diagnostic_v1/*_complete_float64.npz`），
  与质量 receipt 的 EMA 视图（0.479149/0.419929）差 +0.017/+0.025，方向与量级符合 EMA−RAW 关系；
  但 EMA 视图逐点件未在本轮定位到，**EMA 数字本身尚未独立重算**。
- C2 参照在 receipt 中自带 `independently_recomputed=False` 标注——诚实，但意味着 C2 0.8885 仍是引用值。

未完成（缺证据，需第二轮）：从原始 NWB 时间戳/分箱起的全链路对齐（cache→W700 窗口→标签末端→mask→bank）；
×20/÷20 单位契约的独立验证；`RECONSTRUCTED_NOT_BYTE_VERIFIED` 重建 manifest 与 NWB 重推窗口集的交叉验证。
差距量级（0.96 vs 0.48）之大使得"计算对齐错误"仍是活的备择假设，在上述三项完成前不应把 0.48 读成模型上限。

### P0-2 基线身份与 581919 — **支持**

独立复现链（全部本人实测，非引用原审计）：
1. 本机文件 SHA：旧 seed42 包 `4e4dae…`、声明的 seed44/e8 包 `f2f8cd…`（文件名 `t4_m2_seed44_epoch08_…pkl`，与声明一致）。
2. `docker image inspect`（我直接执行）：镜像 `8de56c58…` 的 Config.Cmd 为 `… --model-path /data/decoder.pkl …`。
3. 只读容器内实测：`/data/decoder.pkl` = `4e4dae…`（旧包）；`/artifacts/t4_m2_seed42_identity.pkl` = `f2f8cd…`（声明包）。
4. 因果机制：子 Dockerfile `FROM spint-t4-m2:movement-t4-empty-s42-4e4dae8f` + 新包仅 COPY 到 `/artifacts`、未覆盖 CMD；
   `decode.py` 的 `--model-path` 默认 `/data/decoder.pkl` 并透传。
结论：本地证据链闭合，**581919 官方分数不能归属给声明的 seed44/e8 payload**（远端是否命令覆盖不可本地裁决——维持原审计的条件式结论，本轮无新证据改变它）。对照 581973 默认路径即所选包，非系统性打包缺陷。

### P0-3 M1 小幅提升 — **仅局部支持（点估计复现；统计显著性不支持）**

独立重算（`m1_p1_frozen_same31252_quality_v1.json` 绑定的三个 npz，文件 SHA 先行核验一致；三份 target 逐位相同）：
- FLAT pooled 0.811652 / ROUTE 0.812196 / ORIGINAL 0.809289，equal-session delta +0.002588/+0.003023，
  worst session（ses-20120928）−0.010900/−0.006523 —— **与 receipt 逐位一致**。
- 块自助（按 start 连续段分块，300/1000 bins 两档，2000 次重采样）：
  FLAT delta 95% CI [−0.0017, +0.0065] / [−0.0015, +0.0065]，P(>0)=0.85–0.88；
  ROUTE delta 95% CI [−0.0011, +0.0069] / [−0.0009, +0.0068]，P(>0)=0.91–0.93。
  **两臂 95% CI 均含零**：+0.0024/+0.0029 在时间相关采样下不可区分于零，不得写成"显著提升/非劣已证"。
- 选择依赖（本人从 receipt 表格读出并复核）：ENDPOINT24 视图 delta 为**负**（FLAT −0.0105 / ROUTE −0.0093），
  正收益仅存在于 EMA 选点视图；ROUTE−FLAT = +0.0005，远低于预注册 +0.003 采纳门。
结论：数字真实、方向为正，但证据等级 = "点估计为正的单种子开发面对照"，任何正式主张需多 seed 或更强效应。

### P0-4 M2 QueryAge pooled 提升 — **支持（数字与程序）；实质 = 局部成功**

- 四 session 逐位复现（我的公式）：FLAT pooled 0.143064（per-session 0.1939/0.2571/0.1271/0.0247）、
  ROUTE pooled 0.296396（0.5160/0.5112/0.3957/**−0.238162**）——与 receipt 完全一致。
- 冻结程序核验：`selection_freeze.json`（02:14:18Z，`SELECTION_FROZEN_PRE_EXPORT_PRE_SCORE`、`no_scoring_yet=True`）
  早于 ext4 评分 receipt（02:23:55Z，`parameter_updates=0`）9.5 分钟；选点规则为 source-minival，FLAT e2 是
  规则产物而非偷看（与 S1 的 e2 现象同构——minival argmax 的已知偏早行为）。
- 实质判读：ROUTE 的 pooled 提升完全由 10-30 两 session 驱动，11-19 灾难性崩溃（−0.238）；
  FLAT pooled 低于 Original。receipt 自带 `qualification=development comparison…not untouched heldout`——诚实。
- 残缺：Original 0.2292 参照的逐点件本轮未定位重算（缺证据，次要件）。

### P1-8 网络族表述 — **仅局部支持（构造器级已证，forward 级未做）**

构造器/继承链实测：M2 `M2QueryAgeFamilyDecoder(M2FamilyDecoder(SmallTransformerDecoder))`（S1 小 Transformer 血统）；
H1 显式导入 `make_v2_unscaled_dot_localbalanced_pair`（unscaled-dot/local-balanced preset 真实存在且被披露）+
三任务共享 `QueryTemporalStack`（current_query_v2）与 `FamilySetFrontend`。与既有
`AUDIT_QUERYAGE_THREE_TASK_NETWORK_FAMILY_20260906.md` 的方向一致。未完成：state-key 集合比对、
ROUTE 零门与 FLAT 的数值对齐复验、H1 preset 例外在论文表述中的落地检查。

### 本轮未执行（缺证据/需资源）

P1-5 数据暴露全景表、P1-6 加速 runtime 与 selected 权重完整流式等价的重证明、P1-7 latency 同配置独立重测
（CPU 可做但需数小时专用窗口，且共享主机干扰需按清单披露口径重设）、P0-1 的 NWB 级深对齐。
另：P1-10 已知 dual_track 旧根 receipt 0644 未封存问题未修复，本轮我对所有关键 npz/json 先记哈希再读，
未发现读取窗口内有变动（mtime 均早于本轮审核开始时间）。

## 总裁定

| 项 | 判定 |
|---|---|
| P0-1 | 仅局部支持（ORIGINAL 与 RAW 数字复现；全链路对齐未做） |
| P0-2 | **支持**（本地归属失败坐实；远端不可裁决维持条件式） |
| P0-3 | 仅局部支持（点估计复现；95% CI 含零；选择依赖） |
| P0-4 | 支持（数字与冻结程序）/ 实质局部成功（11-19 崩溃） |
| P1-8 | 仅局部支持（构造器级） |
| 其余 | 缺证据（本轮范围外） |

对现行为的最小影响：M1 家族收益主张应降级为"点估计为正、CI 含零"；M2 ROUTE 主张必须附 11-19 −0.238；
581919 官方分不得引用为 seed44/e8 的成绩；H1 家族数字在 NWB 级对齐完成前不得读为模型上限。

## 复现命令（关键三条）

```bash
# P0-2 镜像内 payload 实测
docker run --rm --network none --read-only --entrypoint /usr/bin/sha256sum \
  sha256:8de56c58… /data/decoder.pkl /artifacts/t4_m2_seed42_identity.pkl
# P0-3/P0-4 重算：见本报告方法节（npz 路径 + 1-SSres/SStot 中心化公式 + 按 start 连续段分块自助）
# P0-1：original_h1_minival_native_float64.npz 同公式重算
```
