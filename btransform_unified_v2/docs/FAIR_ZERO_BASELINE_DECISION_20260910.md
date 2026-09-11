# 公平零线 / 协议基线记录（2026-09-10）

论文主对照仍是各任务 official FULL 对 official `ACTIVITY_ONLY`。本文件只记录“新 session 不使用行为标签身份”这一条额外线：M2/H1 能写出合理本地数字的写数字；M1 不写 srcbank / literal-zero 分数，改为记录待做的发放率重训。本地 HO3 / EXT6 / HO-M3 不是 official，不得与 official HO 相减。未授权提交。

## 决定

M1 的字面零 `NONE`（official `582224` HO `-0.677`）和 source-frozen srcbank（local HO3 last_source `-0.753`、source_mean `-0.918`）都不进主表。崩塌顺序与 HO 相对 source 的发放率抬升一致（约 2.1× / 3.3× / 1.3×），不是 EMG DC 跨日对不齐。`NONE` 只作无身份下限，可留附录。

若 M1 还要一条低于 ACT 的公平线，构造是：**只用当日 M10 无标签逐单元平均发放率作身份，T=0，重训**（NORM_ONLY）。不在 FULL 权重上只换 bank。尚未开训、尚未封包。本地门槛：HO3 chR² ≥ 0.3，且每 session DC 罚项 < 0.1；若贴着 ACT local `0.561827`，须改写 ACT 的论文句子。

M2 的 last_source 数字合理，写入下表。H1 srcbank 于 2026-09-10 用户叫停：`own` 已收回 `0.467651`，`last_source` / `source_mean` 未完成；本路线不再继续，改由另一 agent 做新的消融。

## 本地协议基线（非 official）

| 任务 | 本地面 | 权重 | 构造 | 等权分数 | 对照（同面，非 official） |
| --- | --- | --- | --- | ---: | --- |
| M1 | HO3 chR² | 待重训 | 当日无标签逐单元平均发放率，T=0 | 未跑 | FULL local `0.569075`；ACT local `0.561827`；NONE local `-1.215` |
| M2 | EXT6 等权 R² | `582189` e9 EMA，不重训 | 三 HO/EXT tag 复制 `ses-2020-10-28-Run1` 的 E0/T，保留各 query `unit_mask` | **`0.258669953325137`** | own `0.390057879406667`（收回参照 `0.3900576650553506`）；ACT local `0.18402467250439059`；NONE local `-0.012659512830027883` |
| H1 | HO-M3 grouped-7 | `582196` e16 EMA，不重训 | 复制 `ses-19250120T115537` 的 E0/T | **已取消**（仅 `own` `0.467651`；无 last_source） | FULL local `0.46765143362182887`；ACT local `0.2660451704314089` |

M2 source_mean 为 `0.08983019607474484`，低于 last_source，不作为写入值。M2 last_source 收据 [m2_srcbank_ext6_v1/receipt.json](../results/final_ablation_official_v1/m2_srcbank_ext6_v1/receipt.json)。M1 srcbank 诊断收据 [m1_srcbank_ho3_v1/receipt.json](../results/final_ablation_official_v1/m1_srcbank_ho3_v1/receipt.json)，仅作否决证据。
