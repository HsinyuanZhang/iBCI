# M2 joint multi-seed EXT6 候选验证状态

本文件是对四个本地 RIFT EXT6 selection receipt 的只读清单，不训练、评分、打包或提交。所有 RIFT 候选都扫描同一 six-session query surface 的 24 个 EMA checkpoints（epochs 1–24），以六个 session R² 的未加权 arithmetic mean 选择 earliest maximum。每个 selected row 都声明 `partial: false`、`n_windows: 15403` 与 `view: EMA`；完整 receipt 在 top level 声明 `official_test_used: false` 与 `evalai_opened: false`。

| 本地 RIFT 候选 | Seed / selected EMA epoch | six-session mean | Δ vs RIFT concat e9 | manifest SHA-256 | score receipt SHA-256 | canonical validation audit |
| --- | --- | ---: | ---: | --- | --- | --- |
| concat | 42 / e9 | 0.3900576650553506 | 0 | `5d1bb0bf640c3d6ef646a0a6f7366bd2d3ff6befc506325da2d0c7aa7ca38884` | `6a985e1f3fb9392198725d6920efb7ea24b41052ddae84a335ae66ea72022e31` | 已存在：`26158347862d2e48a8baae13a72b7ec3cdf0cfc4e7a29f45541d8f0d1de8621b` |
| joint D | 42 / e13 | 0.3478275158419855 | -0.0422301492133651 | `cdefd513a9c9eba240e0baf7ad045c720599354ba89abf301258669200817fb3` | `ba4af2892bedbaa9766b27642624e9ad6438a4515904ea40724a6a219220b5b6` | 已存在：`5d70094307e15eea99642944fc9f4ed22d267c252f7e4f561376f896669481ac` |
| joint D | 43 / e8 | 0.385522723151031 | -0.0045349419043196 | `e1d1df7bb268f505e59e4c7700ef95f4cd3700a736151b3798da48e21c0a481e` | `94408ec5acbe7da6a3bda56d9a4e2e92cbfdf454ed6ef4d8e0e7b540d64ea5c1` | 已通过：`1e374d0d804599ee4f1249d86e464ab9fc6832027a18aa0582b99599542462dd` |
| joint D | 44 / e11 | 0.36275330837379643 | -0.0273043566815542 | `98d72f930567261a565d92bb4e3493f2734afe19aa67005bb14d1509dc603a96` | `5dafe85de481e677f2eaee0840441d2c46f2e3437cc6d114bcef8429207986d4` | 已通过：`3c395884e1937a40468c1c93f8335e9f09e6a49bb0a86f9e801f90b67632518c` |

D43/D44 的 canonical `validation_audit.json` 均已通过 shared-state-reference auditor。每个 audit 覆盖 279 项检查，包括全部 24 checkpoints 的 identity/hash/raw-EMA-finite binding，以及 selected EMA package 121 个 state entries 的严格重建和 byte equality：90 个 unique named parameters、30 个 parameter-alias state keys、1 个 buffer。auditor SHA-256 为 `bde4da83bfcfb9c7e1796409ce89346e3c19af5a99e4c7d8ea5a2dbbac852a96`。

## Selected six-session rows

| Candidate | 10-30 R1 | 10-30 R2 | 11-18 R1 | 11-19 R1 | 11-24 R1 | 11-24 R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| concat e9 | 0.5847563901572550 | 0.5032298447602720 | 0.33396945921081034 | 0.24492190332616237 | 0.3907147752721388 | 0.28275361760546436 |
| joint D42 e13 | 0.5606938984567025 | 0.4257687432760734 | 0.27752232330152526 | 0.28128051584353864 | 0.3511260116115470 | 0.19057360256252598 |
| joint D43 e8 | 0.5268030189874273 | 0.47150887652726137 | 0.3532410967289048 | 0.3018228430626483 | 0.4146698406086974 | 0.24509066299124682 |
| joint D44 e11 | 0.5891281010209151 | 0.43186952222337716 | 0.3433514699555087 | 0.2747483730161896 | 0.41185888304187324 | 0.12556350098491498 |

第一张表的 delta 比较 common local EXT6 surface 上各自独立选择的 checkpoint。它们只是 candidate inventory 数值，不是 paired mechanism estimate、training-seed effect、confidence interval、p value 或 official-test result。

## External reference boundary

较早的 BT SMALL concat submission 582047 在同一 six-session local EXT6 protocol 上选择 EMA e8，值为 `0.3990687579684978`。它属于独立的 external model family，不是 local RIFT candidate。RIFT concat e9 在该 aligned local surface 上低 `0.0090110929131472`；这种比较不建立 architecture effect 或 official-result ordering。其 provenance 见 `docs/M2_SUBMISSION_EPOCH_PICK_ALIGNMENT_20260908.md` 与 `docs/M2_CONCAT_EXT6_FULL_CURVE_ALIGNMENT_20260908.md`。

早先 D44 `validation_audit_attempt1.json` 保留为 provenance：其失败来自 auditor 的 alias-reference basis 问题。修正没有改变训练、checkpoint、EMA package 或 scoring receipt。 

本 inventory 不产生 final rank、submission recommendation、EvalAI action 或任何 688 claim。D43/D44 的 canonical final audits 已经验证，但其 local candidate rows仍只表达本地 EXT6 selection evidence。
