# M2 AOF-S Local Build V1 结果与恢复边界

日期：2026-09-03  
状态：**FAIL（receipt codec 错误；没有生成 payload、Docker 或性能结果）**

## 1. Immutable result

结果根：

```text
tfpd_exploration/results/m2_aof_scalar_static_package_v1/local_build
```

精确成功前缀/失败拓扑：

```text
attempt.json
attempt.json.sha256
predecessor_authority.json
predecessor_authority.json.sha256
failure.json
failure.json.sha256
```

三个 body SHA：

```text
attempt.json
8b25aeef0a5c6b39b97e7a5685e37fc69bbdf66bbaeef3bc0a3f7abcc9811872

predecessor_authority.json
c05e67949d3acc4599d9f996422af9e584342d9440463614a0bbbddaafb81df7

failure.json
65a7b119f4899e51340826c92e8b35e2084bdd8d474ffe4021eff5829293a2d2
```

全部六个叶子均为 `0444`、`nlink=1`，basename-bound sidecar 复核通过。
attempt 绑定的 30-leaf source closure：

```text
568f544f14a962dccc3c3bfe80814f4c3f39d579937814f92bfd4613298654ad
```

## 2. Failure point

生产链已经完成：

- externally reviewed closure map/digest admission；
- fresh canonical result/artifact root admission；
- immutable attempt publication；
- AOF-M 与 exact official 581361 predecessor authority publication。

它在第一个 source session `ses-2020-10-19-Run1` 的 payload 构建 authority
核验中停止：

```text
ExportError:
581361 activity/support/raw-T4/side/native identity drift:
ses-2020-10-19-Run1
```

停止发生在 payload 写入、artifact root 创建、source same-window replay、host minival、
Docker build 和 container minival 之前。没有 EvalAI、网络、GPU、optimizer、target update
或新的 R2。

## 3. Bounded diagnosis

对同一个 source session 使用完全相同的 loader 与 `build_static_identity_inputs` 做只读
逐字段差分，结果如下：

| 字段 | 当前重建 | exact c51 receipt | 结果 |
|---|---|---|---|
| native identity SHA | `41248f...052f` | `41248f...052f` | exact |
| activity SHA | `a2c1bc...544f` | `a2c1bc...544f` | exact |
| normalized side SHA | `568e7d...0adb` | `568e7d...0adb` | exact |
| raw T4 SHA | `493434...7085` | `493434...7085` | exact |
| normalized T4 SHA | `baee7e...be24` | `baee7e...be24` | exact |
| selected-index SHA | `05c8f0...5fd` | `05c8f0...5fd` | exact |
| side-evidence indices | `[1,3,11,13]` | `[1,3,11,13]` | exact |
| top-level `record.selected_indices` | `[1,3,11,13]` | field absent | **codec error** |

因此 failure message 中的笼统 “drift” 不代表历史数据、T4、identity 或选择算子漂移。
真实原因是 V1 exporter 同时要求：

```python
evidence["selected_indices"] == record.get("selected_indices")
```

但 official c51 receipt 的合法 schema 只把该字段放在：

```text
record.side_evidence.selected_indices
```

V1 已经另行严格核验了这个真实位置及 `selected_indices_sha256`。新增的 top-level 要求
是实现者臆造的重复字段，必须删除，而不是给真实历史 receipt 补字段或放宽其他核验。

## 4. Scientific interpretation

本次 V1 没有生成 prediction、R2 或 container score，因而：

- 不能读作 AOF-S 正结果或负结果；
- 不改变 source OOF scalar `+0.00635837`、`6/7` 的开发证据；
- 不改变 exact 581361 official comparator `0.2897439880`；
- 不授权任何 EvalAI 提交；
- 不允许覆盖或删除 V1 immutable failure root。

它只证明 V1 production receipt codec 比 official c51 schema 多要求了一个不存在的顶层字段。

## 5. V2 recovery contract

允许的最小 successor 只包含：

1. 新的 package/result/artifact namespace；
2. held-FD、no-follow、exact six-leaf 绑定本 V1 failure graph；
3. `selected_indices` 只从 official `side_evidence` 读取；
4. 继续 exact-check `selected_indices_sha256`、activity、side、raw/normalized T4 与 native
   identity；
5. 完整复用 V1 route-owned payload、source bridge、host minival、offline Docker 和 offline
   container-minival 生命周期；
6. 新闭包、独立 root-reviewed non-self-referential map/digest；
7. no-data tests 先用 actual c51 receipt 证明不再要求 invented top-level field。

V2 不得改变 beta、decoder、post identity 算子、13-session roster、checkpoint、T4、support、
behavior scale、Docker base 或网络 NO-GO。若 V2 再失败，必须保留自己的 immutable failure
root；不得回写 V1。

