# M2 AOF-S V2 本地构建结果（Docker SDK 环境失败）

日期：2026-09-03  
结论：**工程失败；不是性能结果；不得据此判断 AOF-S 有效或无效**

## 1. 发生了什么

AOF-S V2 在独立复核 37 项闭包并通过 V1+V2 focused tests `25/25` 后，执行了唯一一次
local-only production build。执行环境明确设置 `CUDA_VISIBLE_DEVICES=''`，未访问 GPU，
也没有 EvalAI、ECR、registry、push、login 或其他网络提交。

V2 已成功完成：

- exact V1 failure predecessor revalidation；
- c51/581361 的 13-session side-evidence receipt codec 修复；
- 新 payload 生成、0444 sealing 和 basename-bound sidecar；
- immutable `attempt.json`、`predecessor_authority.json`、`input_authority.json` 发布。

随后在 Docker build 入口导入 Python Docker SDK 时失败：

```text
ModuleNotFoundError: No module named 'docker'
```

宿主机 `/usr/bin/docker` 与 frozen local base image 均存在；缺失的是当前 `spint` Python
环境里的 `docker` 模块。没有联网安装依赖，也没有放宽 offline container 约束。

## 2. Immutable V2 failure graph

根：

```text
tfpd_exploration/results/m2_aof_scalar_static_package_v2/local_build
```

根恰好有 8 个 leaves（4 个 body + 4 个 sidecar），均为 `0444`、`nlink=1`：

```text
attempt.json                 e400fe4240cc37e1f7c31e1355dd63a1978d0762c4a2b09f700b9a434d54d768
predecessor_authority.json   587dc61e6f4de9516384911dffdbf7d394472c3104d9217a3168adace80245cf
input_authority.json         f1695edc4155b0270e5663eaf036e86113e16887638a989bc91d50d8e409fe41
failure.json                 9ee7000ec553b7e047f563a6002feeb1a78666519cea1063da0b7b826cb5e22e
```

`failure.json` 如实记录：

```text
published_prefix = [attempt.json, predecessor_authority.json, input_authority.json]
exception_class = ModuleNotFoundError
exception_message = No module named 'docker'
network_submission = false
```

没有 `build.json`、`validation.json` 或 `terminal.json`。

## 3. 已封 payload（非终态候选，禁止直接提交）

V2 在失败前已经生成并封存 4-leaf artifact graph：

```text
tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2/
```

```text
t4_m2_seed42_dopt4_act30_aofs_identity.pkl
  65b8001156a7cdb58efd4bbff9547d88443cfa983546558009771f3d4c18ab20

t4_m2_seed42_dopt4_act30_aofs_identity.receipt.json
  b17e5dc09d99e84ee570baedae68c7a4dae84c0bf7923705a02f6c19704ca336
```

receipt 记录 `session_record_count=13`，并绑定 V1 failure predecessor 与 V2
`side_evidence.selected_indices` codec。由于 Docker build、host/container minival 和 terminal
尚未完成，这些 bytes **不是可提交终态**。

## 4. 合法 successor

只能使用新的 V3 root，held-FD 绑定上述 V2 8-leaf failure graph 与 4-leaf artifact graph。
V3 唯一允许变化是把 Python Docker SDK client 换成 route-owned、固定 argv 的
`/usr/bin/docker` CLI：

- build：`--network=none --pull=false`；
- run：`--network=none --pull=never --rm`；
- 禁止 shell、login、push、registry、pull 或任何网络提交；
- 复用已封 V2 payload/receipt，不重新选择 beta、session、support 或方法；
- 仍必须完成 host minival、offline container minival、输出 SHA sealing 和 terminal/failure XOR。

因此 V2 不提供新 R2；AOF-S 当前科学状态仍是 source OOF `+0.00635837` 候选，等待 untouched
local/offline package 验证以及未来独立授权的官方评测。
