# M2 AOF-S V3 本地构建结果（容器文件布局失败）

日期：2026-09-03  
结论：**工程失败；不是性能结果；不得据此判断 AOF-S 有效或无效**

## 1. 已完成与失败位置

AOF-S V3 在 root 独立重跑 V1/V2/V3 focused tests（`29 passed, 5 skipped`）、逐字节复核
45-leaf closure，并确认本地 base image ID 后，执行了唯一一次 CPU/offline production build。
执行未使用 GPU，不包含网络、pull、push、registry、login 或 EvalAI 提交。

V3 成功完成：

- exact V2 8-leaf failure graph 与 V2 4-leaf sealed payload graph revalidation；
- 复用 V2 payload，不重算 beta、support、identity 或 decoder；
- host-side input authority publication；
- `/usr/bin/docker build --network=none --pull=false`；
- local base image ID 和 built image ID 验证。

容器 minival 随后在读取数据、构造 decoder 或产生 prediction 之前失败。V3 Dockerfile 将
`aofs_static_decoder.py` 平铺复制为：

```text
/workspace/aofs_static_decoder.py
```

但该文件的 host/package bootstrap 按仓库内路径计算：

```python
_REPO_ROOT = Path(__file__).resolve().parents[3]
```

平铺路径没有第四层 parent，因此异常为：

```text
IndexError: 3
```

这是容器文件布局错误，不是 payload、模型、beta、M2 数据或 R2 结果。

## 2. Immutable V3 failure graph

根：

```text
tfpd_exploration/results/m2_aof_scalar_static_package_v3/local_build
```

根恰好有 8 个 leaves（4 body + 4 basename-bound sidecar），均为 `0444`、`nlink=1`：

```text
attempt.json                 554afff1d91855fdaec905350170635d9a838c1b2b92957dd3997eead5ae4a13
predecessor_authority.json   9bdbea7dc088695fe60465409d657c95cac7a6b465abbbdf97bce76da0d105d7
input_authority.json         b6f66cb7cb404240e019201fed41a1721128807861a47099ed5726e0611bf8ed
failure.json                 40573a526803614fcda0652d202ba0b4a4362478fe7df01c666486473d20e8c2
```

`failure.json` 如实记录 prefix：

```text
attempt.json
predecessor_authority.json
input_authority.json
```

没有 `build.json`、`validation.json` 或 `terminal.json`。V3 artifact root 只有 sealed
`reused_v2_payload_authority.json` pair 与一个无 prediction/target 的空
`container_minival/` 目录；不得称为 validated artifact。

V3 使用的 externally reviewed closure：

```text
377824b212ff31354a9e25034d1fe128834aa9aab82ab3d36b819441948e1239
```

已构建但验证失败的 local image：

```text
tag spint-m2:aof-scalar-static-v3-local-v1
id  sha256:d05523e6448635a17b93480a99b0c03fb718369cc037aa3895d256d09bee4ac9
```

它没有被 push，也不是可提交终态。

## 3. 合法 V4 successor

V4 必须使用新 result/artifact root 并 held-FD 绑定上述 V3 8-leaf failure graph，同时继续
绑定 V2 4-leaf payload graph。唯一允许的变化是容器内文件布局；科学 bytes 必须不变：

- V1 `aofs_static_decoder.py` body 不修改；
- V1 `laws.py`、`decode.py` body 不修改；
- V2 payload SHA 保持
  `65b8001156a7cdb58efd4bbff9547d88443cfa983546558009771f3d4c18ab20`；
- 将 decoder 放在足够深的 package-compatible path，使 `parents[3]` 合法；
- `decode.py` 仍为 entrypoint，payload 仍为 `/data/decoder.pkl`；
- Docker 仍为 local-only：build network none/pull false，run network none/pull never/remove；
- M2 data 只读挂载 `/input`，fresh output 挂载 `/output`，四个 evaluator env 精确绑定。

V4 必须重新完成 host minival、offline container minival、prediction/target SHA sealing 和
terminal/failure XOR；V3 不提供任何新 R2。
