# sub-M 跨动物 V3R2：Native-M2 后执行交接

状态：`READY_BUT_NOT_AUTHORIZED_PENDING_NATIVE_M2`

## 已冻结内容

- 15 个 sub-M center-out sessions；`sua` 与 deterministic `pseudo_mua` 两个 view。
- `shared_t4` 与 `shared_ts4` 两臂，seeds 42/43/44，共 180 cells。
- activity carrier 只用 chronological first 30 rewarded trials。
- T4/TS4 fit pool 用 chronological first 50 rewarded trials。
- query 只来自 rewarded trial 50 之后的 valid windows。
- 结论边界仅为 paired `T4-TS4` attachment/content contrast；不是 T4-vs-SPINT，也不是 native threshold-MUA claim。

不可变 prelaunch：

- draft SHA-256：`13f113de25b20621681282fece849f130c3bc9c8c82fbb10eca4aa803400dd74`
- receipt SHA-256：`b2f68bf29582486f1cb9a85c79d23e52239d34decfecb7838a7fa59c8fabea8e`
- seal SHA-256：`89cbf98e826a52120a9b50524694eaa4e49444ba0bcb4e220aa5222949647e37`
- execution-policy SHA-256：`9da10303f59c054b50560bf24b30322d615e0627fb1d5d8ea19bb06947407b2a`
- review-template SHA-256：`9fc8439d679b3b32d64edb6e2745a70aaea0b125487727e84c1358a669ffcb0b`

当前 dry-run 已证明：authorization、nonce、checkpoint/NWB/normalizer access、model forward、R2 均为 0。sub-M endpoint 尚未评分。

## Native-M2 完成后唯一允许的顺序

1. 取得 Native-M2 completion receipt，并重新检查本文件列出的 source/prelaunch hashes、运行 host/Python、external-NWB root 与 fresh output root。
2. 生成一次短时、未签名的 canonical V3 envelope：

   ```bash
   /home/xinyuan/miniconda3/envs/spint/bin/python \
     sua_exploration/scripts/prepare_dandi688_subm_co_score_only_authorization_v3r2.py \
     --mode unsigned-envelope \
     --native-m2-completion-receipt /ABS/PATH/native_m2_completion_receipt.json \
     --authorization-id subm_co_v3r2_YYYYMMDD_HHMM \
     --score-output-root /home/xinyuan/Work_host/SPINT/sua_exploration/results/dandi_000688_subm_co_score_only_v3r2_runs/RUN_ID \
     --external-nwb-root /ABS/PATH/verified_subm_nwb_root \
     --output /ABS/PATH/subm_co_v3r2_authorization.json \
     --validity-seconds 600
   ```

3. 独立复核 envelope 后，在 workspace 外使用 pinned Ed25519 private key 对 canonical JSON 原始字节生成 detached base64 signature。准备脚本不会读取、生成或保存私钥。
4. 在签名有效期内启动一次 CPU-only execution：

   ```bash
   CUDA_VISIBLE_DEVICES='' /home/xinyuan/miniconda3/envs/spint/bin/python \
     sua_exploration/scripts/run_dandi688_subm_co_score_only_v3r2.py \
     --mode score \
     --authorization /ABS/PATH/subm_co_v3r2_authorization.json \
     --signature /ABS/PATH/subm_co_v3r2_authorization.sig.b64 \
     --output-root /home/xinyuan/Work_host/SPINT/sua_exploration/results/dandi_000688_subm_co_score_only_v3r2_runs/RUN_ID \
     --external-nwb-root /ABS/PATH/verified_subm_nwb_root
   ```

5. 只在 180/180 immutable prediction/metric cells、matrix seal 与 NPZ-recomputed aggregate 全部闭合后解释结果。部分输出不允许重试，也不允许据此调整 gate。

## 当前禁止事项

- Native-M2 完成前生成/签署有效 capability 或 claim nonce。
- 提前打开 sub-M endpoint、checkpoint 或 R2。
- 修改已冻结的 session、seed、checkpoint、normalizer、30/50/50 budget、gate 或 query window。
- 把 `T4-TS4` 写成 `T4-SPINT` 或 absolute T4 gain。
- 使用本 review template 直接执行；它明确不是 authorization envelope。

历史 V5/V5R2 测试对两份后来追加 `SUPERSEDED` 横幅的 Markdown 使用旧完整文件 SHA，因此在混合回归中会按设计失败。V3R2 对两个 276-byte 横幅逐字节验证，并证明去掉该前缀后的正文 SHA 与历史 seal 精确一致；没有修改旧 seal 或旧正文。
