# Historical M2 581919: default-entrypoint payload mismatch

Read-only inspection on 2026-09-06. No image, payload, submission, registry,
or historical receipt was changed. The existing local image was inspected
with networking disabled and a read-only root filesystem; the remote service
was not contacted.

The local registration/package records bind submission581919 to image
`sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f`.
Its actual default command supplies `--model-path /data/decoder.pkl`, and its
actual `/decode.py` passes that argument directly to `T4CachedIdentityDecoder`.

| Role | Path inside this image | Observed payload SHA-256 |
| --- | --- | --- |
| Read by the default command | `/data/decoder.pkl` | `4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0` |
| Declared seed44/e8 selection | `/artifacts/t4_m2_seed42_identity.pkl` | `f2f8cd4c046a5880e9716d61981cee4aa0652d33e411212fa7be217cef05b051` |

The first payload matches the older local seed42 MOVE-T4/profile-free package.
The second matches the selected seed44/e8 package recorded in the historical
submission metadata. Both files are present, but the default command references
only the first. This is consistent with the child Dockerfile copying the new
payload into `/artifacts` while inheriting a command that reads `/data`.

The payload mismatch is established for the recorded local image. The further
statement about the remote score is conditional: **if the remote evaluator used
that image's default command without a model-path override, the run used the
older seed42 payload.** No remote command-override audit has been performed.
Therefore, the recorded official score cannot currently be attributed with
confidence to the declared seed44/e8 payload. The official number itself is
not changed or discarded by this audit.

For comparison, the current Transformer e8 image for581973,
`sha256:c3297f4c96b70ab00f67a00fb25c5acade47394bccd3b181ab57571b29e720ed`,
has its expected payload at the default `/data/decoder.pkl` path, SHA
`4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4`.
Its runtime and entrypoint files also match the existing local e8 package.

The additive [machine-readable audit](../results/family_runtime_v1/m2_581919_default_payload_audit_v1.json)
records exact paths and hashes. Relevant unchanged sources are the
[child Dockerfile](../submissions/evalai_m2_movement_t4_empty_epochpick_v1/Dockerfile),
[base entrypoint](../submissions/evalai_m2_movement_t4_empty_v1/decode.py), and
[historical registration record](../submissions/evalai_m2_movement_t4_empty_epochpick_v1/artifacts/evalai_push_state.json).

The new local timing comparisons report both SPINT payloads separately and
load the image's original SPINT module. They do not silently replace the
declared model with the default-loaded model in a quality table. Fixing or
resubmitting the historical package would be a separate action requiring
explicit scope; this audit performs neither.
