# M2 RIFT-R50 static ablation packer

This directory creates one fresh, standalone package for either
`ACTIVITY_ONLY` or `NONE`. It is the static control for the frozen original
M2 RIFT R50 concat decoder, seed 42, source-seven 24 × 3165 training recipe;
it is not the smaller transformer and it does not contain a live joint encoder.

```bash
python3 pack_and_verify.py --stage build --arm ACTIVITY_ONLY \
  --bank-cache-root BANK_CACHE --run-dir FORMAL_RUN \
  --selection-dir EXT6_SELECTION --dest /fresh/m2_activity_only

python3 pack_and_verify.py --stage host --arm ACTIVITY_ONLY \
  --bank-cache-root BANK_CACHE --run-dir FORMAL_RUN \
  --selection-dir EXT6_SELECTION --dest /fresh/m2_activity_only \
  --public-cache-root FROZEN_PUBLIC_CACHE
```

`build` requires the new file-per-array cache ABI: root `manifest.json`,
`source_train`, `source_minival`, `ext4`, `session`, and `ext6_query`; each
material session provides `mapping.json`, `e0_u.pt`, `T.npy`, `X_store.npy`,
`target_store.npy`, and `eligible_starts.npy`. It rejects serialized
`bank_payload.pkl` inputs. Exactly 13 canonical M2 tags are sealed: seven
held-in source sessions and six held-out query sessions. Every static row is
finite `E0[96,50]`, `T[96,4]`, and bool mask `[96]`, with byte SHA checks.
`ACTIVITY_ONLY` requires zero T (the frozen P0/EMPTY zero-side condition);
`NONE` requires zero E0 and T.

The packer validates a completed 24-epoch / 75,960-step train receipt and a
completed EXT6 receipt with all 24 checkpoint SHA/report entries. It
rederives the earliest maximum `equal_session_mean` epoch and checks the exact
selected EMA state path and SHA. The build receipt deliberately says
`BUILT_NOT_HOST_VERIFIED`.

`host` uses real public cached X only. It compares the packed cached-KV runtime
with a separate R50 state-temporal forward path for B1, native B7, and a mixed
source/query roster, with at least 80 real bins and a `1e-5` gate. For query
window caches it reconstructs chronological traces only from exact overlap and
strips the declared 49-bin mapping prefix. It also tests a partial batch
followed by full-row resumption against a full consumed-history reference, and
a genuine all-zero valid input. Only a passing run writes
`artifacts/host_verify.json`; it also stores the untouched first 40 raw bins
and their full expected output trace for container smoke.

The Dockerfile is CPU-only and functional, but neither stage builds or pushes
an image, registers, or submits anything.
