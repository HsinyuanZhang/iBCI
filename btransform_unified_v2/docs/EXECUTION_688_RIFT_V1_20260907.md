# DANDI 000688 CO RIFT V1 execution contract

This route freezes `sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` (SHA-256 `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`): 27 local source sessions and six local validation sessions. It does not construct or open the formal-test split. Any future comparison with a published SPINT table must state the split and scoring protocol separately.

Each session uses a 50-bin (20 ms) input window. Activity identity is the frozen seed-42 B3S SPINT encoder's complete identity provider: `pre_pool` mean over M30 calibration trials, then `post_pool([mean; directional-M10])`, giving `E0 [N,50]`. The carrier is the existing sparse-event directional M10 profile `[a_R,c_R,m_R,delta_b]`, where the first three entries are R700 T4 columns and the fourth is R700 fourth column minus H300 fourth column. The source route's existing normalizer, winsorizer and admitted reliability mask are retained.

Sessions have different unit counts. The runner pads every session to the source/validation global `Nmax`; padded neural, E0 and carrier rows are zero and the corresponding `unit_mask` entries are false. The RIFT frontend receives that mask, so padded rows cannot be attended. Real zero spike bins retain a true time mask.

The formal budget is 12 epochs with seed 42. Epochs 8--11 in zero-based numbering are weight-averaged. Validation is scored only after that fixed average; it cannot select an epoch. `preflight` performs an actual CPU gradient, real-profile construction, and full-window/streaming parity on real local data. Use CPU limits `taskset -c 8,9 nice -n 10`, `OMP_NUM_THREADS=2`, and `MKL_NUM_THREADS=2`.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/sua_exploration:$PWD/streaming_calibration_exp:$PWD/btransform_unified_v2/src:$PWD/btransform_unified_v1/src" \
  taskset -c 8,9 nice -n 10 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/dandi688_train.py --dest btransform_unified_v2/results/rift_v1/dandi688_rift_cpu_preflight_v1 --stage preflight --device cpu --cpu-threads 2
```

A new formal training destination must be empty and always runs exactly 12 epochs. A resume checkpoint must reside in the same destination and is accepted only after its schema, formal status, source hashes, target-cache hashes, configuration digest, model state, optimizer state, RNG state, epoch and global step all match. Smoke, malformed/wrong-epoch checkpoints, and already-completed destinations are rejected.

`score` first validates the full 12-epoch `train_receipt.json` and then loads the matching formal `average_e8_e11.pt`. It writes prediction, target, and query-start arrays for every validation session; each array hash and the finite equal-session mean MSE are recorded in `score_receipt.json`.

The default `--prepared-cache btransform_unified_v2/results/rift_v1/dandi688_prepared_cache_v1` is created once from the 27 train and 6 validation sessions. Its atomic `prepared_contract.json` binds the strict manifest, data-materialization source hashes, and every raw/bank/query/target array hash. It is capped below 2 GiB and contains no formal-test data. Subsequent preflight, train, resume, and score runs load and re-verify this immutable bundle; the runner source hash remains separately bound in each training contract.
