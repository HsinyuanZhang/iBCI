# M1 q=3 EMG-AFC4 all-source final path

This path is a preparation interface, not a submission.  It is intentionally
fit-only for local data: the common teacher uses all four held-in calibration
sessions, and Full/B4 students use the same teacher and the same source-frozen
PCA/normalizer.  Every identity is chronological M10 (`[0,10)`).  The three
public held-out calibration NWBs are read only by the offline exporter, where
they contain exactly ten trials; hidden future/evaluation files are never
opened locally.

## CPU preflight (safe to run now)

From `streaming_calibration_exp/`:

```bash
CUDA_VISIBLE_DEVICES= \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  scripts/preflight_m1_all_source_final.py \
  --out ../sua_exploration/results/m1_all_source_final_preflight_v1
```

The receipt contains `preflight.json` and `preflight.sha256`.  It records the
four source file hashes, train batch shapes, the no-validation/no-query
contract, and matching Full/B4 source basis/normalizer hashes.

## GPU commands (do not run as part of preflight)

The commands below are the reviewed launch templates.  Replace `GPU` and
`TEACHER` with the selected device and the immutable path to the all-source
teacher checkpoint.  Full and B4 must receive the same `TEACHER` path and
SHA256.

1. Common teacher, all four held-in sessions, fixed 20 epochs:

```bash
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp
CUDA_VISIBLE_DEVICES=GPU \
  /home/xinyuan/miniconda3/envs/spint/bin/python -u src/train.py \
  experiment=m1_afc4_source_decoder_all_source \
  train=true test=false require_baseline_validation=false seed=42 \
  run_id=m1_afc4_source_decoder_all_source_dev20
```

The callback selects one checkpoint by source `train/loss` only after the
fixed 20-epoch run.  Do not assume that the selected checkpoint is
`epoch_019.ckpt`: read Lightning's actual `best_model_path`, then verify that
checkpoint's internal epoch/global step and hash before using it.  The data
hook writes `all_source_train_manifest.json` and its SHA256 receipt beside the
Hydra run directory.  For example:

```bash
sha256sum /path/to/teacher/checkpoints/best_ckpt/epoch_NNN.ckpt
sha256sum /path/to/teacher/all_source_train_manifest.json
```

2. Final Full and B4 students, both fixed 12 epochs, task-only objective:

```bash
TEACHER=/path/to/teacher/checkpoints/best_ckpt/epoch_019.ckpt
CUDA_VISIBLE_DEVICES=GPU0 \
  /home/xinyuan/miniconda3/envs/spint/bin/python -u src/train.py \
  experiment=m1_afc4_emg_full_all_source_final \
  train=true test=false require_baseline_validation=false seed=42 \
  model.teacher_ckpt_path="$TEACHER" \
  run_id=m1_afc4_emg_full_all_source_final_dev12

CUDA_VISIBLE_DEVICES=GPU1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python -u src/train.py \
  experiment=m1_afc4_emg_b4_all_source_final \
  train=true test=false require_baseline_validation=false seed=42 \
  model.teacher_ckpt_path="$TEACHER" \
  run_id=m1_afc4_emg_b4_all_source_final_dev12
```

No `test=true`, held-in query offset, or held-out flag is legal for these
training commands.  For each student, use the callback's actual
`best_model_path` selected by source `train/loss`; do not hard-code
`epoch_011.ckpt` merely because the budget is 12 epochs.  Record that
checkpoint's internal epoch/global step, SHA256, resolved config, and split
manifest.  (`save_last` may also be present.)

## Offline payload export and image build

After each student checkpoint has been independently hashed, export its
decoder plus seven cached identities on CPU.  This exporter checks direct vs
cached vs decoder-only outputs for exact equality and writes a receipt; it
does not contact EvalAI.

```bash
cd /home/xinyuan/Work_host/SPINT
python sua_exploration/evalai_m1_threeway/export_afc4_payload.py \
  --arm full \
  --checkpoint /path/to/full/checkpoints/best_ckpt/epoch_011.ckpt \
  --resolved-config /path/to/full/resolved_config.yaml \
  --teacher-checkpoint "$TEACHER" \
  --output sua_exploration/evalai_m1_threeway/artifacts/m1_afc4_full_all_source.pkl

python sua_exploration/evalai_m1_threeway/export_afc4_payload.py \
  --arm b4 \
  --checkpoint /path/to/b4/checkpoints/best_ckpt/epoch_011.ckpt \
  --resolved-config /path/to/b4/resolved_config.yaml \
  --teacher-checkpoint "$TEACHER" \
  --output sua_exploration/evalai_m1_threeway/artifacts/m1_afc4_b4_all_source.pkl
```

Build from `sua_exploration/evalai_m1_threeway/` only after the payload
receipts are reviewed:

```bash
docker build -f Dockerfile.afc4.cached \
  --build-arg PAYLOAD_FILE=m1_afc4_full_all_source.pkl \
  --build-arg ARM=full \
  --build-arg PAYLOAD_SHA256=$(sha256sum artifacts/m1_afc4_full_all_source.pkl | cut -d' ' -f1) \
  --build-arg CHECKPOINT_SHA256=$(sha256sum /path/to/full/checkpoints/best_ckpt/epoch_011.ckpt | cut -d' ' -f1) \
  --build-arg TEACHER_SHA256=$(sha256sum "$TEACHER" | cut -d' ' -f1) \
  -t spint-m1-afc4-full-all-source:dev12 .
```

The image copies `afc4_cached_identity_decoder.py` and
`decode_afc4_cached.py`.  Its default command invokes
`FalconEvaluator(..., phase="test")` with the official M1 batch size 4; the
runtime receives only neural observations and hashed dataset tags, resets its
cached identity, and performs no online calibration/backpropagation.

`submit_evalai.py` remains read-only unless the caller explicitly supplies its
`--execute` switch.  The helper now exposes two unregistered candidate entries
(`full` and `b4`) and a no-side-effect `--plan` mode.  Before an image exists,
the CPU-only plan is:

```bash
cd /home/xinyuan/Work_host/SPINT
python sua_exploration/evalai_m1_threeway/submit_evalai.py \
  --arm full --plan
python sua_exploration/evalai_m1_threeway/submit_evalai.py \
  --arm b4 --plan
```

After both payloads/images and their receipts are reviewed, create
`sua_exploration/evalai_m1_threeway/artifacts/evalai_m1_afc4_candidates.json`
with this exact schema (replace every placeholder with the independently
recorded value; do not reuse a Full value for B4):

```json
{
  "schema_version": "evalai_m1_afc4_submission_candidates_v1",
  "challenge_id": 2319,
  "phase_id": 4599,
  "team_id": 41975,
  "candidates": {
    "full": {
      "arm": "full",
      "image_tag": "spint-m1-afc4-full-all-source:dev12",
      "image_id": "sha256:<64 lowercase hex>",
      "payload_sha256": "<64 lowercase hex>",
      "checkpoint_sha256": "<64 lowercase hex>",
      "teacher_sha256": "<64 lowercase hex>",
      "method_name": "M1 q3 EMG-AFC4 Full all-source M10",
      "method_description": "<the reviewed Full description>"
    },
    "b4": {
      "arm": "b4",
      "image_tag": "spint-m1-afc4-b4-all-source:dev12",
      "image_id": "sha256:<64 lowercase hex>",
      "payload_sha256": "<64 lowercase hex>",
      "checkpoint_sha256": "<64 lowercase hex>",
      "teacher_sha256": "<64 lowercase hex>",
      "method_name": "M1 q3 EMG-AFC4 B4 all-source M10",
      "method_description": "<the reviewed B4 description>"
    }
  }
}
```

The helper validates challenge/phase/team identity, independent Full/B4 image
IDs/tags/payload hashes, and Docker labels for payload/checkpoint/teacher
hashes before any network call.  Run ordinary read-only preflight separately
for each arm; it may query only phase/challenge/quota metadata and the user's
submission list:

```bash
python sua_exploration/evalai_m1_threeway/submit_evalai.py \
  --arm full \
  --manifest sua_exploration/evalai_m1_threeway/artifacts/evalai_m1_afc4_candidates.json
python sua_exploration/evalai_m1_threeway/submit_evalai.py \
  --arm b4 \
  --manifest sua_exploration/evalai_m1_threeway/artifacts/evalai_m1_afc4_candidates.json
```

This task stops before those image/manifest steps and never invokes
`--execute`; no hidden result files are inspected.
