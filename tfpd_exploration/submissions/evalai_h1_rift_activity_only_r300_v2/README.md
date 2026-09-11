# H1 RIFT R300 ablation package template

Builds one fresh local payload from a completed 32-epoch formal H1 ablation run and its sealed 27-tag bank directory. It is prepared for later host/runtime verification and Docker work; it does not claim host, container, registry, or EvalAI success and performs no external action.

```bash
python3 pack_and_verify.py --arm ACTIVITY_ONLY --banks <sealed-bank-dir> --run-dir <formal-run-dir> --dest <fresh-package-dir>
```

## Local package ready, not submitted

Recovery v2 identity:

- image: `h1-rift-activity-only-r300-e15:v1`
- image ID: `sha256:475bc3e036fc9634e3d4caa6a4212d1e2378b9a49b142c1ecd3e81b4987e04a6`
- payload SHA-256: `6becdb88798a088d3fa4136eac1c5ce53e8660977e54c3f70d3718f377ac09d6`
- selected epoch: 15; local HO-M3 grouped-seven mean `0.2660451704314089` (not official)

Read-only preflight (no push/register):

```bash
/usr/bin/python3 tfpd_exploration/submissions/evalai_h1_rift_activity_only_r300_v2/submit.py
```

Execute only after explicit user authorization, with both immutable IDs:

```bash
/usr/bin/python3 tfpd_exploration/submissions/evalai_h1_rift_activity_only_r300_v2/submit.py \
  --execute \
  --confirm-image-id sha256:475bc3e036fc9634e3d4caa6a4212d1e2378b9a49b142c1ecd3e81b4987e04a6 \
  --confirm-payload-sha256 6becdb88798a088d3fa4136eac1c5ce53e8660977e54c3f70d3718f377ac09d6
```
