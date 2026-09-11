# H1 RIFT R300 ablation package template

Builds one fresh local payload from a completed 32-epoch formal H1 ablation run and its sealed 27-tag bank directory. It is prepared for later host/runtime verification and Docker work; it does not claim host, container, registry, or EvalAI success and performs no external action.

```bash
python3 pack_and_verify.py --arm ACTIVITY_ONLY --banks <sealed-bank-dir> --run-dir <formal-run-dir> --dest <fresh-package-dir>
```
