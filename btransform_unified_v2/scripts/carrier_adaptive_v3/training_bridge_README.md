# carrier_adaptive_v3 training bridge

`training_bridge.py` is deliberately source-only. It starts no process; `--out` is an immutable JSON plan file below `results/carrier_adaptive_v3`, while required `--run-root` separately owns planned runs and prepared directories below that same root. It invokes frozen v2 trainers only in a future fresh subprocess, never by in-process import or global mutation.

The safe reusable mode uses exact v2 scheduler horizons: M1 24, H1 32. Existing v2 source curves can be inspected and any newly declared source-only prefix stopping rule can be replayed when the scheduler trajectory is identical. They cannot be reused after a changed max horizon, scheduler, optimizer, carrier/model injection, or any other trajectory-changing recipe.

A new max epoch, LR horizon, or early-stop rule requires copying these files into `carrier_adaptive_v3`, with new schemas and result roots: `protocol.py`, `m1_train.py`, `m1_score.py`, `h1_train.py`, `h1_prepare.py`, `source_seal.py`, `paired_audit.py`, and `source_run_queue.py`. Reuse `m1_data.py` only through a copied v3 wrapper because its pack schema/environment variables are v2-specific. A v3 model wrapper is required only if carrier semantics or injection actually changes; do not alter v2 shared models.

The v3 receipt must record max epochs, full scheduler configuration, every EMA checkpoint, early-stop state, source-only selected epoch, source/pack hashes, and zero target optimizer/selection use. Target preparation may begin only after a v3 source gate.
