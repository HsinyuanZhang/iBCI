# M1 trained RIFT CPU runtime

The benchmark accepts only a completed formal M1 proj-add run and its matching
completed score receipt. It loads the receipt-selected EMA checkpoint, verifies
its SHA against the scan, then measures `CpuRiftRuntime(..., temporal_backend="cached")`
on actual continuous M1 source neural bins and frozen M10/B3 banks. B1 and B8
use 300 warmup advances and three rounds of 100 timed decode calls. Cached
logical KV accounting is separate from process RSS and is not added to it.

The receipt binds run metadata, score receipt and selected checkpoint hashes.
It checks 401 true observed advances including a valid zero-bin count, stream
reordering and reset parity against the reference streaming adapter. No ONNX
or random/smoke weights are permitted.
