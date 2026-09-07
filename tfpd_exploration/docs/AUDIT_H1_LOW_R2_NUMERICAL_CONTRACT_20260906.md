# H1 QueryAge split-12: low-R2 numerical-contract audit

Scope: static code and JSON/NPZ metadata audit only, performed while the
authorized 12-epoch paired run is live. No checkpoint was deserialized, no
model forward was run, and no cache/GPU/training/protocol file was changed.

## Result

There is **no static evidence of an accidental factor-of-20 mismatch** between
training and EMA selection, nor of an R2 reduction/aggregation mismatch. The
live epoch-8 values (FLAT `0.2131546`, ROUTE `0.2152028`) are therefore not,
on the evidence available here, explained by a simple prediction/target unit
error. This is not an efficacy conclusion: it cannot rule out optimization,
data, architecture, cache-content, or implementation behavior requiring an
authorized post-completion replay.

## Contract trace

| Stage | Code evidence | Numeric space |
|---|---|---|
| Training collate | `h1_optimized_v4/paired_train.py:25` stacks source `velocity[start+699]` then multiplies it by `20` | target is decoder/raw space, `20 ×` native velocity |
| Training forward/loss | `formal_prefix_train.py:250-257`, specifically the MSE at line 254 | `forward_last(...)` is compared directly to that raw-space target; no division is applied in the loss |
| Update / EMA order | `formal_prefix_train.py:257` calls `optimizer.step()` before `ema.update_after_step(model)` | EMA sees the post-step RAW parameters |
| EMA initialization/update | `ema.py:26-39` clones post-first-step parameters; later uses `shadow = .9995*shadow + .0005*RAW` | FP32 parameter shadow, one update per optimizer step |
| Selection parameter choice | `ema.py:55-69` copies shadow to the model, calls scorer in eval mode, then restores RAW exactly | reported selection is EMA-only; raw model is not silently retained |
| Selection source / endpoints | `formal_prefix_score.py:49-67` uses `query_starts + 699`; `:112-122` forms W700 windows and uses `velocity[ends]` | same endpoint convention as collate (`start + 699`) and native source velocity |
| Selection output scale | `formal_prefix_score.py:116-122` computes `forward_last(x, bank) / 20.0`, then compares to native `velocity[ends]` | decoder/raw prediction converted once to native, matching native target |
| R2 precision | `formal_prefix_score.py:25-36` converts both arrays to FP64 before residual/denominator arithmetic | pooled native FP64 R2 |
| Aggregation | `formal_prefix_score.py:140-149` returns pooled concat R2; complete-only additionally computes equal-session mean and worst session | selection criterion is pooled concat, not mean-of-session R2 |

The exact same forward API is used by training (`formal_prefix_train.py:254`)
and scoring (`formal_prefix_score.py:116`). The difference in only the target
space is explicit and paired: train target `velocity * 20`; selection
prediction `forward_last / 20` versus unscaled `velocity`. Algebraically they
are the same native residual up to a constant scale:

`MSE(raw_prediction, 20*velocity) = 400 * MSE(raw_prediction/20, velocity)`.

Thus this design changes gradient magnitude relative to native-unit MSE, but
does **not** change the optimum or create an R2 unit mismatch.

## EMA and channel/order checks

- `DecoderEMA.score_with_ema` saves every trainable RAW tensor, installs only
  the shadow tensors, and restores RAW in a `finally` block (`ema.py:55-69`).
  The scorer independently snapshots trainable RAW tensors and rejects a
  mutation (`formal_prefix_score.py:77-89, 96-135`); the worker adds a state
  digest guard (`formal_prefix_train.py:233-239`).
- The scorer preserves the ndarray's last dimension; it neither selects nor
  permutes velocity channels (`formal_prefix_score.py:118-122`). R2 sums
  residuals and target variation across the existing seven columns
  (`formal_prefix_score.py:30-33`). Static code therefore shows no channel
  reorder, averaging, or accidental single-channel calculation.
- Session ordering is canonical `sorted(cache["minival"].items())`
  (`formal_prefix_score.py:39-46, 107`), and complete scoring records both
  session IDs and endpoint indices (`:122-149`). This rules out a hidden
  dictionary-order aggregation difference within this scorer.

## Deliberate train/selection differences (not scale bugs)

Training applies deterministic cold prefixing and whole-unit dropout before
the forward (`formal_prefix_train.py:250-254`); selection calls the scorer in
`eval()` with no supplied dropout mask (`formal_prefix_score.py:104-117`).
Selection uses full W700 source windows; training uses a p=.5 left-zero prefix
treatment. This is an intentional augmentation/evaluation gap documented in
the frozen recipe, not evidence that targets or prediction units are misaligned.
It remains a plausible *quality* contributor, but cannot be relabelled a
numerical accounting bug without a completed authorized replay.

## Excluded and not excluded

Excluded by static trace:

- train `×20` with selection target still `×20`;
- selection prediction divided twice or not divided at all;
- RAW rather than EMA governing selection;
- FP32-only R2 or mean-of-session R2 replacing pooled concat R2;
- obvious endpoint `start+698`/`start+700` mismatch;
- explicit channel reordering in the train/selection paths.

Not excluded without later authorized artifacts/replay:

- the actual cache's velocity units/content and every session's channel
  semantic ordering;
- model-internal `forward_last` behavior, trained weights, optimizer dynamics,
  and EMA numerical values;
- source/cache corruption that nevertheless passed the frozen authority
  validator;
- whether the p=.5-prefix/dropout training distribution produces low general
  quality on the frozen full-window selection surface.

No code bug is asserted. If an eventual completed archive shows a scale issue,
the falsifiable chain is: compare its stored FP64 target to cache
`velocity[end]`, then compare stored prediction to an authorized direct
`forward_last(window)/20`; a uniform approximately `20×` residual ratio would
locate the error on the raw/native boundary. That test is intentionally not
performed during this live run.
