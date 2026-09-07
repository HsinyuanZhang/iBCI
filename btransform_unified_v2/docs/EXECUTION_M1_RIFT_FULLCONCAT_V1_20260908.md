# M1 RIFT full-concat control

`m1_concat_train.py` preserves the formal M1 RIFT train, resume, checkpoint,
EMA, state-machine score, source-contract hashing, padding, and held-out-calib
scan logic from `m1_train.py`. It changes only the frozen decoder interface to
`RiftConcatDecoder("m1")`: each unit token is `[local16 | E0_100 | rSyn3_4]`
with `token_in=120`; `proj_dim` is explicitly `null` and no P16 projection is
claimed.

The full-E0 first affine input is folded from the matched proj-add initialization
as `Wlocal * local + Wlocal * P(E0) + Wcarrier * carrier`. Shared decoder
parameters are byte-equal at seed 42 and the folded affine output differs by
less than `2e-6`, as enforced by the contract test. `RiftConcatDecoder` keeps
the inherited RIFT final norm and readout as the only owner; no unused decoder
head is registered.

The recipe is fixed: four source sessions, 213336 windows, B32, 6665 updates
per epoch, 24 epochs / 159960 updates, R100 D4 windows `(25,25,25,24)`,
AdamW `1e-4`, EMA `.9995`, whole-unit dropout `.1`, frozen B3 Sfix e11 E0,
and M10 rSyn3. Held-out-calib scoring is the same 3881-window visible HO trio
and remains post-training only. No formal run is queued by this diagnostic.
