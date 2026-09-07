# M2 RIFT R50 mechanism protocol

This is the new, independently trained representative M2 source-seven/ext4
mechanism surface.  It binds the frozen source manifest, 24 epochs, 3,165
updates per epoch (75,960 total), seed 42 default, EMA, and ext4 epoch scan.
Only `source_train` produces gradients; ext4 is the declared development
selection surface and no official test is read.

The four information arms are `B_ACTIVITY_ONLY`, `C_CARRIER_ONLY`,
`D_JOINT`, and `D_SHUFFLE`.  C has no activity encoder, fixes E0 to zero, and
sends only the carrier through the direct path.  D-shuffle applies the fixed
seed-101 unit permutation separately for each session before both the encoder
carrier side features and the direct carrier fusion.  Its checkpoint retains
the permutation buffers, making the correspondence auditable.

Two independently trained architecture controls use the same D information
arm and training protocol: `--aggregation mean` replaces all slots with a
masked mean of 256-dimensional unit tokens followed by an explicit 256-to-256
projection; `--temporal nonattention` replaces temporal attention with four
pre-LN depthwise causal convolution blocks (windows 13, 12, 12, 12, dilation
one), pointwise mixing, residuals, and 512-wide FFNs.  Its raw receptive field
is 50: `1 + 12 + 11 + 11 + 11 + 4`.

Example smoke invocation:

```bash
python scripts/rift_v1/m2_mechanism_train.py --dest results/rift_v1/m2_mechanism_smoke \
  --arm D_JOINT --aggregation slots --temporal attention --device cpu --max-updates-smoke 1
```

Formal invocations require exactly 24 epochs.  `run_meta.json`, checkpoints,
and receipts bind arm, architecture choice, source/cache hashes, and the
source-only/external-selection state machine.
