# Work Order: CDM-D Activity-Only Quick V2 Environment Successor

Date: 2026-08-28

Status: authorized non-governing matched engineering successor

## Predecessor

V1 failed honestly before data, checkpoint, CUDA, or forward because the launch
environment omitted the held-data roots required by the inherited parser. V2
must descriptor-validate the immutable V1 failed graph before reserving a new
root:

- root `tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v1`;
- attempt SHA `be00e6346383d90808d4772d7a46589a07c3765fb0317113e6dbb201581e48b7`;
- failure SHA `42210c7eba20af2123e5d586338efe5f40c6459148efdf3e12065d510028cb25`;
- failure stage `materialize_inputs`;
- input authority null;
- no terminal or result.

No V1 file or result may be modified, removed, or retried.

## Environment repair

Before output reservation, require the actual process environment to equal:

```text
SUBC_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
SUBM_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<selected GPU index>
```

Recheck these strings and the held V1 graph immediately before terminal
publication. The path strings are authority inputs; the inherited held reader
retains responsibility for no-follow descriptor validation when data are
opened after attempt publication.

## Unchanged science

V2 imports V1's exact ActivityOnlyMemory, ActivityOnlyRuntime, V8 input replay,
same-input comparison, M10/M4 order, within/external order, metric, summary,
and decision thresholds. It adds no carrier proposal, complementary-group
forward, target gradient, checkpoint change, M30 cell, or new performance
choice.

The fresh result root is:

`tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v2`

