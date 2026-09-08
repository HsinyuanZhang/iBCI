# M2 六 arm mechanism controls：seed42 EXT4 M33

本文件记录 completed `m2_mechanism_controls_summary_v1` 的只读、receipt-bound 描述性汇总。输入为 `results/diagnostics_v1/m2_mechanism_controls_summary_v1.json`，SHA-256 `b55cb2d71b4b5d704bfce0a3c8a6ae1fe4077b27cbdb3762399ba463d0c71de5`；生成它的 frozen helper 是 `scripts/diagnostics_v1/summarize_m2_mechanism_controls.py`，SHA-256 `7b3c4c84f7013bc1bbac19553ed3c6cb8d87c5ed1c14ecb35debeb6aeba66bf8`。

该 helper 对 B、C、D、shuffle、mean、nonattn 六个 formal arm 重验：每 arm 的 train/score receipt、24 个 checkpoint、live source hashes、cache hashes、完整 EXT4 EMA map 和 selection binding。summary 内的 `arms.*.receipt_sha256`、`source_hashes`、`cache_hashes` 与 `checkpoint_sha256_by_epoch` 是全部逐文件 provenance binding；共同 frozen manifest digest 为 `a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a`。

三个 mechanism-arm validation audits 也已存在：shuffle SHA-256 `07ea8a52aabcdb8c5ee00390a64013c9407b1f5cc05d6c3fe1d1daed64b4aa62`、mean SHA-256 `63f74346fbea0e91da66de6fb7d785094831ccc506bf9e1d9b628c908ce28b00`、nonattn SHA-256 `4d8ccde62686df456e9216d1c1c4b871c939907d2a26845e0c071f85cfd2ab27`。这不替代 summary 对全部六 arm 和 144 个 checkpoint 的实际重验。

## 六 arm selected 与 fixed-e24 rows

所有数值来自 fixed EXT4 M33 的四个 visible session、2,069 windows。每个 arm 分别在自己的 24-epoch EMA curve 上选择 earliest best equal-session mean；fixed-e24 是同一 arm 的 epoch 24 row。

| Arm | selected EMA epoch | selected equal-session R² | fixed e24 equal-session R² | selected − e24 |
| --- | ---: | ---: | ---: | ---: |
| B activity-only | 7 | 0.2668254644 | 0.2005157086 | +0.0663097558 |
| C carrier-only calibration | 19 | 0.2757857807 | 0.2286786849 | +0.0471070958 |
| D joint | 17 | 0.3935032533 | 0.3610221970 | +0.0324810563 |
| shuffle | 10 | 0.2783682297 | 0.1868703207 | +0.0914979090 |
| mean | 12 | 0.2932165628 | 0.2433790287 | +0.0498375341 |
| nonattn | 8 | 0.3939113798 | 0.3707244012 | +0.0231869786 |

## D-minus-control descriptives

| Contrast | independently selected EMA | fixed epoch 24 |
| --- | ---: | ---: |
| D − B | +0.1266777889 | +0.1605064884 |
| D − C | +0.1177174726 | +0.1323435120 |
| D − shuffle | +0.1151350235 | +0.1741518762 |
| D − mean | +0.1002866905 | +0.1176431683 |
| D − nonattn | -0.0004081265 | -0.0097022043 |

D-minus-mean 为正，但 D-minus-nonattn 在两个预先显示的 epoch policy 下都不为正。因此这个单 seed control set 不支持 temporal-attention 收益的结论；这不是“等效测试通过”。D-minus-shuffle 的 equal-session mean 为正，但 selected rows 的四个 session 中有两个负 delta：`ses-2020-10-30-Run2` 与 `ses-2020-11-18-Run1`。这些负值保留在 summary，未删除或重加权。

## 解释边界

这是 seed42、visible EXT4 M33 development evidence。四个 session 是同一 model seed 内的重复测量，不能视为四个独立 seed，也不产生 CI、p 值、显著性或 multiseed architecture-effect claim。C 的 calibration information 是 carrier-only，但 query activity 仍然使用；它不是“无 activity”条件。

本文件不涉及 EXT6 candidate selection、official test、submission 或最终排名。基于该 completed summary 的 v2 图已经独立生成并视检：可点击查看 [PNG](../results/diagnostics_v1/m2_mechanism_controls_plots_v2/m2_mechanism_controls.png) 与 [PDF](../results/diagnostics_v1/m2_mechanism_controls_plots_v2/m2_mechanism_controls.pdf)。其 figure-data SHA-256 为 `bb4d002539f78d51134f147e8b5c4a0d5ed493e26a8315e466fd2431747779e4`，validation audit SHA-256 为 `951ea884981384d98cc9f052afcbe0de0ae345e02b99b00c5c1925afbfac9a23`。validation 已重验六个 arm 的 actual score receipts、selected/e24 rows、全部 contrast 的独立重算，以及 artifact/source hashes；v1/v2 的所有 numerical arrays 完全一致。 
