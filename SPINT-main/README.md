# SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![FALCON Benchmark](https://img.shields.io/badge/FALCON-Benchmark-orange)](https://eval.ai/web/challenges/challenge-page/2319/evaluation)

Official codebase for the paper:

> **SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding**<br>
> Trung Le, Hao Fang, Jingyuan Li, Tung Nguyen, Lu Mi, Amy Orsborn, Uygar Sumbul, Eli Shlizerman<br>
> *NeurIPS 2025*

## Overview

Intracortical brain-computer interfaces (iBCIs) translate neural population activity into motor commands, yet their long-term utility is constrained by recording nonstationarity: shifts in the composition and tuning of recorded units across sessions progressively degrade decoders trained on prior data. We introduce SPINT, a Spatial Permutation-Invariant Neural Transformer whose context-dependent representations are invariant to the size and ordering of recorded units, supporting few-shot adaptation to unseen sessions without parameter updates. SPINT achieves state-of-the-art cross-session decoding on the [FALCON benchmark](https://eval.ai/web/challenges/challenge-page/2319/evaluation) across three intracortical motor tasks while being gradient-free and using minimal unlabeled calibration trials.

## Installation

**Prerequisites:** [Mamba](https://github.com/conda-forge/miniforge) (or Conda) and [Docker](https://docs.docker.com/engine/install/).

```bash
bash setup.sh
mamba activate spint
```

## Data Setup

SPINT was evaluated using the [FALCON benchmark](https://eval.ai/web/challenges/challenge-page/2319/evaluation) datasets, hosted on [DANDI](https://dandiarchive.org/):

| Task | DANDI ID |
|------|----------|
| M1   | [000941](https://dandiarchive.org/dandiset/000941) |
| M2   | [000953](https://dandiarchive.org/dandiset/000953) |
| H1   | [000954](https://dandiarchive.org/dandiset/000954) |

Download the dandisets into `data/` at the repo root, i.e., `<repo>/data/000941/`, `<repo>/data/000953/`, `<repo>/data/000954/`.

## Training

To train with default hyperparameters (replace `<task>` with `m1`, `m2`, or `h1`):

```bash
python src/train.py data=falcon_<task> model=falcon_<task>
```

If needed, override any hyperparameter from the command line, e.g.,
```bash
python src/train.py data=falcon_m1 model=falcon_m1 trainer=gpu model.optimizer.lr=1e-4 trainer.max_epochs=50 <...>
```
(see `configs/` for tunable hyperparameters)

## Evaluation

### Local Evaluation

1. Package a trained model as a decoder (replace <NNN> with desired checkpoint epoch):

```bash
python third_party/falcon_challenge/spint_decoder.py \
  --run_dir logs/train/runs/<run_id> \
  --checkpoint epoch_<NNN>.ckpt
# saves to local_data/spint_<task>.pkl
```

`--checkpoint` accepts a bare filename (searched under `checkpoints/best_ckpt/` then `checkpoints/periodic_ckpt/`), a path relative to the run dir, or an absolute path.

2. Run local evaluation (recommended values for `--batch-size`: M1=4, M2=7, H1=8):

```bash
python third_party/falcon_challenge/spint_sample.py \
  --evaluation local \
  --model-path local_data/spint_<task>.pkl \
  --split <task> \
  --phase minival \
  --batch-size <batch_size>
```

### EvalAI Submission

Build the Docker image:

```bash
docker build --build-arg TASK=<task> --build-arg BATCH_SIZE=<batch_size> \
             -t spint_<task>:latest -f third_party/falcon_challenge/spint_sample.Dockerfile .
```

Submitting to the [FALCON challenge](https://eval.ai/web/challenges/challenge-page/2319/evaluation) needs the `evalai` CLI. Due to a dependency conflict with the SPINT env, install it in its own Python 3.6 environment:

```bash
mamba create -n evalai -c conda-forge python=3.6 -y
mamba activate evalai
pip install evalai
evalai set_token <your-token>   # from https://eval.ai/web/profile
```

Then push:

```bash
evalai push spint_<task>:latest --phase few-shot-test-2319 --private
```

## Citation

```bibtex
@inproceedings{le2025spint,
  title={SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding},
  author={Le, Trung and Fang, Hao and Li, Jingyuan and Nguyen, Tung and Mi, Lu and Orsborn, Amy and S{\"u}mb{\"u}l, Uygar and Shlizerman, Eli},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgments

Evaluation code (`third_party/falcon_challenge/`) is adapted from the [FALCON challenge](https://github.com/snel-repo/falcon-challenge) (MIT License). We thank the FALCON challenge organizers for the benchmark, datasets, and technical support with the submissions.

The repo scaffolding (Hydra configs, Lightning entry points, callbacks, and logging utilities) is built on top of the template at [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) (MIT License).

Distributed-training sampler code (`third_party/catalyst/`) is adopted from [catalyst-team/catalyst](https://github.com/catalyst-team/catalyst) v21.5 (Apache License 2.0).

## License

This project is licensed under the BSD Modified License (BSD 3-Clause). See [LICENSE](LICENSE) for details.

Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab.
