#!/usr/bin/env python3
"""Pick the strongest MOVE-T4/EMPTY checkpoint on the visible M2 held-out surface.

The selection grid is the already-used paired seed set (42, 43, 44) crossed
with epochs 1..12.  Every cell is a deterministic prefix replay from the same
canonical zero-initialized adapter.  Only the six locally visible M2 external
sessions select the checkpoint; EvalAI is not contacted by this program.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import time

import numpy as np
import torch

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_hold_film_probe_v1.physical import (
    _atomic_bytes,
    _atomic_json,
    _state_sha,
    install_film,
    score_arm,
    train_film,
)
from tfpd_exploration.src.m2_movement_t4_ablation_v1.physical import (
    _fit_source_normalizer,
    _override,
)
from tfpd_exploration.src.m2_movement_t4_film_ablation_v1 import plan
from tfpd_exploration.src.m2_movement_t4_film_ablation_v1.physical import _head_state


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def execute() -> dict:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES=0 required")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER=PCI_BUS_ID required")
    require(not RESULT.exists(), f"immutable result root already exists: {RESULT}")
    RESULT.mkdir(parents=True)
    _atomic_json(
        RESULT / "attempt.json",
        {
            "schema": "m2_movement_t4_empty_epoch_pick_v1_attempt",
            "status": "STARTED_BEFORE_TORCH_CUDA_OR_DATA",
            "selection_surface": "six locally visible external M2 sessions",
            "seeds": list(plan.SEEDS),
            "epochs_one_based": list(range(1, plan.EPOCHS + 1)),
            "evalai_opened": False,
            "hidden_evalai_score_used": False,
        },
    )
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    student = model.student.to(device)
    base_encoder = student.id_encoder
    decoder_before = _state_sha(student.decoder.state_dict())
    base_before = _state_sha(base_encoder.state_dict())
    base_train = data_module.train_dataset
    base_external = data_module.val_heldout_dataset
    mean, std, normalizer = _fit_source_normalizer(base_train)
    train, train_hashes = _override(base_train, mean, std)
    external, external_hashes = _override(base_external, mean, std)
    datasets = {"within_post30": train, "external_official_query": external}

    canonical_payload = torch.load(
        ROOT / probe_plan.RESULT_ROOT_RELATIVE / "film_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    canonical = {
        key: value.detach().cpu().clone()
        for key, value in canonical_payload["p0"].items()
    }
    zero = np.asarray(plan.EMPTY_MASK, dtype=np.float32)
    curve: list[dict] = []
    state_by_cell: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    for seed in plan.SEEDS:
        for epochs in range(1, plan.EPOCHS + 1):
            student.id_encoder = base_encoder
            install_film(student, device, film_input=plan.FILM_INPUT, seed=int(seed))
            student.id_encoder.load_state_dict(canonical, strict=True)
            student.id_encoder.freeze_base_path()
            training = train_film(
                student=student,
                train_dataset=train,
                device=device,
                shuffle=False,
                contrast_mask=zero,
                horizon=plan.HORIZON,
                t4_mode="native",
                learning_rate=plan.LEARNING_RATE,
                epochs=epochs,
                seed=int(seed),
                windows_per_session=plan.WINDOWS_PER_SESSION,
            )
            score = score_arm(
                student=student,
                datasets=datasets,
                device=device,
                batch_size=1024,
                shuffle=False,
                arm=f"MOVE-EMPTY-s{seed}-e{epochs:02d}",
                contrast_mask=zero,
                horizon=plan.HORIZON,
                t4_mode="native",
            )
            summary = score["summaries"]["external_official_query"]
            session_values = score["session_maps"]["external_official_query"]
            row = {
                "seed": int(seed),
                "epoch_one_based": int(epochs),
                "external_equal_session_mean": float(summary["equal_session_mean"]),
                "external_worst_session": float(min(session_values.values())),
                "external_per_session_r2": session_values,
                "within_equal_session_mean": float(
                    score["summaries"]["within_post30"]["equal_session_mean"]
                ),
                "head_state_sha256": score["film_head_sha256"],
                "training_last_epoch": training["history"][-1],
            }
            curve.append(row)
            state_by_cell[(int(seed), int(epochs))] = _head_state(student)
            print(
                f"epoch-pick seed={seed} epoch={epochs:02d} "
                f"external={row['external_equal_session_mean']:.9f} "
                f"worst={row['external_worst_session']:.9f}",
                flush=True,
            )

    # Deterministic policy: higher visible-held-out mean, then higher worst
    # session, then earlier epoch, then lower seed.
    selected = sorted(
        curve,
        key=lambda row: (
            -row["external_equal_session_mean"],
            -row["external_worst_session"],
            row["epoch_one_based"],
            row["seed"],
        ),
    )[0]
    selected_state = state_by_cell[(selected["seed"], selected["epoch_one_based"])]

    # The epoch-12 replays must reproduce the previously sealed final heads.
    replay: dict[str, dict] = {}
    for seed in plan.SEEDS:
        previous = torch.load(
            ROOT / plan.RESULT_ROOT_RELATIVE / f"seed{seed}/film_heads.pt",
            map_location="cpu",
            weights_only=False,
        )["EMPTY"]
        observed = state_by_cell[(int(seed), plan.EPOCHS)]
        exact = all(torch.equal(previous[key], observed[key]) for key in previous)
        require(exact and set(previous) == set(observed), f"seed {seed} epoch-12 replay drift")
        replay[str(seed)] = {
            "exact": True,
            "previous_head_sha256": _state_sha(previous),
            "replayed_head_sha256": _state_sha(observed),
        }

    buffer = io.BytesIO()
    torch.save(
        {
            "schema": "m2_movement_t4_empty_epoch_pick_v1_head",
            "selection": selected,
            "state_dict": selected_state,
        },
        buffer,
    )
    _atomic_bytes(RESULT / "selected_head.pt", buffer.getvalue())
    result = {
        "schema": "m2_movement_t4_empty_epoch_pick_v1",
        "status": "SELECTED_BEFORE_EVALAI",
        "selection_rule": (
            "max visible external equal-session mean; higher worst-session, "
            "earlier epoch, lower seed tie-breaks"
        ),
        "selection_surface": "six locally visible external M2 sessions",
        "selected": selected,
        "curve": curve,
        "epoch12_replay": replay,
        "selected_head_state_sha256": _state_sha(selected_state),
        "source_normalizer": normalizer,
        "side_sha256": {
            "within_post30": train_hashes,
            "external_official_query": external_hashes,
        },
        "frozen_checkpoint_sha256": metadata["checkpoint_sha256"],
        "frozen_decoder_sha256": decoder_before,
        "frozen_base_encoder_sha256": base_before,
        "elapsed_seconds": float(time.monotonic() - started),
        "evalai_opened": False,
        "hidden_evalai_score_used": False,
    }
    require(_state_sha(student.decoder.state_dict()) == decoder_before, "decoder changed")
    require(_state_sha(base_encoder.state_dict()) == base_before, "base encoder changed")
    _atomic_json(RESULT / "selection.json", result)
    _atomic_json(
        RESULT / "terminal.json",
        {
            "schema": "m2_movement_t4_empty_epoch_pick_v1_terminal",
            "status": "TERMINAL_SELECTED_BEFORE_EVALAI",
            "selected": selected,
            "evalai_opened": False,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            {
                "status": "DRY_RUN",
                "result_root": str(RESULT),
                "grid": [list(plan.SEEDS), [1, plan.EPOCHS]],
            }
        )
        return
    result = execute()
    print({"status": result["status"], "selected": result["selected"]})


if __name__ == "__main__":
    main()
