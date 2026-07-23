"""Gate 1: B0 batch vs B1 exact trial streaming equivalence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import BatchReferenceEncoder, TrialStreamingEncoder, build_encoder
from src.models.falcon_module import FalconLitModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=Path("../SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/streaming_calibration/gate1_b1_equivalence.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = FalconLitModule.load_from_checkpoint(str(args.teacher_ckpt.resolve()), weights_only=False)
    net: SpintModel = teacher.net.to(device)
    net.eval()

    batch_size, num_trials, trial_len, num_neurons = 2, 33, 100, 96
    window = 50
    neural = torch.randn(batch_size, window, num_neurons, device=device)
    calib = torch.randn(batch_size, num_trials, trial_len, num_neurons, device=device)

    with torch.no_grad():
        y_batch = net(neural, calib_trialized_neural_features=calib)

        b0 = BatchReferenceEncoder(net.fc_id_in, net.fc_id_out, window)
        b1 = TrialStreamingEncoder(net.fc_id_in, net.fc_id_out, window)
        e_batch = b0.forward_batch(calib)
        e_stream = b1.forward_batch(calib)

        id_batch = e_batch
        id_stream = e_stream
        src = neural.permute(0, 2, 1)
        src_dec = net.fc_in(src + id_stream)
        rep = net.fc_in(net.rep).to(src_dec)
        out, _ = net.transformer(rep.repeat(batch_size, 1, 1), src_dec)
        y_manual = net.fc_out(out).permute(0, 2, 1)

    e_diff = (e_batch - e_stream).abs().max().item()
    y_diff = (y_batch - y_manual).abs().max().item()

    # trial order invariance
    perm = torch.randperm(num_trials)
    e_perm = b1.forward_batch(calib[:, perm])
    order_diff = (e_stream - e_perm).abs().max().item()

    # M=1 and M=33
    e_m1 = b1.forward_batch(calib[:, :1])
    e_m33 = b1.forward_batch(calib)
    m1_shape_ok = tuple(e_m1.shape) == (batch_size, num_neurons, window)

    # M=0 should fail
    m0_failed = False
    try:
        state = b1.reset_stream(batch_size, num_neurons, calib.device, calib.dtype)
        b1.finalize_identity(state)
    except ValueError:
        m0_failed = True

    passed = e_diff < 1e-6 and y_diff < 1e-6 and order_diff < 1e-6 and m0_failed and m1_shape_ok
    payload = {
        "gate": "Gate1_B1_exact_streaming",
        "passed": passed,
        "max_abs_E_diff": e_diff,
        "max_abs_y_diff": y_diff,
        "trial_order_max_abs_E_diff": order_diff,
        "m0_error_raised": m0_failed,
        "m1_shape_ok": m1_shape_ok,
        "threshold": 1e-6,
        "teacher_checkpoint": str(args.teacher_ckpt.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
