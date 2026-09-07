"""Emit the static, machine-readable H1 alignment constructibility receipt.

This audit does not inspect an NWB payload or produce decoder metrics.  It
only hashes and cites the checked data adapters which determine whether a
no-target-label route can be constructed without a new neural-only adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from alignment_constructibility import alignment_cost_contract, h1_alignment_arm_schema


ROOT = Path(__file__).resolve().parents[2]
H1_PILOT = ROOT / "SPINT-main/src/data/h1_m4_eb_pilot.py"
H1_TARGET = ROOT / "SPINT-main/src/data/h1_carrierid_date_lodo_target.py"
H1_SOURCE = ROOT / "SPINT-main/src/data/h1_carrierid_date_lodo_source.py"
SUA_MODULE = ROOT / "sua_exploration/mc_maze/datamodule.py"
RT_MODULE = ROOT / "streaming_calibration_exp/src/data/rt_datamodule.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report() -> dict[str, object]:
    return {
        "receipt_schema": "label_free_alignment_constructibility_v2",
        "supersedes": "H1_LABEL_FREE_ALIGNMENT_CONSTRUCTIBILITY_AUDIT_v1.json",
        "status": "CONDITIONAL_H1_ONLY_CONSTRUCTIBLE__NO_GPU_AUTHORIZATION__NEURAL_ONLY_TARGET_ADAPTER_REQUIRED",
        "scope": {
            "target_decoder_r2_computed": False,
            "target_nwb_payload_read": False,
            "target_behavior_or_velocity_read": False,
            "gpu_started": False,
        },
        "generic_cca_or_procrustes": {
            "status": "FAIL_CLOSED_NO_PAIRED_OBJECT_OR_ANCHOR",
            "reason": "Unpaired source and target neural supports have no common sample/condition correspondence. CCA needs paired observations; Procrustes needs a matched row basis. Behavioral phase, velocity, target labels, or timestamps cannot be introduced without changing the no-label contract.",
        },
        "conditional_h1_ordered_channel_solution": {
            "status": "CONSTRUCTIBLE_IF_FIXED_ORDERED_176_CHANNEL_INDEX_IS_DECLARED_A_VALID_ANCHOR",
            "not_general": "This is H1-only. It cannot be claimed for variable-N SUA sorted units or RT channels, where index order is not a stable cross-session correspondence.",
            "anchor": "H1 fixed ordered channel index, excluding declared all-zero channel 66; this is not a biological identity claim.",
            "source_only_fit": "Fit source mean/scale and [175,4] reference loading basis inside each source-LODO fold only. Reference PCA is block-weighted after per-recording centering: every valid M=4 source block has equal weight, so a recording with more blocks contributes more rows.",
            "target_unlabeled_solve": "From chronological target support neural rates only, compute target PCA axes; solve 4x4 orthogonal Procrustes against source reference; restore dead channel 66 as literal zero; apply fixed 1/sqrt(175) orthonormal-loading scale (not a source-fitted statistic).",
            "gauge_and_fail_closed": [
                "deterministic SVD sign convention (largest absolute channel loading; lower index breaks ties)",
                "Procrustes polar factor fixes order/sign/rotation only relative to frozen source reference",
                "require enough support blocks, target rank >= 4, finite values, and numerical full rank of the 4x4 cross-Gram (sigma_min > 1e-8)",
                "dead channel 66 is never normalized or inferred; its output row is zero",
            ],
            "conditioning_limit": "sigma_min > 1e-8 is a numerical definability check only. It is not a reliability gate; no source-only conditioning threshold or calibration-stability evidence is closed.",
            "cost_contract": alignment_cost_contract(),
            "literature_scope": "A-FULL is an H1-only ordered-channel carrier inspired by aligned-factor-analysis/Procrustes geometry. It is not an implementation or reproduction of a published competing adaptation method. In particular, it does not implement NoMAD-style target-time backpropagation or read-in/readout updates, and it does not by itself close the paper limitation that competing adaptation methods have not been implemented.",
        },
        "existing_adapter_blocker": {
            "status": "FAIL_CLOSED_FOR_CURRENT_H1_TARGET_PATH",
            "reason": "The existing target dataset loads velocity through load_record and calls label-supervised fit_frozen_carrier. A compliance run requires a new target neural-only support adapter with a poison behavior/velocity field test before any decoder experiment.",
            "static_evidence": [
                "h1_m4_eb_pilot.py: load_record receives neural and velocity; fit_frozen_carrier consumes trial.velocity",
                "h1_carrierid_date_lodo_target.py: target support is first four chronological trials, query begins at fifth trial, and current carrier call is fit_frozen_carrier",
            ],
        },
        "required_future_controls": {
            "frozen_arm_schema": h1_alignment_arm_schema(),
            "training_requirement": "A-FULL, A-Z4, and A-RS must be separately trained, with identical source-LODO dates, seeds, checkpoint rule, source mean/scale, carrier width, and target support/query boundary. A-FULL@RS is same-checkpoint diagnostic only. H-C is a labeled operational reference only if all fields match; H-LS/H-RS remain contextual unless exact protocol parity is established.",
            "source_lodo": "Every outer target date must fit its source mean/scale and reference basis from source recordings only; the orthonormal carrier scale is a fixed algebraic constant, and no target recording contributes to fitting.",
            "target_boundary": "Target support is the first four chronological trials; query starts at the fifth chronological trial. The new adapter must expose only neural support rates to the alignment solve.",
        },
        "minimal_viable_alternative_if_ordered_index_is_rejected": {
            "status": "NO_GENERIC_LABEL_FREE_ALIGNMENT_BASELINE_IS_IDENTIFIABLE",
            "alternatives": [
                "obtain a real cross-session channel/unit correspondence or stable metadata anchor, then preregister it",
                "use matched behavioral/phase anchors and explicitly relabel the method weakly supervised rather than label-free",
                "keep the current source-trained compact-consumer claim and do not create a fictitious generic CCA/Procrustes baseline",
            ],
        },
        "static_source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (H1_PILOT, H1_TARGET, H1_SOURCE, SUA_MODULE, RT_MODULE)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    args = parser.parse_args()
    rendered = json.dumps(report(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
