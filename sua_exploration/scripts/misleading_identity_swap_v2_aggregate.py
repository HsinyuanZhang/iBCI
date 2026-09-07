#!/usr/bin/env python3
"""Pure aggregation scaffold for an already-complete synthetic/real score matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import misleading_identity_swap_v2_core as core


def validate_terminal_against_official(
    terminal: dict, *, cell: str, official: dict, official_sha: str,
    current_bindings: dict,
) -> None:
    core.require(terminal.get("cell") == cell and
                 terminal.get("receipt_kind") == "misleading_identity_swap_v2_cell_terminal",
                 f"{cell}: terminal cell/kind drift")
    core.require(terminal.get("status") == "CELL_TRAINING_COMPLETE__DEVELOPMENT_NOT_OFFICIAL",
                 f"{cell}: terminal status drift")
    core.require(terminal.get("official_preflight_path") == str(core.OFFICIAL_PREFLIGHT_PATH.resolve()) and
                 terminal.get("official_preflight_sha256") == official_sha,
                 f"{cell}: terminal official-preflight binding drift")
    core.require(terminal.get("implementation_bindings") == current_bindings and
                 terminal.get("implementation_bindings_sha256") == official["implementation_bindings_sha256"],
                 f"{cell}: terminal implementation closure drift")
    for key in ("matching_authority_path", "matching_authority_sha256", "source_lineage_path",
                "source_lineage_sha256", "initial_state_path", "initial_state_file_sha256",
                "initial_state_dict_sha256", "cell_output_root"):
        core.require(terminal.get(key) == official.get(key),
                     f"{cell}: terminal {key} differs from official")
    core.require(terminal.get("cell_output_path") == official["cell_output_paths"][cell],
                 f"{cell}: terminal cell output topology drift")
    core.require(terminal.get("seed") == 42 and terminal.get("epochs") == 12 and
                 terminal.get("score_epochs_one_based") == list(range(5, 13)),
                 f"{cell}: terminal seed/epoch policy drift")


def validate_four_terminal_common(terminals: dict[str, dict]) -> None:
    core.require(set(terminals) == set(core.CELLS), "four-cell terminal topology drift")
    fields = ("matching_authority_path", "matching_authority_sha256", "source_lineage_path",
              "source_lineage_sha256", "initial_state_path", "initial_state_file_sha256",
              "initial_state_dict_sha256", "seed", "epochs", "score_epochs_one_based",
              "python_isolation", "torch_runtime", "official_preflight_sha256",
              "implementation_bindings_sha256")
    reference = terminals[core.CELLS[0]]
    for cell, terminal in terminals.items():
        for field in fields:
            core.require(terminal.get(field) == reference.get(field),
                         f"four-cell terminal common binding drift: {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-json", type=Path)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--official-preflight", type=Path, default=core.OFFICIAL_PREFLIGHT_PATH)
    args = parser.parse_args()
    if args.score_root is not None:
        if args.out is not None:
            core.assert_immutable_pair_fresh(args.out, label="Stage-P aggregate")
        from scripts.misleading_identity_swap_v2_preflight import load_verified_official_preflight
        official, official_sha = load_verified_official_preflight(args.official_preflight)
        from scripts.misleading_identity_swap_v2_preflight import current_implementation_bindings
        current_bindings = current_implementation_bindings()
        scores = {cell: {domain: {} for domain in core.DOMAINS} for cell in core.CELLS}
        score_payloads = {cell: {domain: {} for domain in core.DOMAINS} for cell in core.CELLS}
        bindings = {}
        authority_by_domain = {domain: set() for domain in core.DOMAINS}
        terminals = {}
        for cell in core.CELLS:
            for domain in core.DOMAINS:
                for mode in core.EVAL_INPUT_MODES:
                    path = args.score_root / f"{cell}__{domain}__{mode}.json"
                    payload, digest = core.load_verified_immutable_json(path)
                    core.require(payload.get("receipt_kind") == "misleading_identity_swap_v2_domain_score",
                                 f"score receipt kind drift: {path}")
                    core.require((payload.get("cell"), payload.get("domain"), payload.get("evaluation_input_mode"))
                                 == (cell, domain, mode), f"score receipt topology drift: {path}")
                    core.require(payload.get("target_optimizer_or_backward_steps") == 0,
                                 "target update occurred")
                    core.require(payload.get("formal_subc_test_nwb_opened") is False,
                                 "formal data was opened")
                    core.require(payload.get("implementation_bindings") == current_bindings,
                                 f"score implementation closure drift: {path}")
                    core.require(payload.get("official_preflight_path") == str(core.OFFICIAL_PREFLIGHT_PATH.resolve()) and
                                 payload.get("official_preflight_sha256") == official_sha,
                                 f"score official-preflight binding drift: {path}")
                    terminal_path = Path(str(payload.get("cell_terminal_receipt_path", "")))
                    terminal, terminal_sha = core.load_verified_immutable_json(terminal_path)
                    core.require(terminal_sha == payload.get("cell_terminal_receipt_sha256"),
                                 f"score terminal receipt SHA drift: {path}")
                    expected_terminal_path = Path(official["cell_output_paths"][cell]) / "terminal_receipt.json"
                    core.require(terminal_path.resolve() == expected_terminal_path.resolve(),
                                 f"{cell}: terminal path is not canonical")
                    validate_terminal_against_official(
                        terminal, cell=cell, official=official, official_sha=official_sha,
                        current_bindings=current_bindings,
                    )
                    core.require(payload.get("source_matching_authority_path") == terminal.get("matching_authority_path") and
                                 payload.get("source_matching_authority_sha256") == terminal.get("matching_authority_sha256") and
                                 payload.get("source_lineage_path") == terminal.get("source_lineage_path") and
                                 payload.get("source_lineage_sha256") == terminal.get("source_lineage_sha256"),
                                 f"score/source terminal lineage drift: {path}")
                    if cell in terminals:
                        core.require(terminals[cell]["path"] == str(terminal_path.resolve()) and
                                     terminals[cell]["sha256"] == terminal_sha and
                                     terminals[cell]["payload"] == terminal,
                                     f"{cell}: score matrix references multiple terminal receipts")
                    else:
                        terminals[cell] = {"path": str(terminal_path.resolve()),
                                           "sha256": terminal_sha, "payload": terminal}
                    scores[cell][domain][mode] = float(payload["mean_r2"])
                    score_payloads[cell][domain][mode] = payload
                    bindings[f"{cell}/{domain}/{mode}"] = {"path": str(path.resolve()), "sha256": digest}
                    authority_by_domain[domain].add(payload["target_matching_authority_sha256"])
        for domain, digests in authority_by_domain.items():
            core.require(len(digests) == 1, f"{domain}: T4/Z4/cell target authority bytes differ")
        validate_four_terminal_common({cell: row["payload"] for cell, row in terminals.items()})
        aggregate = core.aggregate_stage_p(scores)
        per_session_interactions = {}
        for domain in core.DOMAINS:
            rosters = [set(score_payloads[cell][domain][mode]["per_session_mean_r2"])
                       for cell in core.CELLS for mode in core.EVAL_INPUT_MODES]
            core.require(all(roster == rosters[0] for roster in rosters),
                         f"{domain}: per-session score roster drift")
            per_session_interactions[domain] = {}
            for session in sorted(rosters[0]):
                value = lambda cell, mode: float(
                    score_payloads[cell][domain][mode]["per_session_mean_r2"][session]
                )
                per_session_interactions[domain][session] = (
                    value("swap_t4", "clean") - value("clean_t4", "clean")
                    - value("swap_z4", "clean") + value("clean_z4", "clean")
                )
        external_sessions = sorted(
            score_payloads["swap_t4"]["external_subject_M"]["clean"]["per_session_mean_r2"]
        )
        robustness_per_session = {}
        for session in external_sessions:
            value = lambda cell, mode: float(
                score_payloads[cell]["external_subject_M"][mode]["per_session_mean_r2"][session]
            )
            robustness_per_session[session] = (
                value("swap_t4", "swapped_diagnostic") - value("swap_t4", "clean")
                - value("clean_t4", "swapped_diagnostic") + value("clean_t4", "clean")
            )
        aggregate.update({
            "status": "DEVELOPMENT_STAGE_P_AGGREGATED__NOT_OFFICIAL",
            "score_receipts": bindings,
            "implementation_bindings": current_bindings,
            "official_preflight_path": str(core.OFFICIAL_PREFLIGHT_PATH.resolve()),
            "official_preflight_sha256": official_sha,
            "cell_terminal_receipts": {
                cell: {"path": row["path"], "sha256": row["sha256"]}
                for cell, row in terminals.items()
            },
            "target_authority_sha256_by_domain": {
                domain: next(iter(digests)) for domain, digests in authority_by_domain.items()
            },
            "carrier_interaction_per_session_non_rescuing": per_session_interactions,
            "external_t4_robustness_contrast_per_session_diagnostic_non_rescuing": robustness_per_session,
        })
        if args.out is None:
            print(json.dumps(aggregate, indent=2, sort_keys=True)); return 0
        _body, _side, digest = core.write_immutable_json_pair(args.out, aggregate)
        print(json.dumps({"status": aggregate["status"], "verdict": aggregate["verdict"],
                          "out": str(args.out.resolve()), "sha256": digest}, indent=2, sort_keys=True))
        return 0
    if args.scores_json is None:
        print(json.dumps({
            "status": "DRY_RUN__NO_SCORE_FILES_OPENED",
            "screen_id": core.SCREEN_ID,
            "required_matrix_shape": "4 cells x 2 domains x 2 input modes",
            "writes_output": False,
        }, indent=2, sort_keys=True))
        return 0
    scores = core.load_json_object(args.scores_json)
    aggregate = core.aggregate_stage_p(scores)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
