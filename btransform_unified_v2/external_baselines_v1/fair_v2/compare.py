#!/usr/bin/env python3
"""Join completed, sealed FAIR V2 receipts; never fit or select a model."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TASKS = ("m1", "m2", "h1")
ARMS = {
    "wf_zs_h0": "WF-ZS, one smoothed bin",
    "diag_z_wf": "diag-z + WF, ten smoothed bins",
    "coral_wf": "CORAL + WF, source-selected shrinkage",
    "aligned_fa_wf": "AlignedFA + WF, all-electrode posterior",
    "aligned_fa_stable_wf": "AlignedFA + WF, stable-only posterior variant",
    "static_identity": "Static RIFT, fixed-final EMA",
    "static_diag_z": "Static RIFT + diag-z, same frozen EMA",
    "static_coral": "Static RIFT + CORAL, same frozen EMA",
    "rift_fixed_final": "Full RIFT, fixed-final EMA",
    "rift_historical_ho_selected": "Full RIFT, historical local-HO epoch selection",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main_scores(task: str, report: dict) -> dict:
    if task == "h1":
        group = report["grouped_seven"]
        assert group["n_groups"] == 7
        standard, legacy = group["standard_mean"], group["legacy_mean"]
    else:
        standard, legacy = report["standard_equal_session_mean"], report["legacy_equal_session_mean"]
    return {
        "standard": standard,
        "legacy": legacy,
        "aggregation": "equal mean of seven groups, concatenate recordings within group" if task == "h1" else "equal recording mean",
        "n_recordings": report["n_sessions"],
        "n_windows": report["n_windows"],
        "recording_mean_diagnostic": {
            "standard": report["standard_equal_session_mean"],
            "legacy": report["legacy_equal_session_mean"],
        },
    }


def build() -> dict:
    verification_path = RESULTS / "verification_v2/receipt.json"
    verification = read(verification_path)
    assert verification["status"] == "PASS"
    reference_path = RESULTS / "rift_reference_v2/receipt.json"
    reference = read(reference_path)
    readiness = read(HERE.parent / "submissions/READINESS.json")
    frozen = readiness["packages"]
    assert len(frozen) == 6
    assert all(x["status"] == "FROZEN_PROTOCOL_INVALID_FOR_COMPARISON" and not x["submission_eligible"] for x in frozen)
    result = {
        "schema": "fair_v2_completed_comparison_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_LOCAL_RESEARCH_RESULTS",
        "scope": "public calibration query faces; no official test scores or new submissions",
        "final_comparison_surface": "EvalAI official held-out only; local scores diagnose implementation/protocol and do not establish final method ranking",
        "generator_sha256": sha(Path(__file__)),
        "input_receipt_sha256": {},
        "metric_definitions": {
            "standard": "1 - sum_o SSE_o / sum_o sum_t (Y_to - mean_t Y_to)^2; sklearn variance_weighted R2",
            "legacy": "1 - SSE / sum_to (Y_to - mean_to Y_to)^2; historical flattened R2",
            "aggregation": "M1: equal 3-recording mean; M2: equal 6-recording mean; H1: equal 7-group mean after concatenation of the paired recordings",
        },
        "selection_rules": {
            "linear": "fixed source grids; per-source chronological 80/20 CV, 21-bin endpoint purge; final pooled source refit",
            "static": "same existing fixed-final EMA within each task; identity/diag/CORAL prescribed, no target-label choice among them",
            "rift_fixed_final": "retrospective target-independent final-epoch reporting rule: M1/M2 E24, H1 E32; old curves had visible local target labels",
            "rift_historical_ho_selected": "historical local-HO earliest-maximum epoch, target query labels used; separate selection regime",
        },
        "h1_channel_correspondence": "positional unit rows assumed; physical correspondence across dates remains unverified",
        "frozen_v1_packages": [x["package"] for x in frozen],
        "tasks": {},
    }
    for path in (verification_path, reference_path):
        result["input_receipt_sha256"][str(path.relative_to(HERE.parent))] = sha(path)
    for task in TASKS:
        linear_path = RESULTS / f"{task}_v2/receipt.json"
        linear = read(linear_path)
        assert linear["status"] == "COMPLETED"
        assert not linear["target_labels_used_for_selection"] and not linear["target_labels_used_for_fit"]
        static_dir = RESULTS / f"static_{task}_v2"
        static_path = static_dir / ("report_rescored.json" if task != "h1" else "report.json")
        static = read(static_path)
        assert static["task"] == task and static["device"] == "cpu"
        for path in (linear_path, static_path):
            result["input_receipt_sha256"][str(path.relative_to(HERE.parent))] = sha(path)
        rows = {}
        for arm, report in linear["reports"].items():
            rows[arm] = main_scores(task, report)
            rows[arm]["selected"] = linear["selected"][arm]
        for arm, report in static["arms"].items():
            rows[f"static_{arm}"] = main_scores(task, report)
            rows[f"static_{arm}"]["checkpoint_sha256"] = static["checkpoint_sha256"]
        for rule in ("fixed_final", "historical_ho_selected"):
            row = reference["tasks"][task][rule]
            rows[f"rift_{rule}"] = {
                "standard": row["standard_equal_session_or_group_mean"],
                "legacy": row["legacy_equal_session_or_group_mean"],
                "epoch": row["epoch"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "target_labels_used_for_epoch_selection": row["target_labels_used_for_epoch_selection"],
                "n_windows": row["n_windows"],
                "score_provenance": "algebraic restatement of sealed per-recording/group score using matching-Y SST; no new network forward",
            }
        assert len({row["n_windows"] for row in rows.values()}) == 1
        pairs = {
            "coral_wf_minus_diag_z_wf": ("coral_wf", "diag_z_wf"),
            "static_diag_z_minus_identity": ("static_diag_z", "static_identity"),
            "static_coral_minus_diag_z": ("static_coral", "static_diag_z"),
            "rift_fixed_minus_diag_z_wf": ("rift_fixed_final", "diag_z_wf"),
            "rift_fixed_minus_static_diag_z": ("rift_fixed_final", "static_diag_z"),
        }
        result["tasks"][task] = {
            "source_recordings": linear["source_session_count"],
            "source_windows": linear["source_windows"],
            "target_support_native_bins": {s: r["support_provenance"]["native_bins"] for s, r in linear["target_calibration"].items()},
            "rows": rows,
            "paired_deltas": {name: {metric: rows[a][metric] - rows[b][metric] for metric in ("standard", "legacy")} for name, (a, b) in pairs.items()},
        }
    return result


def document(result: dict) -> str:
    lines = [
        "# Fair gradient-free controls v2: completed local results",
        "",
        "Final method comparisons use EvalAI official held-out results only. The numbers in this document are local "
        "public-calibration diagnostics; they neither establish nor refute an official held-out advantage for RIFT. "
        "The six v1 packages are frozen and ineligible for submission, and their old recommendation has been withdrawn. "
        "The corrected all-source, source-selected linear controls no longer exhibit the severe negative-score failure on this local face. "
        "CORAL adds essentially zero over diagonal calibration here. Static + diag-z also gives a strong M1 local control; "
        "this finding is an implementation/protocol diagnostic, not a final model ranking.",
        "",
        "These are local public-calibration query scores. All table entries below use the same task-specific query rows. "
        "Standard R² centers each behavioral output separately before variance weighting; legacy R² uses one global mean of flattened behavior. "
        "M1 averages three recordings, M2 six recordings, and H1 seven groups after concatenating each group's two recordings. "
        "The JSON also preserves the H1 14-recording diagnostic; it must not replace the seven-group main result.",
        "",
        "| Method | M1 standard | M1 legacy | M2 standard | M2 legacy | H1 standard | H1 legacy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in ARMS.items():
        values = [f"{result['tasks'][task]['rows'][arm][metric]:.6f}" for task in TASKS for metric in ("standard", "legacy")]
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines += [
        "",
        "The final row has a different selection budget: historical full-RIFT epochs were selected using local held-out query labels. "
        "It is retained for continuity in these local diagnostics; neither local row substitutes for an official held-out result. "
        "The fixed-final full-RIFT row uses E24/E24/E32 and the historical row E3/E10/E16 (M1/M2/H1). "
        "Fixed-final is a retrospective target-independent reporting rule applied to existing curves; it is not evidence that the historical curves were unseen. "
        "Full-RIFT standard/legacy pairs are algebraically recovered from sealed per-recording or per-group scores and matching target SST; no new full-RIFT inference was performed.",
        "",
        "## Paired increments",
        "",
        "| Difference, standard R² | M1 | M2 | H1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    delta_labels = {
        "coral_wf_minus_diag_z_wf": "CORAL + WF minus diag-z + WF",
        "static_diag_z_minus_identity": "Static + diag-z minus static identity",
        "static_coral_minus_diag_z": "Static + CORAL minus static + diag-z",
        "rift_fixed_minus_diag_z_wf": "Full RIFT fixed-final minus diag-z + WF",
        "rift_fixed_minus_static_diag_z": "Full RIFT fixed-final minus static + diag-z",
    }
    for key, label in delta_labels.items():
        values = [f"{result['tasks'][task]['paired_deltas'][key]['standard']:+.8f}" for task in TASKS]
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines += [
        "",
        "Source CV selected CORAL shrinkage 1.0 for all three tasks. In the linear implementation this replaces the covariance "
        "with an isotropic matrix, removing off-diagonal alignment. Residual score differences below 5e-6 are not evidence of "
        "cross-channel CORAL benefit. All prescribed static arms are reported; none was picked using target scores. "
        "Their CORAL setting was fixed in advance at diagonal covariance shrinkage 0.1 and ridge 0.001, so this experiment does not establish an optimum over static CORAL settings.",
        "",
        "## Protocol and source selection",
        "",
        "All five linear arms pool every canonical source recording for final supervised fitting. "
        "Each source is split chronologically 80:20 by eligible endpoint, with a 21-native-bin endpoint separation to prevent causal-feature overlap. "
        "Selection maximizes equal-source-recording standard R², then refits on all eligible source rows. "
        "The saved source selection is sealed before target evaluation data are opened. "
        "Each full, unpadded raw recording is filtered continuously from zero state using the normalized FALCON 12-tap exponential "
        "kernel (20-ms bins, tau 240 ms, extent 1), without trial resets. Session mean/std are fitted only on that session's "
        "selected native support bins after the same filter. The ridge intercept is unpenalized; behavior is not standardized. "
        "CORAL and AlignedFA align every non-reference source to the latest source reference and then pool all source labels.",
        "",
        "| Task | Source recordings / supervised windows | Target recordings / scored windows | diag-z alpha | AFA alpha / K / stable fraction |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for task in TASKS:
        item = result["tasks"][task]
        rows = item["rows"]
        afa = rows["aligned_fa_stable_wf"]["selected"]
        config = afa["configuration"]
        lines.append(f"| {task.upper()} | {item['source_recordings']} / {item['source_windows']:,} | {rows['diag_z_wf']['n_recordings']} / {rows['diag_z_wf']['n_windows']:,} | {rows['diag_z_wf']['selected']['alpha']:g} | {afa['alpha']:g} / {config['latent_dim']} / {config['stable_fraction']:g} |")
    lines += [
        "",
        "The alpha grid is {1e2, 1e3, 1e4, 1e5}; CORAL shrinkage is {0, 0.1, 0.5, 1}; "
        "FA dimension is {10, 20, 40}; stable fraction is {0.5, 0.75, 1}. "
        "The one-bin WF-ZS arm selected alpha 1e3 in all three tasks. "
        "FA uses sufficient-statistics covariance EM with three deterministic starts, per-sample tolerance 1e-6, "
        "private-variance floor 1e-6 and a 10,000-iteration budget (one 50,000-iteration retry at the same tolerance). "
        "All stored source fits and selected target fits passed convergence checks. "
        "Both posterior variants selected K40 and the same fraction within each task; M1 selected 0.75, M2/H1 1.0. "
        "The all-electrode posterior follows the author release's inference form. Stable-only posterior is the separately labeled requested robustness variant. "
        "The Degenhart author code uses stable rows to estimate rotation, then all electrodes for posterior inference. "
        "This corrects the premise that all-electrode inference itself departed from the published method.",
        "",
        "Native support uses the same trial budgets: M1 first 10 eval-valid trials, M2 first 33 trials, H1 first 3 valid TrialNum trials. "
        "All three tasks read native NWB neural bins; none uses interpolated calibration matrices. "
        "Every target recording's exact bin count, trial IDs, NWB hash and support indices are in its receipt. "
        "The static-network arms use the same raw support but map it to pooled source raw-support mean/covariance before local_conv; "
        "they do not add WF smoothing to a network trained on raw inputs. "
        "All network weights remain identical within the identity/diag/CORAL triplet. "
        "H1 continues to assume corresponding positional unit rows across dates; physical channel correspondence is unverified.",
        "",
        "The one-bin arm is a FALCON-style local reproduction: it matches the demo's default history and exact filter, "
        "while using the consistent calibration-statistics and source-selection contract stated here. "
        "The official demo has different normalization surfaces, alpha grid and CV scoring. "
        "Published private-test WF-ZS scores are not directly comparable to this public local face. "
        "The user's provisional M2 value near 0.16 was not a retained receipt; this exact declared protocol produced 0.136042. "
        "The result is reported as measured, without target-based retuning.",
        "",
        "## Evidence and reproduction",
        "",
        "- [Protocol and commands](../fair_v2/README.md), [primary-source implementation audit](OFFICIAL_WF_AFA_AUDIT.md).",
        "- [Machine-readable comparison](../fair_v2/results/comparison_v2.json) binds every input score receipt by SHA-256.",
        "- [Independent linear verification](../fair_v2/results/verification_v2/receipt.json): reload saved numeric models without fitting, independently stream all held-out recordings, maximum prediction difference 0.0; scores, source selections, native support and code hashes verified.",
        "- Linear receipts: [M1](../fair_v2/results/m1_v2/receipt.json), [M2](../fair_v2/results/m2_v2/receipt.json), [H1](../fair_v2/results/h1_v2/receipt.json).",
        "- Static receipts: [M1](../fair_v2/results/static_m1_v2/report_rescored.json), [M2](../fair_v2/results/static_m2_v2/report_rescored.json), [H1](../fair_v2/results/static_h1_v2/report.json); [static implementation and validation](FAIR_V2_STATIC_RESULTS.md).",
        "- [Full-RIFT metric restatement](../fair_v2/results/rift_reference_v2/receipt.json) binds original score receipts, checkpoint hashes, target hashes and both SST denominators.",
        "- [V1 package freeze](../submissions/READINESS.json). No v2 image was pushed, no method registered, and no new official submission made for this work.",
        "",
        "Regenerate this table after all receipts exist: `python fair_v2/compare.py` from `external_baselines_v1`. "
        "The joiner reads JSON only; it performs no fitting, prediction or model selection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    result = build()
    (RESULTS / "comparison_v2.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (HERE.parent / "docs/FAIR_V2_RESULTS.md").write_text(document(result))
    print(json.dumps({task: {arm: {key: row[key] for key in ("standard", "legacy")} for arm, row in result["tasks"][task]["rows"].items()} for task in TASKS}, indent=2))


if __name__ == "__main__":
    main()
