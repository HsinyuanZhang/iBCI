"""Shared constants and disclosures for fair_v2 EvalAI packs."""
from __future__ import annotations

import hashlib
from pathlib import Path

PACK = Path(__file__).resolve().parent
FAIR_V2 = PACK.parent
BASELINES = FAIR_V2.parent
UNIFIED_V2 = BASELINES.parent
WORKSPACE = UNIFIED_V2.parent
RESULTS = FAIR_V2 / "results"
SUBMISSIONS = BASELINES / "submissions"

CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
TEAM = "HKU-ECE"
TEAM_ID = 41975

TASKS = {
    "m1": {
        "base_image": "spint-original-m1:e9-epoch019-052e9ea",
        "max_batch": 4,
        "channels": 64,
        "outputs": 16,
        "roster": 7,
        "context": 100,
        "heldin": 4,
        "heldout": 3,
        "cal_budget": "first 10 eval-valid trials",
        "behavior_scale": 1.0,
    },
    "m2": {
        "base_image": "spint-m2:e8-epoch027-76f0fb2",
        "max_batch": 7,
        "channels": 96,
        "outputs": 2,
        "roster": 13,
        "context": 50,
        "heldin": 7,
        "heldout": 6,
        "cal_budget": "first 33 trials",
        "behavior_scale": 5.0,
    },
    "h1": {
        "base_image": "h1-epfilm-c1:no-readout-v1-523d3d2e",
        "max_batch": 8,
        "channels": 176,
        "outputs": 7,
        "roster": 27,
        "context": 300,
        "heldin": 13,
        "heldout": 14,
        "cal_budget": "first 3 valid TrialNum trials",
        "behavior_scale": 20.0,
    },
}

STATIC_CKPTS = {
    "m1": {
        "path": UNIFIED_V2 / "learnable_recency_v1/results/m1_static_s42/epoch_024.pt",
        "sha256": "562e0febc2818e2fc6952e383a5dea7fadbc6493034cee3557a315e1538cf4df",
        "epoch": 24,
    },
    "m2": {
        "path": UNIFIED_V2 / "learnable_recency_v1/results/m2_static_learned_slope_s42/epoch_024.pt",
        "sha256": "8313e7bb202d43def2885b7e7c4c74dc8466af81aaccb18b437333ae86801612",
        "epoch": 24,
    },
    "h1": {
        "path": UNIFIED_V2 / "learnable_recency_v1/results/h1_static_s42/epoch_032.pt",
        "sha256": "b79adeda51955c14bb60e4643835a5d8d6862cdb2480134fd9fd42a04a765b51",
        "epoch": 32,
    },
}

# Public-calibration diagnostics only. Official ranking is EvalAI held-out.
LOCAL_SCORES = {
    ("m1", "diag_z_wf"): {"standard": 0.461263, "legacy": 0.583432},
    ("m2", "diag_z_wf"): {"standard": 0.136042, "legacy": 0.136046},
    ("h1", "diag_z_wf"): {"standard": 0.114324, "legacy": 0.115412},
    ("m1", "coral_wf"): {"standard": 0.461263, "legacy": 0.583432},
    ("m2", "coral_wf"): {"standard": 0.136037, "legacy": 0.136041},
    ("h1", "coral_wf"): {"standard": 0.114322, "legacy": 0.115410},
    ("m1", "aligned_fa_stable_wf"): {"standard": 0.368145, "legacy": 0.510457},
    ("m2", "aligned_fa_stable_wf"): {"standard": 0.081052, "legacy": 0.081057},
    ("h1", "aligned_fa_stable_wf"): {"standard": 0.057677, "legacy": 0.058838},
    ("m1", "static_rift_diag_z"): {"standard": 0.586175, "legacy": 0.681561},
    ("m2", "static_rift_diag_z"): {"standard": 0.271371, "legacy": 0.271375},
    ("h1", "static_rift_diag_z"): {"standard": 0.256093, "legacy": 0.257010},
    ("m1", "static_rift_coral"): {"standard": 0.536408, "legacy": 0.643304},
    ("m2", "static_rift_coral"): {"standard": 0.248885, "legacy": 0.248889},
    ("h1", "static_rift_coral"): {"standard": 0.276458, "legacy": 0.277351},
}

SELECTION_BUDGET = (
    "New linear baselines used source-only hyperparameter selection "
    "(80/20 chronological per source recording, 21-native-bin gap). "
    "Query labels did not enter fit or selection. "
    "Existing official RIFT candidates used public calibration query labels "
    "for epoch pick; that is a different selection budget. "
    "Fixed-final RIFT rows are local diagnostics only and do not replace "
    "existing official RIFT candidates."
)

# Offline windowed vs sealed static preds are float32, not bit-exact.
# Host/container streaming uses the same bound (see replay_static.py).
STATIC_EXPORT_TOLERANCE = 2e-5

LINEAR_PROTOCOL = (
    "Shared WF: 240 ms causal exponential smooth (FALCON 12-tap, tau=240ms, "
    "extent=1) on each full unpadded raw 20-ms recording from zero state, "
    "no trial reset; per-session z-score from that session's native "
    "calibration bins after the same filter; then aligner; 10-bin causal "
    "history; unpenalized-intercept ridge readout. No interpolated support."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_for(tag: str) -> str:
    return "s_" + "".join(c if c.isalnum() else "_" for c in tag)


def package_name(task: str, arm: str) -> str:
    return f"{task}_{arm}_v2"


def image_tag(task: str, arm: str) -> str:
    return f"fair-v2-{task}-{arm}:cpu"


def method_name(task: str, arm: str) -> str:
    labels = {
        "diag_z_wf": "diag-z + WF",
        "coral_wf": "CORAL + WF (source-selected shrinkage=1)",
        "aligned_fa_stable_wf": "AlignedFA stable-posterior + WF",
        "static_rift_diag_z": "static-RIFT + diag-z",
        "static_rift_coral": "static-RIFT + CORAL",
    }
    return f"Fair v2 {task.upper()} {labels[arm]}"


def method_description(task: str, arm: str, selected: dict | None = None) -> str:
    scores = LOCAL_SCORES[(task, arm)]
    h1 = (
        " H1 physical cross-date channel correspondence is unverified; "
        "positional unit rows are assumed."
        if task == "h1"
        else ""
    )
    budget = " " + SELECTION_BUDGET
    if arm == "diag_z_wf":
        alpha = selected.get("alpha") if selected else "?"
        return (
            f"Fair-v2 linear floor: {LINEAR_PROTOCOL} Session z-score only "
            f"(no cross-channel aligner). Source-selected ridge alpha={alpha}. "
            f"Local public-calibration standard R2={scores['standard']:.6f} "
            f"(not official).{h1}{budget}"
        )
    if arm == "coral_wf":
        return (
            f"Fair-v2 linear CORAL+WF. {LINEAR_PROTOCOL} Source-selected "
            "CORAL shrinkage=1.0 on this task, which replaces the covariance "
            "with an isotropic matrix and removes off-diagonal alignment. "
            "This is a declared degeneracy / official confirmation arm, not an "
            "independent cross-channel CORAL increment. Residual local |Δ| vs "
            f"diag-z is <5e-6. Local standard R2={scores['standard']:.6f} "
            f"(not official).{h1}{budget}"
        )
    if arm == "aligned_fa_stable_wf":
        cfg = (selected or {}).get("configuration", {})
        return (
            "Fair-v2 AlignedFA STABLE-SUBSET POSTERIOR variant + WF "
            "(not the author-form all-electrode posterior). "
            f"{LINEAR_PROTOCOL} Source-selected K={cfg.get('latent_dim', 40)}, "
            f"stable fraction={cfg.get('stable_fraction', '?')}. "
            "Fits passed convergence checks. Solver settings and normalized "
            "input differ from the original raw-count MATLAB experiment. "
            f"Local standard R2={scores['standard']:.6f} (not official).{h1}{budget}"
        )
    if arm == "static_rift_diag_z":
        return (
            "Same-capacity static-RIFT control: one already-trained static EMA "
            "checkpoint; only the pre-local_conv neural calibration changes. "
            "Network keeps raw-input processing; no WF smoothing is added. "
            "Frontend is per-session diag-z mapping target raw support onto "
            "pooled source raw-support mean/std. Same frozen network weights "
            "as the static+CORAL sibling. "
            f"Local standard R2={scores['standard']:.6f} (not official).{h1}{budget}"
        )
    if arm == "static_rift_coral":
        return (
            "Same-capacity static-RIFT control: same frozen EMA checkpoint as "
            "static+diag-z; only the pre-local_conv neural calibration changes. "
            "No WF smoothing. Static CORAL was fixed in advance "
            "(diagonal covariance shrinkage 0.1, ridge 0.001), not "
            "source-selected like the linear CORAL arm. "
            f"Local standard R2={scores['standard']:.6f} (not official).{h1}{budget}"
        )
    raise ValueError(arm)
