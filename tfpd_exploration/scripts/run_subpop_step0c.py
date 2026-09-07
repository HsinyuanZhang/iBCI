#!/usr/bin/env python3
"""Step 0C — test-time unit-loss robustness curve and subset ensembling.

Implements HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3
"Step 0C" (and the §5 receipt requirements).  ZERO GPU training:
inference-only measurements on two EXISTING sealed checkpoints, scored on the
external 15 sub-M sessions under the exact A2-matched convention
(`src/tfpd_lane/matched_scorer.session_r2`, last-bin-of-window,
variance-weighted R2, equal weight per session, strict-27 source-only
normalizers with semantic SHA f062506c..., final-four SWA).

(i)  Unit-loss robustness curve.  Whole-unit Bernoulli removal at the EXACT D
     site — after `src = activity + identity`, before `fc_in`
     (`spint.py:445-458`) — on a route-owned eval wrapper that replicates the
     parent decode path (`src/tfpd_lane/sparsification.py` is NOT imported or
     modified; this file owns its own copy of the wrapper):
       - ``zero_nogain`` PRIMARY, deployment semantics: dropped units' src rows
         zeroed, survivors untouched (a unit silently lost);
       - ``zero_gain``   secondary, D's training semantics: same, survivors
         scaled by 1/(1-f);
       - ``padding``     diagnostic, Cell-T preview: dropped units excluded via
         ``key_padding_mask`` passed to ``model.decoder.transformer(...)`` —
         src NOT zeroed, no gain, so no constant placeholder token is created.
     Grid {0, 0.1, 0.25, 0.5, 0.75} x 3 mask seeds x both checkpoints;
     padding at the 4 nonzero fractions x 1 seed.  Fraction 0 must reproduce
     the sealed native governing external numbers bit-exactly
     (arm A 0.2603564786414305, D 0.4179362749059995 — LOADED from the
     SHA-verified sealed receipts, never hardcoded).

(ii) Test-time subset ensembling on the SAME unmodified checkpoints.  For
     K in {1, 4, 8}, member m draws p_m ~ U(0,1) and a whole-unit
     Bernoulli(p_m) mask (D's training law), decodes with ``zero_gain``
     (primary) / ``zero_nogain`` (secondary), and predictions are AVERAGED
     across members per window before scoring; plus the native baseline.

Authorization, loader, integrity and receipt conventions are copied from
``scripts/run_sparsify_score.py`` / ``scripts/run_pop_robust_score.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
# Three-layer `src` namespace (tfpd_exploration / streaming_calibration_exp /
# sua_exploration) resolved exactly once, ROOT first, then the streaming src
# appended to the same package path — the run_pop_robust_score.py pattern.
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))
import src as _src_pkg  # noqa: E402  (tfpd_exploration/src wins)

_STREAMING_SRC = str(REPO / "streaming_calibration_exp" / "src")
if _STREAMING_SRC not in _src_pkg.__path__:
    _src_pkg.__path__.append(_STREAMING_SRC)

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BOUND_PATTERNS = (
    "scripts/run_subpop_step0c.py",
    "tests/test_subpop_step0c.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/arm_common.py",
    "scripts/run_a2_matched_rescore.py",
)
SEPARATE_SESSION = "sub-M_ses-CO-20141203"

# ---- frozen experiment grid (HANDOFF §3 Step 0C) --------------------------
SPEC_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75)
SPEC_N_MASK_SEEDS = 3
SPEC_ENSEMBLE_K = (1, 4, 8)
MIN_KEEP = 4  # padding-only floor; HANDOFF §3 Cell T trap (fully-masked softmax)
SEED_SALT = "subpop_step0c_v1"
SEED_RULE_DOC = (
    "torch.Generator().manual_seed(int.from_bytes(sha256('"
    + SEED_SALT
    + "|{condition_key}|{seed_idx}|{member_idx}').digest()[:8], 'big') "
    "% (2**63 - 1)); member_idx is 'none' for single-mask conditions; each "
    "generator is consumed sequentially over the FIXED evaluation order "
    "(sessions sorted by name, chunks of 128 windows); the rule is "
    "checkpoint-independent, so arm A and D receive bit-identical mask draws "
    "(paired masks) on the identical batch order"
)

# ---- sealed artifacts (verified before any data is opened) ----------------
CHECKPOINTS = {
    "armA": {
        "path": ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt",
        # bound from the spec / the arm-A terminal receipt (status ARM_TERMINAL)
        "bound_sha256": "920eb4c9827dbe66f98e110d9e4880d6f2ea4180fe9acc8ac30cacf3073f6ad0",
        "terminal_receipt": ROOT / "results/admission_arms_v1/armA_direct_t4_48/terminal_receipt.json",
        "terminal_status": "ARM_TERMINAL",
        "note": "arm A direct T4 48-epoch final-four SWA (no dropout)",
    },
    "D": {
        "path": ROOT / "results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt",
        # the exact value is READ from the terminal receipt at run time and
        # bound there (nothing is trusted about the file before that check)
        "bound_sha256": None,
        "terminal_receipt": ROOT / "results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json",
        "terminal_status": "CELL_TERMINAL",
        "note": "cell D — arm A recipe + dynamic_dropout(0,1), 2 heads, final-four SWA",
    },
}
# sealed native governing external references, loaded (never hardcoded)
REFERENCE_RECEIPTS = {
    "armA_external_native": {
        "path": ROOT / "results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json",
        "sha256": "0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e",
        "pointer": ("results", "armA_direct_t4_48_swa_lastbin", "external", "mean_r2"),
        "source": "a2_matched_rescore_v1_r1 (SHA-verified)",
    },
    "D_external_native": {
        "path": ROOT / "results/sparsification_score_v1/sparsification_score_receipt.json",
        "sha256": "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f",
        "pointer": ("results", "D_swa", "governing_last_bin", "external", "mean_r2"),
        "source": "sparsification_score_v1 (SHA-verified)",
    },
}
INTERVENTIONS = ("none", "zero_nogain", "zero_gain", "padding")
CURVE_KINDS = ("zero_nogain", "zero_gain")  # zero_nogain is PRIMARY
ENSEMBLE_KINDS = ("zero_gain", "zero_nogain")  # zero_gain is primary here


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def intervention_generator_seed(condition_key: str, seed_idx: int,
                                member_idx: int | None = None) -> int:
    """Documented deterministic seed per (condition, seed index, member)."""
    payload = "|".join((
        SEED_SALT, str(condition_key), str(int(seed_idx)),
        "none" if member_idx is None else str(int(member_idx)),
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def date_block(session: str) -> str:
    return "2014" if "-2014" in session else ("2015" if "-2015" in session else "other")


# ---------------------------------------------------------------------------
# route-owned eval wrapper (independent of src/tfpd_lane/sparsification.py)
# ---------------------------------------------------------------------------
class SubpopEvalModel(torch.nn.Module):
    """Parent decode-path replica with ONE insertion at the exact D site.

    Holds the parent's ``decoder`` / ``id_encoder`` by reference (same module
    objects, same names, no parameter copy), replicates
    ``StreamingSpintModel.decode_with_identity`` for the coupled,
    non-routed, ungated B3S graph, and inserts the unit-loss intervention
    between ``src = activity + identity`` and ``fc_in``.  With intervention
    ``"none"`` the output is BITWISE equal to the parent eval path (the parent
    multiplies by an exact-ones dropout mask there, and x * 1.0 == x).
    """

    def __init__(self, parent, min_keep: int = MIN_KEEP):
        super().__init__()
        self.decoder = parent.decoder
        self.id_encoder = parent.id_encoder
        if getattr(parent, "decoder_mode", "coupled") != "coupled":
            raise ValueError("Step 0C requires the coupled per-unit decoder")
        if getattr(parent, "fixed_slot_router", None) is not None:
            raise ValueError("Step 0C requires the non-routed decoder graph")
        self.min_keep = int(min_keep)
        self.intervention = "none"
        self.fraction = 0.0
        self.mask_generator: torch.Generator | None = None
        self.reset_mask_stats()
        self.eval()  # inference-only by construction (decode guards on training)

    # -- intervention state ------------------------------------------------
    def set_intervention(self, intervention: str, fraction: float,
                         generator: torch.Generator | None = None):
        if intervention not in INTERVENTIONS:
            raise ValueError(f"unknown intervention {intervention!r}")
        if intervention == "none":
            fraction = 0.0
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in [0,1], got {fraction!r}")
        self.intervention = intervention
        self.fraction = float(fraction)
        self.mask_generator = generator
        return self

    def reset_mask_stats(self) -> None:
        self.mask_stats = {
            "n_mask_draws": 0,
            "n_units_masked_total": 0,
            "n_units_total": 0,
            "kept_fraction_min": 1.0,
            "kept_fraction_max": 0.0,
            "n_samples_all_units_dropped": 0,
            "n_samples_min_keep_floored": 0,
        }

    def consume_mask_stats(self) -> dict:
        stats = dict(self.mask_stats)
        n = stats["n_units_total"]
        stats["realized_kept_fraction_mean"] = (
            float(stats["n_units_masked_total"] / n) if n else None
        )
        if stats["kept_fraction_max"] < stats["kept_fraction_min"]:
            stats["kept_fraction_min"] = None
            stats["kept_fraction_max"] = None
        return stats

    # -- masks -------------------------------------------------------------
    def _draw_keep(self, batch: int, units: int, keep_prob: float) -> torch.Tensor:
        """Whole-unit Bernoulli keep mask [B, N] (1 keep / 0 drop), CPU-drawn
        with the condition generator then moved to the src device, so the draw
        is device-independent and exactly reproducible."""
        if self.mask_generator is None:
            raise RuntimeError("an intervention is active but no generator is set")
        mask = torch.bernoulli(
            torch.full((batch, units), float(keep_prob)), generator=self.mask_generator
        )
        return mask

    def _record(self, keep: torch.Tensor) -> torch.Tensor:
        per_sample = keep.float().mean(dim=1)
        s = self.mask_stats
        s["n_mask_draws"] += 1
        s["n_units_masked_total"] += int(keep.sum().item())
        s["n_units_total"] += int(keep.numel())
        s["kept_fraction_min"] = min(s["kept_fraction_min"], float(per_sample.min().item()))
        s["kept_fraction_max"] = max(s["kept_fraction_max"], float(per_sample.max().item()))
        s["n_samples_all_units_dropped"] += int((keep.sum(dim=1) == 0).sum().item())
        return keep

    def _enforce_min_keep(self, keep: torch.Tensor) -> torch.Tensor:
        """Padding-only floor: a sample with fewer than ``min_keep`` survivors
        keeps the ``min_keep`` lowest-canonical-index units instead (Cell-T
        trap: a fully masked softmax returns NaN, which D never hits)."""
        starved = keep.sum(dim=1) < self.min_keep
        if bool(starved.any()):
            fix = torch.zeros_like(keep)
            fix[:, : self.min_keep] = 1.0
            keep = torch.where(starved.unsqueeze(1), fix, keep)
            self.mask_stats["n_samples_min_keep_floored"] += int(starved.sum().item())
        return keep

    # -- the exact D site ---------------------------------------------------
    def apply_intervention(self, src: torch.Tensor):
        """Returns (src, key_padding_mask).  The mask multiplication happens
        after `src = src + identity` and before `fc_in` — spint.py:445-458."""
        if self.intervention == "none":
            return src, None
        fraction = self.fraction
        batch, units = src.shape[0], src.shape[1]
        if self.intervention == "padding":
            keep = self._record(self._draw_keep(batch, units, 1.0 - fraction))
            keep = self._enforce_min_keep(keep)
            # True = excluded from the attention softmax; src NOT zeroed
            return src, ~keep.to(device=src.device, dtype=torch.bool)
        keep = self._record(self._draw_keep(batch, units, 1.0 - fraction)).to(
            device=src.device, dtype=src.dtype
        )
        masked = src * keep.unsqueeze(-1)  # dropped rows -> exact zeros
        if self.intervention == "zero_nogain" or fraction >= 1.0:
            return masked, None
        return masked / (1.0 - fraction), None  # zero_gain; f>=1 never divides

    def decode_with_identity(self, neural: torch.Tensor, identity: torch.Tensor,
                             neuron_gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.training:
            raise RuntimeError("the Step 0C wrapper is inference-only; call .eval()")
        src = neural.permute(0, 2, 1)
        if neuron_gate is not None:
            src = src * neuron_gate
        src = src + identity
        src, key_padding_mask = self.apply_intervention(src)  # <-- the site
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(
            rep.repeat(src.size(0), 1, 1), src, key_padding_mask=key_padding_mask
        )
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def compute_identity(self, calib_trials, side_features=None, electrode_ids=None):
        return self.id_encoder.forward_batch(
            calib_trials, side_features=side_features, electrode_ids=electrode_ids
        )

    def forward(self, neural, calib_trials=None, identity=None, side_features=None,
                electrode_ids=None):
        neuron_gate = None
        if identity is None:
            if calib_trials is None:
                raise ValueError("either calib_trials or identity must be provided")
            if hasattr(self.id_encoder, "forward_batch_with_gate"):
                identity, neuron_gate = self.id_encoder.forward_batch_with_gate(
                    calib_trials, side_features=side_features
                )
            else:
                identity = self.compute_identity(
                    calib_trials, side_features=side_features, electrode_ids=electrode_ids
                )
        behavior = self.decode_with_identity(neural, identity, neuron_gate=neuron_gate)
        return behavior, identity


def build_step0c_model(seed: int = 42):
    """The exact arm-A/D graph that produced the sealed reference numbers.

    Both SWAs are strict-loaded into the A2-shaped ``StreamingSpintModel``
    built by the sealed ``src/tfpd/spintshape_module.build_spintshape_model``
    (2 heads, ``dynamic_dropout=False``) — exactly what
    ``run_sparsify_score.py`` does for ``armA_swa`` and ``D_swa``.
    ``dynamic_dropout`` carries no parameters and is train-mode-only, so the
    loaded graph is score-identical to the D-shaped build at eval.
    """
    from src.tfpd.spintshape_module import build_spintshape_model

    return build_spintshape_model(seed=seed)


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------
def verify_checkpoints(sha256_file) -> dict:
    """SHA-bind both SWAs against their terminal receipts BEFORE any data open.

    The D value is read from the terminal receipt and bound there; arm A is
    checked against both the spec-bound constant and its terminal receipt.
    """
    torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
    integrity = {}
    for key, spec in CHECKPOINTS.items():
        path = Path(spec["path"])
        receipt_path = Path(spec["terminal_receipt"])
        if not path.is_file():
            raise SystemExit(f"checkpoint missing: {path}")
        if not receipt_path.is_file():
            raise SystemExit(f"terminal receipt missing: {receipt_path}")
        payload = json.loads(receipt_path.read_text())
        problems = []
        if payload.get("status") != spec["terminal_status"]:
            problems.append(f"status {payload.get('status')!r}")
        receipt_sha = payload.get("swa_sha256", payload.get("swa", {}).get("sha256"))
        if receipt_sha is None:
            problems.append("terminal receipt carries no swa sha256")
        live_sha = sha256_file(path)
        if receipt_sha is not None and live_sha != receipt_sha:
            problems.append("SWA SHA mismatch vs terminal receipt")
        bound = spec["bound_sha256"] or receipt_sha
        if bound is not None and live_sha != bound:
            problems.append("SWA SHA mismatch vs bound value")
        sidecar = Path(str(path) + ".sha256")
        if sidecar.is_file() and sidecar.read_text().split()[0] != live_sha:
            problems.append("sidecar SHA mismatch")
        if problems:
            raise SystemExit(f"checkpoint {key} integrity failed: {problems}")
        integrity[key] = {
            "path": str(path),
            "sha256": live_sha,
            "bound_sha256": bound,
            "bound_from": "spec+terminal receipt" if spec["bound_sha256"] else "terminal receipt",
            "terminal_receipt": str(receipt_path),
            "terminal_receipt_sha256": sha256_file(receipt_path),
            "terminal_status": payload.get("status"),
            "epochs_run": payload.get("epochs_run"),
            "note": spec["note"],
        }
    return integrity


def load_reference_values(sha256_file) -> dict:
    """Sealed native governing external means, LOADED from SHA-verified receipts."""
    references = {}
    for key, spec in REFERENCE_RECEIPTS.items():
        path = Path(spec["path"])
        if not path.is_file():
            raise SystemExit(f"reference receipt missing: {path}")
        body_sha = sha256_file(path)
        if body_sha != spec["sha256"]:
            raise SystemExit(f"reference receipt SHA mismatch ({key}): {body_sha}")
        payload = json.loads(path.read_text())
        node = payload
        for step in spec["pointer"]:
            node = node[step]
        references[key] = {
            "value": float(node),
            "receipt": str(path),
            "receipt_sha256": body_sha,
            "pointer": "/".join(spec["pointer"]),
            "source": spec["source"],
        }
    return references


# ---------------------------------------------------------------------------
# condition forwards
# ---------------------------------------------------------------------------
def native_forward(wrapper: SubpopEvalModel):
    wrapper.set_intervention("none", 0.0)

    def forward(neural, calib, side, _w=wrapper):
        return _w(neural, calib_trials=calib, side_features=side)

    return forward


def curve_forward(wrapper: SubpopEvalModel, kind: str, fraction: float,
                  seed_idx: int, condition_key: str):
    generator = torch.Generator().manual_seed(
        intervention_generator_seed(condition_key, seed_idx)
    )
    wrapper.set_intervention(kind, fraction, generator)

    def forward(neural, calib, side, _w=wrapper):
        return _w(neural, calib_trials=calib, side_features=side)

    return forward


def ensemble_forward(wrapper: SubpopEvalModel, kind: str, k: int, seed_idx: int,
                     condition_key: str):
    """K members, each p_m ~ U(0,1) + whole-unit Bernoulli(p_m) mask (D's law).

    The identity is computed once per batch; member decodes differ only in the
    mask draw.  Predictions are averaged per window before scoring.
    """
    p_generator = torch.Generator().manual_seed(
        intervention_generator_seed(f"{condition_key}__p", seed_idx)
    )
    p_values = [
        float(torch.rand((), generator=p_generator).item()) for _ in range(k)
    ]
    member_generators = [
        torch.Generator().manual_seed(
            intervention_generator_seed(condition_key, seed_idx, member_idx=m)
        )
        for m in range(k)
    ]

    def forward(neural, calib, side, _w=wrapper, _ps=p_values, _gs=member_generators):
        identity = _w.compute_identity(calib, side_features=side)
        total = None
        for p_value, generator in zip(_ps, _gs):
            _w.set_intervention(kind, p_value, generator)
            member = _w.decode_with_identity(neural, identity)
            total = member if total is None else total + member
        return total / k, identity

    return forward, p_values


def score_condition(rescorer, matched_scorer, forward, ext_ds, ext_starts, device,
                    label: str) -> dict:
    result = rescorer.score_last_bin(
        forward, ext_ds, ext_starts, device, matched_scorer.session_r2, output_scale=1.0
    )
    for row in result["per_session"]:
        if not math.isfinite(row["r2"]):
            raise SystemExit(f"non-finite per-session R2 in {label}: {row}")
    if not math.isfinite(result["mean_r2"]):
        raise SystemExit(f"non-finite governing mean in {label}")
    return result


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--authorize-target", default="")
    parser.add_argument("--fractions", type=float, nargs="+", default=None)
    parser.add_argument("--n-mask-seeds", type=int, default=None)
    parser.add_argument("--ensemble-k", type=int, nargs="+", default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--limit-windows-per-session", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    authorized = args.authorize_target == AUTH_VALUE
    if not args.dry_run and not authorized:
        print("external scoring requires --authorize-target " + AUTH_VALUE, file=sys.stderr)
        return 3

    smoke = bool(args.smoke)
    fractions = tuple(args.fractions) if args.fractions is not None else (
        (0.0, 0.5) if smoke else SPEC_FRACTIONS
    )
    n_mask_seeds = args.n_mask_seeds if args.n_mask_seeds is not None else (
        1 if smoke else SPEC_N_MASK_SEEDS
    )
    ensemble_k = tuple(args.ensemble_k) if args.ensemble_k is not None else (
        (2,) if smoke else SPEC_ENSEMBLE_K
    )
    limit_sessions = args.limit_sessions
    if smoke and limit_sessions is None:
        limit_sessions = 2
    limit_windows = args.limit_windows_per_session
    if smoke and limit_windows is None:
        limit_windows = 2000
    out_root = Path(args.output_root) if args.output_root is not None else (
        ROOT / ("results/subpop_step0c_v1_smoke" if smoke else "results/subpop_step0c_v1")
    )

    plan = {
        "schema": "tfpd_subpop_step0c_v1",
        "handoff": "HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md #3 Step 0C",
        "checkpoints": sorted(CHECKPOINTS),
        "surface": "external 15 sub-M sessions (A2-matched convention)",
        "scorer": "src/tfpd_lane/matched_scorer.session_r2 (last bin, variance-weighted, equal session weight)",
        "fractions": list(fractions),
        "n_mask_seeds": n_mask_seeds,
        "ensemble_k": list(ensemble_k),
        "padding": {"fractions": [f for f in fractions if f > 0.0], "n_seeds": 1},
        "min_keep_padding": MIN_KEEP,
        "seed_rule": SEED_RULE_DOC,
        "smoke": smoke,
        "limit_sessions": limit_sessions,
        "limit_windows_per_session": limit_windows,
        "authorized": authorized,
    }
    if args.dry_run:
        print(json.dumps({**plan, "status": "DRY_RUN__NO_NWB_OPENED"}, indent=1))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3
    if out_root.exists():
        print(f"fresh output root required: {out_root}", file=sys.stderr)
        return 2
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_root.mkdir(parents=True)

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    rescorer = _load_module(
        "tfpd_a2_rescorer_step0c", ROOT / "scripts/run_a2_matched_rescore.py"
    )
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- integrity BEFORE any data is opened ------------------------------
    checkpoint_integrity = verify_checkpoints(arm_common.sha256_file)
    references = load_reference_values(arm_common.sha256_file)

    # ---- external surface, exact A2-matched convention --------------------
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    surfaces = rescorer.build_surfaces(args, a2, "t4")
    (_within_ds, _within_starts), (ext_ds, ext_starts) = surfaces
    if limit_sessions is not None:
        keep_sessions = sorted(ext_starts)[: int(limit_sessions)]
        ext_starts = {s: ext_starts[s] for s in keep_sessions}
    if limit_windows is not None:
        ext_starts = {s: v[: int(limit_windows)] for s, v in ext_starts.items()}
    n_sessions = len(ext_starts)
    print(json.dumps({"surface": "external", "n_sessions": n_sessions,
                      "sessions": sorted(ext_starts),
                      "limit_windows_per_session": limit_windows}), flush=True)

    results: dict[str, dict] = {}
    integrity_blocks: dict[str, dict] = {}
    for model_key in ("armA", "D"):
        path = Path(CHECKPOINTS[model_key]["path"])
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {(k[len("model."):] if k.startswith("model.") else k): v
                 for k, v in state.items()}
        model = build_step0c_model(seed=42)
        model.load_state_dict(state, strict=True)
        del state
        state_before = arm_common.state_sha256(model)
        model.to(device).eval()
        wrapper = SubpopEvalModel(model, min_keep=MIN_KEEP)
        wrapper.eval()
        model_results: dict[str, dict] = {}

        def register(label, kind, fraction, seed_rows, extra=None):
            mean_over_seeds = float(np.mean([r["mean_r2"] for r in seed_rows]))
            per_session = {}
            for row in seed_rows:
                for entry in row["per_session"]:
                    per_session.setdefault(entry["session"], []).append(entry["r2"])
            n_windows = {e["session"]: e.get("n_windows")
                         for e in seed_rows[0]["per_session"]}
            model_results[label] = {
                "intervention": kind,
                "fraction_removed": fraction,
                "n_seeds": len(seed_rows),
                "governing_mean_r2": mean_over_seeds,
                "per_seed": seed_rows,
                "per_session_mean_over_seeds": {
                    s: float(np.mean(v)) for s, v in per_session.items()
                },
                "n_windows_per_session": n_windows,
                **(extra or {}),
            }

        # native baseline (wrapper with no intervention) --------------------
        wrapper.reset_mask_stats()
        native = score_condition(rescorer, matched_scorer, native_forward(wrapper),
                                 ext_ds, ext_starts, device, f"{model_key}/native")
        register("native", "none", 0.0, [native | {"seed_idx": None,
                                                   "mask_stats": wrapper.consume_mask_stats()}])

        # (i) unit-loss robustness curve -----------------------------------
        for kind in CURVE_KINDS:
            for fraction in fractions:
                label = f"{kind}_f{fraction}"
                seed_rows = []
                for seed_idx in range(n_mask_seeds):
                    wrapper.reset_mask_stats()
                    forward = curve_forward(wrapper, kind, float(fraction), seed_idx, label)
                    row = score_condition(rescorer, matched_scorer, forward, ext_ds,
                                          ext_starts, device, f"{model_key}/{label}/s{seed_idx}")
                    row["seed_idx"] = seed_idx
                    row["mask_stats"] = wrapper.consume_mask_stats()
                    seed_rows.append(row)
                    print(json.dumps({"model": model_key, "condition": label,
                                      "seed": seed_idx,
                                      "mean_r2": round(row["mean_r2"], 6)}), flush=True)
                register(label, kind, float(fraction), seed_rows)

        # padding diagnostic (nonzero fractions, single seed) ---------------
        for fraction in [f for f in fractions if f > 0.0]:
            label = f"padding_f{fraction}"
            wrapper.reset_mask_stats()
            forward = curve_forward(wrapper, "padding", float(fraction), 0, label)
            row = score_condition(rescorer, matched_scorer, forward, ext_ds, ext_starts,
                                  device, f"{model_key}/{label}")
            row["seed_idx"] = 0
            row["mask_stats"] = wrapper.consume_mask_stats()
            register(label, "padding", float(fraction), [row])
            print(json.dumps({"model": model_key, "condition": label,
                              "mean_r2": round(row["mean_r2"], 6)}), flush=True)

        # (ii) test-time subset ensembling ---------------------------------
        for kind in ENSEMBLE_KINDS:
            for k in ensemble_k:
                label = f"ens_{kind}_K{k}"
                seed_rows = []
                for seed_idx in range(n_mask_seeds):
                    wrapper.reset_mask_stats()
                    forward, p_values = ensemble_forward(wrapper, kind, int(k),
                                                         seed_idx, label)
                    row = score_condition(rescorer, matched_scorer, forward, ext_ds,
                                          ext_starts, device, f"{model_key}/{label}/s{seed_idx}")
                    row["seed_idx"] = seed_idx
                    row["member_p_values"] = p_values
                    row["member_expected_gain"] = [
                        (None if p >= 1.0 else round(1.0 / (1.0 - p), 6)) for p in p_values
                    ]
                    row["mask_stats"] = wrapper.consume_mask_stats()
                    seed_rows.append(row)
                    print(json.dumps({"model": model_key, "condition": label,
                                      "seed": seed_idx, "mean_r2": round(row["mean_r2"], 6)}),
                          flush=True)
                register(label, kind, None, seed_rows,
                         extra={"k_members": int(k),
                                "member_p_law": "p_m ~ U(0,1) per member (D's training law)"})

        state_after = arm_common.state_sha256(model)
        if state_before != state_after:
            raise SystemExit(f"scoring mutated the checkpoint state: {model_key}")
        integrity_blocks[model_key] = {
            **checkpoint_integrity[model_key],
            "loaded_graph": "src/tfpd/spintshape_module.build_spintshape_model(seed=42)",
            "graph_note": (
                "2 heads, dynamic_dropout=False (the run_sparsify_score.py build for "
                "both armA_swa and D_swa); dynamic_dropout carries no parameters and "
                "is train-mode-only, so the eval score is identical to the D-shaped build"
            ),
            "strict_load": True,
            "state_unchanged_during_scoring": True,
            "grads_all_none_after": all(
                p.grad is None for p in model.parameters()
                if not isinstance(p, torch.nn.parameter.UninitializedParameter)
            ),
        }
        results[model_key] = model_results
        del model, wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- fraction-0 cross-check against the sealed native numbers ---------
    crosscheck = {}
    for model_key, ref_key in (("armA", "armA_external_native"), ("D", "D_external_native")):
        native_mean = results[model_key]["native"]["governing_mean_r2"]
        sealed = references[ref_key]["value"]
        f0 = {}
        for kind in CURVE_KINDS:
            label = f"{kind}_f0.0"
            if label in results[model_key]:
                f0[kind] = {
                    "mean_r2": results[model_key][label]["governing_mean_r2"],
                    "minus_native": results[model_key][label]["governing_mean_r2"] - native_mean,
                }
        crosscheck[model_key] = {
            "native_reproduced": native_mean,
            "sealed_reference": sealed,
            "native_minus_sealed": native_mean - sealed,
            "fraction_zero_points": f0,
            "reference_receipt": references[ref_key]["receipt"],
            "reference_receipt_sha256": references[ref_key]["receipt_sha256"],
            "note": ("delta should be 0.0 (identical scorer path and graph); "
                     "any nonzero value is reported as measured"),
        }

    # ---- 2014/2015 date blocks + the separately reported session ----------
    governing_tables = {
        model_key: {
            label: block["per_session_mean_over_seeds"]
            for label, block in model_results.items()
        }
        for model_key, model_results in results.items()
    }
    n_windows_reference = results["armA"]["native"]["n_windows_per_session"]
    date_blocks = {}
    for model_key, tables in governing_tables.items():
        for label, table in tables.items():
            for session, value in table.items():
                if session == SEPARATE_SESSION:
                    bucket = "separate_20141203"
                else:
                    bucket = date_block(session)
                date_blocks.setdefault(model_key, {}).setdefault(label, {}).setdefault(
                    bucket, {}
                )[session] = value
    block_statistics = {}
    for model_key, labels in date_blocks.items():
        for label, buckets in labels.items():
            for bucket, entries in buckets.items():
                values = [entries[s] for s in sorted(entries)]
                block_statistics.setdefault(model_key, {}).setdefault(label, {})[bucket] = {
                    "n_sessions": len(values),
                    "block_mean": float(np.mean(values)),
                    "window_share_of_external": float(
                        sum(n_windows_reference.get(s, 0) for s in entries)
                        / max(sum(n_windows_reference.values()), 1)
                    ),
                    "sessions": sorted(entries),
                }

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    receipt = {
        "schema": "tfpd_subpop_step0c_v1",
        "status": "SUBPOP_STEP0C_SMOKE" if smoke else "SUBPOP_STEP0C_SCORED",
        "handoff": "HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md #3 Step 0C / #5",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": plan,
        "integrity": {
            "checkpoints": checkpoint_integrity,
            "intervention_definitions": {
                "site": "after `src = activity + identity`, before `fc_in` (spint.py:445-458)",
                "zero_nogain": (
                    "PRIMARY (deployment semantics: units silently lost): whole-unit "
                    "Bernoulli(f) drop, dropped units' src rows zeroed, survivors untouched"
                ),
                "zero_gain": (
                    "secondary (D's training semantics): same drop, survivors scaled 1/(1-f)"
                ),
                "padding": (
                    "diagnostic (Cell-T preview): dropped units excluded via "
                    "key_padding_mask [B,N] (True=excluded) into "
                    "decoder.transformer(...); src NOT zeroed, no gain; min_keep floor "
                    f"{MIN_KEEP} by lowest canonical unit index"
                ),
                "mask_structure": "per (batch sample, unit) whole-window Bernoulli keep mask [B,N]",
                "seed_rule": SEED_RULE_DOC,
            },
            "native_reference_receipts": references,
            "scored_models": integrity_blocks,
        },
        "native_crosscheck": crosscheck,
        "results": results,
        "n_windows_per_session_external": n_windows_reference,
        "governing_tables_per_session": governing_tables,
        "external_date_block_statistics": {
            "note": ("governing last-bin external, seed-averaged per session; "
                     "separate_20141203 excluded from the year blocks"),
            "blocks": block_statistics,
        },
        "granularity_labels": {
            "governing": "last timestep of window, variance-weighted R2, equal session weight",
        },
        "disclosures": {
            "gpu_training_steps": 0,
            "target_updates_gradients_optimizer_steps": 0,
            "checkpoint_selection_performed": False,
            "checkpoints_modified": False,
            "formal_or_organizer_held_data_opened": False,
            "normalizer_refit_on_target": False,
            "sealed_files_modified": False,
            "smoke_run": smoke,
            "sessions_truncated_to": limit_sessions,
            "windows_per_session_truncated_to": limit_windows,
        },
        "source_closure": {
            "launch": closure,
            "final": closure_final,
            "launch_final_closure_equal": closure["closure_sha256"] == closure_final["closure_sha256"],
        },
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_root / "step0c_receipt.json", receipt)

    summary = {"smoke": smoke, "n_sessions": n_sessions,
               "native_crosscheck": {
                   m: {"reproduced": round(v["native_reproduced"], 10),
                       "sealed": round(v["sealed_reference"], 10),
                       "delta": round(v["native_minus_sealed"], 12)}
                   for m, v in crosscheck.items()}}
    for model_key, model_results in results.items():
        summary[model_key] = {
            "curve_zero_nogain_PRIMARY": {
                f"f{f}": round(model_results[f"zero_nogain_f{f}"]["governing_mean_r2"], 6)
                for f in fractions if f"zero_nogain_f{f}" in model_results
            },
            "curve_zero_gain": {
                f"f{f}": round(model_results[f"zero_gain_f{f}"]["governing_mean_r2"], 6)
                for f in fractions if f"zero_gain_f{f}" in model_results
            },
            "padding_diagnostic": {
                f"f{f}": round(model_results[f"padding_f{f}"]["governing_mean_r2"], 6)
                for f in fractions if f > 0.0 and f"padding_f{f}" in model_results
            },
            "ensemble": {
                label: round(block["governing_mean_r2"], 6)
                for label, block in model_results.items() if label.startswith("ens_")
            },
        }
    print(json.dumps(summary, indent=1))
    print(json.dumps({"receipt": str(out_root / "step0c_receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
