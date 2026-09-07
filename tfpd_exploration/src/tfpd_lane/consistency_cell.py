"""Route-owned Cell C: paired-subset consistency (R-Drop over whole-unit subsets).

Implements HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md section 3
"Cell C - paired-subset consistency" (items 1-6) without touching any sealed
Arm A / D / R / S2 file.  Per training step, on the SAME batch as Arm A / D
(batch 32, identical sampler and order)::

    p1 = p_stream.next(); p2 = p_stream.next()   # two draws per step, fixed order
    keep_1, keep_2  ~  independent per-(batch, unit) Bernoulli(1 - p_i) draws,
                      each identical in LAW to D's unit mask (spint.py:449-455),
                      one from each of two route-owned torch.Generator
                      namespaces (seeds 42_004 / 42_005)
    pred_1 = f(batch, keep_1)                    # forward 1, train() mode
    pred_2 = f(batch, keep_2)                    # forward 2, train() mode
    loss   = behavior_loss(pred_1, target) + behavior_loss(pred_2, target)
             + lambda * consistency(pred_1, pred_2)

- masking site / gain: the exact D site (after ``src = activity + identity``,
  before ``fc_in``), whole-unit keep mask, gain ``1/(1-p_i)`` for its branch;
  ``p_i == 1`` produces the all-zero branch tensor without division (D's exact
  edge semantics).  The shared ``PStream`` and the route-owned model pattern
  are imported from :mod:`src.tfpd_lane.sparsification` (never duplicated).
- behavior_loss is EXACTLY Arm A's masked MSE
  ``((pred - target)**2 * valid_rows).sum() / (valid_rows.sum() * C)`` per
  branch; the consistency term has the same functional form on the behaviour
  predictions ``(pred_1 - pred_2)`` -- same scale, never a latent.
- ``lambda = 0.1`` FROZEN; no sweep.
- skip floor 1 (handoff item 6): a branch whose keep count is 0 for a sample
  is skipped for that sample -- its behavior loss and the consistency term
  drop out for that sample; if BOTH branches are empty the sample contributes
  zero loss.  Triggers are counted and recorded per epoch.  NOTE: with D's law
  the marginal per-(sample, branch) trigger probability is E[p^N] = 1/(N+1)
  (~1.4% at N=70), so the floor is expected to bite regularly, not ~never;
  the recorded counts make the rate auditable instead of assumed.
- both forwards run in train() mode: each forward's internal decoder dropout
  (cross-attention / FFN / residual, p=0.1, global torch RNG) is an
  independent stochastic pass.  This two-stochastic-pass structure IS R-Drop
  (Liang et al., "R-Drop: Regularized Dropout for Neural Networks", NeurIPS
  2021, arXiv:2106.14448 -- task(pred_1) + task(pred_2) + lambda *
  divergence(pred_1, pred_2), with MSE in place of KL for regression); the
  route-owned change is the whole-unit perturbation, not the objective form.
- evaluation / scoring: single forward, mask exact ones, bitwise equal to the
  unsparsified parent path (asserted at launch and in tests).

Compute option (decided, recorded in the launch receipt): "two-view
same-batch" -- the same 32-sample batch per step as D, two forwards and ONE
backward on the summed loss; optimizer-step count identical to Arm A/D
(33,925 x 48); wall-clock ~2x D; each branch's marginal is identical to D's
training law.  Pre-registered: if C - D external mean is small, a
compute-matched D control at 96 epochs is required before any claim.
"""

from __future__ import annotations

import numpy as np
import torch

# single shared definitions; the route rule forbids duplicating these
from src.tfpd_lane import arm_common
from src.tfpd_lane.sparsification import (
    P_STREAM_SEED,
    PStream,
    SparsifiedStreamingSpintModel,
    _ensure_component_paths,
)

PAD_VALUE = -1.0
LAMBDA = 0.1
LAMBDA_FROZEN = True
DRAWS_PER_STEP = 2
SKIP_FLOOR = 1
NUM_HEADS = 2
BRANCH1_KEEP_SEED = 42_004
BRANCH2_KEEP_SEED = 42_005

COMPUTE_OPTION = (
    "two-view same-batch: the SAME 32-sample batch per step as Arm A/D, two "
    "forwards (one per independently drawn whole-unit subset) and ONE backward "
    "on the summed loss; optimizer-step count identical to Arm A/D "
    "(33,925 steps/epoch x 48 epochs); wall-clock ~2x D (~16h40m); effective "
    "distinct-sample batch = 32 (same as D); each branch's marginal is "
    "identical to D's training law"
)
PREREGISTRATION = (
    "if C - D external mean is small, a compute-matched D control at 96 epochs "
    "is required before any claim"
)


# ---------------------------------------------------------------------------
class PairedSubsetMasker:
    """Two independent whole-unit keep-mask streams, each identical in law to D's.

    Each branch owns a dedicated CPU ``torch.Generator`` (seeds 42_004 /
    42_005), so branch draws never touch the global torch RNG, the p stream,
    the data order, or each other.  A draw returns a float keep mask
    ``[B, N]`` (1 keep / 0 drop); the gain ``1/(1-p)`` is applied at the model
    site, exactly like the R/S2 route precedent.  ``p >= 1`` returns the
    all-zero mask (D's edge semantics; the model returns the all-zero branch
    tensor without division).
    """

    def __init__(
        self,
        seed_branch1: int = BRANCH1_KEEP_SEED,
        seed_branch2: int = BRANCH2_KEEP_SEED,
    ):
        self.seeds = {1: int(seed_branch1), 2: int(seed_branch2)}
        self._generators = {
            branch: torch.Generator().manual_seed(seed)
            for branch, seed in self.seeds.items()
        }
        self._ps: dict[int, list[float]] = {1: [], 2: []}
        self._kept: dict[int, list[float]] = {1: [], 2: []}
        self.all_zero_samples: dict[int, int] = {1: 0, 2: 0}

    def draw(self, branch: int, batch_size: int, n_units: int, p: float) -> torch.Tensor:
        """One whole-unit keep mask ``[B, N]`` (CPU, float32) for one branch."""
        if branch not in self._generators:
            raise ValueError(f"branch must be 1 or 2, got {branch!r}")
        if float(p) >= 1.0:
            mask = torch.zeros(batch_size, n_units)
        else:
            mask = torch.bernoulli(
                torch.full((batch_size, n_units), 1.0 - float(p)),
                generator=self._generators[branch],
            )
        self._ps[branch].append(float(p))
        if mask.numel():
            self._kept[branch].extend(mask.float().mean(dim=1).tolist())
        self.all_zero_samples[branch] += int((mask.sum(dim=1) == 0).sum().item())
        return mask

    def reset_epoch(self) -> None:
        for branch in (1, 2):
            self._ps[branch].clear()
            self._kept[branch].clear()
            self.all_zero_samples[branch] = 0

    def branch_summary(self, branch: int) -> dict:
        ps = np.asarray(self._ps[branch], dtype=np.float64)
        kept = np.asarray(self._kept[branch], dtype=np.float64)
        if ps.size == 0:
            return {"n_draws": 0, "n_samples": 0}
        sub_one = ps[ps < 1.0]
        return {
            "n_draws": int(ps.size),
            "n_samples": int(kept.size),
            "kept_fraction_mean": float(kept.mean()) if kept.size else None,
            "kept_fraction_min": float(kept.min()) if kept.size else None,
            "kept_fraction_q25": float(np.quantile(kept, 0.25)) if kept.size else None,
            "kept_fraction_median": float(np.quantile(kept, 0.50)) if kept.size else None,
            "kept_fraction_q75": float(np.quantile(kept, 0.75)) if kept.size else None,
            "kept_fraction_max": float(kept.max()) if kept.size else None,
            "p_max": float(ps.max()),
            "max_gain": float((1.0 / (1.0 - sub_one)).max()) if sub_one.size else None,
            "all_zero_branch_samples": int(self.all_zero_samples[branch]),
        }

    def summary(self) -> dict:
        return {"branch1": self.branch_summary(1), "branch2": self.branch_summary(2)}


class JaccardRecorder:
    """Per-sample |k1 intersect k2| / |k1 union k2| of the two branch keep sets.

    Handoff item 1: independent draws can overlap heavily, and when both draws
    are near-complete the consistency term is trivially satisfied; the realized
    overlap distribution is what distinguishes "the objective does not help"
    from "the constraint never bit".  Samples where BOTH keep sets are empty
    are skipped (0/0) and counted, never averaged in.
    """

    def __init__(self):
        self.values: list[float] = []
        self.skipped_both_empty = 0
        self.samples_seen = 0

    def update(self, keep_1: torch.Tensor, keep_2: torch.Tensor) -> None:
        k1 = keep_1.bool()
        k2 = keep_2.bool()
        inter = (k1 & k2).sum(dim=1)
        union = (k1 | k2).sum(dim=1)
        skip = union == 0
        self.skipped_both_empty += int(skip.sum().item())
        self.samples_seen += int(union.numel())
        if bool((~skip).any().item()):
            self.values.extend((inter[~skip] / union[~skip]).tolist())

    def summary(self) -> dict:
        base = {
            "n": len(self.values),
            "samples_seen": self.samples_seen,
            "skipped_both_empty": self.skipped_both_empty,
        }
        if not self.values:
            return base
        values = np.asarray(self.values, dtype=np.float64)
        base.update(
            {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(values.max()),
            }
        )
        return base

    def reset(self) -> None:
        self.values.clear()
        self.skipped_both_empty = 0
        self.samples_seen = 0


# ---------------------------------------------------------------------------
def arm_a_masked_mse(prediction: torch.Tensor, target: torch.Tensor,
                     row_weight: torch.Tensor) -> torch.Tensor:
    """Arm A's exact masked MSE on the rows selected by ``row_weight`` [B].

    ``((pred - target)^2 .sum(-1) * w).sum() / (w.sum() * C)``.  The
    denominator clamp only ever fires when the selected row set is empty
    (skip floor removed every row), yielding an exact zero loss instead of a
    0/0 NaN; with at least one selected row the denominator is ``>= C >= 1``
    and the clamp is a no-op.
    """
    diff2 = ((prediction - target) ** 2).sum(dim=-1)
    denom = row_weight.sum() * target.shape[-1]
    return (diff2 * row_weight).sum() / torch.clamp(denom, min=1.0)


def paired_consistency_loss(pred_1: torch.Tensor, pred_2: torch.Tensor,
                            behavior: torch.Tensor, keep_1: torch.Tensor,
                            keep_2: torch.Tensor, valid_rows: torch.Tensor,
                            lambda_: float = LAMBDA,
                            skip_floor: int = SKIP_FLOOR):
    """Cell C loss on one batch.  Returns (total, components).

    ``keep_1`` / ``keep_2`` are the branch keep masks [B, N] on the loss
    device; ``valid_rows`` is Arm A's behaviour-pad mask [B, W] (one flag per
    prediction window, exactly ``(behavior != PAD).all(-1)``).  The skip floor
    is per SAMPLE, so each branch's non-empty flag broadcasts across its
    windows.  The consistency term is on the behaviour predictions (never a
    latent) and shares Arm A's scale.
    """
    nonempty_1 = keep_1.sum(dim=1) >= skip_floor  # [B]
    nonempty_2 = keep_2.sum(dim=1) >= skip_floor  # [B]
    if valid_rows.dim() != 2 or valid_rows.shape[0] != keep_1.shape[0]:
        raise ValueError(
            f"valid_rows must be [B, W] with B={keep_1.shape[0]}, got {tuple(valid_rows.shape)}"
        )
    w1 = valid_rows & nonempty_1.unsqueeze(-1)  # [B, W]
    w2 = valid_rows & nonempty_2.unsqueeze(-1)  # [B, W]
    wc = w1 & w2
    loss_1 = arm_a_masked_mse(pred_1, behavior, w1)
    loss_2 = arm_a_masked_mse(pred_2, behavior, w2)
    consistency = arm_a_masked_mse(pred_1, pred_2, wc)
    total = loss_1 + loss_2 + lambda_ * consistency
    components = {
        "loss_branch1": loss_1.detach(),
        "loss_branch2": loss_2.detach(),
        "behavior_loss_sum": (loss_1 + loss_2).detach(),
        "consistency": consistency.detach(),
        "consistency_weighted": (lambda_ * consistency).detach(),
        "total": total.detach(),
        "rows_valid": valid_rows.sum(),
        "rows_branch1": w1.sum(),
        "rows_branch2": w2.sum(),
        "rows_consistency": wc.sum(),
        "branch1_skipped_samples": (~nonempty_1).sum(),
        "branch2_skipped_samples": (~nonempty_2).sum(),
        "both_branches_skipped_samples": (~nonempty_1 & ~nonempty_2).sum(),
    }
    return total, components


# ---------------------------------------------------------------------------
def build_consistency_model(seed: int = 42) -> "PairedConsistencySpintModel":
    """The exact Arm A/D graph (2 heads) with the paired-consistency decode."""
    _ensure_component_paths()
    from src.models.components.spint import SpintModel
    from src.models.components.streaming_encoders import build_encoder
    from src.models.components.streaming_spint import StreamingSpintModel

    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=512, num_covariates=2, window_size=50, num_heads=NUM_HEADS,
        num_layers=1, num_id_layers=1, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True,
        # the ROUTE paired masks replace D's built-in dynamic dropout at the
        # same site; the flag itself carries no parameters
        dynamic_dropout=False,
    )
    id_encoder = build_encoder(
        "B3S", window_size=50, trial_length=100, id_hidden_dim=128,
        hidden_dim=64, side_dim=4,
    )
    inner = StreamingSpintModel(decoder=decoder, id_encoder=id_encoder, decoder_mode="coupled")
    return PairedConsistencySpintModel(inner)


class PairedConsistencySpintModel(SparsifiedStreamingSpintModel):
    """Route-owned wrapper applying per-branch whole-unit masks at the D site.

    Inherits the parent replica of ``decode_with_identity`` unchanged, so the
    ONE insertion stays between ``src = src + identity`` and ``fc_in``.  Per
    branch forward the runner sets ``current_branch`` / ``current_p`` /
    ``current_keep_mask``; with sparsification disabled the mask is exact ones
    and the forward is bitwise equal to the unsparsified parent path.
    """

    def __init__(self, parent):
        super().__init__(parent, cell="C")
        self.current_branch: int | None = None
        self.current_keep_mask: torch.Tensor | None = None

    def apply_sparsification(self, src: torch.Tensor) -> torch.Tensor:
        if (
            not self.sparsification_enabled
            or self.current_p is None
            or self.current_keep_mask is None
        ):
            return src  # evaluation path: exact ones, bitwise parent-equal
        p = float(self.current_p)
        mask = self.current_keep_mask.to(device=src.device, dtype=src.dtype)
        if p >= 1.0:
            self.mask_stats.append(
                {"branch": self.current_branch, "p": p, "structure": "whole_unit_BxN",
                 "all_zero_branch": True, "kept_fraction": 0.0, "max_gain": None}
            )
            return torch.zeros_like(src)
        out = src * mask.unsqueeze(-1) / (1.0 - p)
        self.mask_stats.append(
            {"branch": self.current_branch, "p": p, "structure": "whole_unit_BxN",
             "all_zero_branch": False, "max_gain": 1.0 / (1.0 - p)}
        )
        return out


# ---------------------------------------------------------------------------
def train_epoch_paired_consistency(
    model: PairedConsistencySpintModel,
    optimizer,
    loader,
    lr_fn,
    device,
    p_stream: PStream,
    masker: PairedSubsetMasker,
    jaccard: JaccardRecorder,
    phase_step: int,
    num_heads: int = NUM_HEADS,
    lambda_: float = LAMBDA,
    max_steps: int | None = None,
) -> dict:
    """Route-owned Cell C epoch: two masked forwards per step, one optimizer step."""
    model.train()
    encoder_params, decoder_params = arm_common.param_groups_by_branch(model)
    w_param = model.id_encoder.post_pool[0].weight
    encoder = model.id_encoder
    hidden, side_dim = encoder.hidden_dim, encoder.side_dim
    acc = {
        "loss_total": torch.zeros((), device=device),
        "loss_branch1": torch.zeros((), device=device),
        "loss_branch2": torch.zeros((), device=device),
        "behavior_loss_sum": torch.zeros((), device=device),
        "consistency": torch.zeros((), device=device),
        "rows_valid": torch.zeros((), device=device, dtype=torch.long),
        "rows_branch1": torch.zeros((), device=device, dtype=torch.long),
        "rows_branch2": torch.zeros((), device=device, dtype=torch.long),
        "rows_consistency": torch.zeros((), device=device, dtype=torch.long),
        "branch1_skipped": torch.zeros((), device=device, dtype=torch.long),
        "branch2_skipped": torch.zeros((), device=device, dtype=torch.long),
        "both_skipped": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_loss": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_grad": torch.zeros((), device=device, dtype=torch.long),
        "wside_grad_nonzero": torch.zeros((), device=device, dtype=torch.long),
    }
    steps = examples = 0
    for batch in loader:
        neural, behavior, calib, sessions, side = batch[:5]
        session = sessions[0] if isinstance(sessions, (list, tuple)) else sessions
        del session  # Cell C masks are per-(batch, unit); no session authority
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)
        lr = lr_fn(phase_step + steps)
        optimizer.param_groups[0]["lr"] = float(lr)

        # two p draws per step in fixed order (branch 1 then branch 2)
        p1 = p_stream.next()
        p2 = p_stream.next()
        batch_size = int(neural.shape[0])
        n_units = int(neural.shape[-1])
        keep1_cpu = masker.draw(1, batch_size, n_units, p1)
        keep2_cpu = masker.draw(2, batch_size, n_units, p2)
        jaccard.update(keep1_cpu, keep2_cpu)
        keep1 = keep1_cpu.to(device)
        keep2 = keep2_cpu.to(device)

        # both forwards in train() mode: the decoder's internal dropout makes
        # each an independent stochastic pass (the R-Drop structure)
        model.sparsification_enabled = True
        model.current_branch = 1
        model.current_p = p1
        model.current_keep_mask = keep1
        pred_1, _identity1 = model(neural, calib_trials=calib, side_features=side)
        model.current_branch = 2
        model.current_p = p2
        model.current_keep_mask = keep2
        pred_2, _identity2 = model(neural, calib_trials=calib, side_features=side)
        model.sparsification_enabled = False  # never leak into eval paths

        valid_rows = (behavior != PAD_VALUE).all(dim=-1)
        loss, comp = paired_consistency_loss(
            pred_1, pred_2, behavior, keep1, keep2, valid_rows, lambda_=lambda_
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grads = [q.grad for q in encoder_params + decoder_params if q.grad is not None]
            total_sq = (
                torch.stack(torch._foreach_norm(grads)).pow(2).sum()
                if grads
                else torch.zeros((), device=device)
            )
            acc["nonfinite_grad"] += (~torch.isfinite(total_sq)).long()
            grad = w_param.grad
            if grad is not None:
                acc["wside_grad_nonzero"] += (grad[:, hidden : hidden + side_dim] != 0).sum()
            acc["nonfinite_loss"] += (~torch.isfinite(loss)).long()
        optimizer.step()

        acc["loss_total"] += comp["total"]
        acc["loss_branch1"] += comp["loss_branch1"]
        acc["loss_branch2"] += comp["loss_branch2"]
        acc["behavior_loss_sum"] += comp["behavior_loss_sum"]
        acc["consistency"] += comp["consistency"]
        for acc_key, comp_key in (
            ("rows_valid", "rows_valid"),
            ("rows_branch1", "rows_branch1"),
            ("rows_branch2", "rows_branch2"),
            ("rows_consistency", "rows_consistency"),
            ("branch1_skipped", "branch1_skipped_samples"),
            ("branch2_skipped", "branch2_skipped_samples"),
            ("both_skipped", "both_branches_skipped_samples"),
        ):
            acc[acc_key] += comp[comp_key]
        steps += 1
        examples += batch_size
        if max_steps is not None and steps >= max_steps:
            break
    n = max(steps, 1)
    return {
        "optimizer_steps": steps,
        "forwards": 2 * steps,
        "train_example_windows": examples,
        "train_loss_mean_per_step": float(acc["loss_total"].item()) / n,
        "behavior_loss_sum_mean_per_step": float(acc["behavior_loss_sum"].item()) / n,
        "consistency_loss_mean_per_step": float(acc["consistency"].item()) / n,
        "consistency_weighted_mean_per_step": float(acc["consistency"].item()) * lambda_ / n,
        "branch1_loss_mean_per_step": float(acc["loss_branch1"].item()) / n,
        "branch2_loss_mean_per_step": float(acc["loss_branch2"].item()) / n,
        "rows_valid_total": int(acc["rows_valid"].item()),
        "rows_branch1_total": int(acc["rows_branch1"].item()),
        "rows_branch2_total": int(acc["rows_branch2"].item()),
        "rows_consistency_total": int(acc["rows_consistency"].item()),
        "branch1_skipped_samples": int(acc["branch1_skipped"].item()),
        "branch2_skipped_samples": int(acc["branch2_skipped"].item()),
        "both_branches_skipped_samples": int(acc["both_skipped"].item()),
        "lambda": lambda_,
        "num_heads": num_heads,
        "nonfinite_loss_steps": int(acc["nonfinite_loss"].item()),
        "nonfinite_grad_steps": int(acc["nonfinite_grad"].item()),
    }


__all__ = [
    "BRANCH1_KEEP_SEED",
    "BRANCH2_KEEP_SEED",
    "COMPUTE_OPTION",
    "DRAWS_PER_STEP",
    "JaccardRecorder",
    "LAMBDA",
    "LAMBDA_FROZEN",
    "NUM_HEADS",
    "P_STREAM_SEED",
    "PREREGISTRATION",
    "PStream",
    "PairedConsistencySpintModel",
    "PairedSubsetMasker",
    "SKIP_FLOOR",
    "SparsifiedStreamingSpintModel",
    "arm_a_masked_mse",
    "build_consistency_model",
    "paired_consistency_loss",
    "train_epoch_paired_consistency",
]
