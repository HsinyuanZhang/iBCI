"""Version-B source-LOSO windows with rSyn3/Zero4 carriers, encoder side_dim=0."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import carrier_bank
from . import plan


class DatamoduleError(RuntimeError):
    """Fail closed for Stage-1 data construction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatamoduleError(message)


def _session_name(session: Any) -> str:
    if isinstance(session, bytes):
        text = session.decode("ascii")
    else:
        text = str(session)
    if text.startswith("ses-"):
        return text
    raise DatamoduleError(f"unrecognized session name {session!r}")


class RSyn3CarrierDataset:
    """4-tuple Falcon windows plus the arm's per-session 4-vector carrier."""

    def __init__(self, base: Any, bank: dict[str, Any], arm: str) -> None:
        _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
        self.base = base
        self.bank = bank
        self.arm = arm

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        neural, target, calib, session = self.base[index][:4]
        name = _session_name(session)
        carrier = carrier_bank.carrier_for_arm(self.bank, name, self.arm)
        n_units = int(np.asarray(calib).shape[-1])
        _require(carrier.shape == (n_units, 4), f"carrier/unit mismatch {carrier.shape} vs {n_units}")
        return neural, target, calib, session, np.ascontiguousarray(carrier, dtype=np.float32)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


VersionBInferenceWindowDataModule: type | None = None


def _version_b_inference_window_class(base: type) -> type:
    """Bind the inference-only Version-B subclass once the no-minival base is importable."""
    global VersionBInferenceWindowDataModule
    cached = VersionBInferenceWindowDataModule
    if isinstance(cached, type) and issubclass(cached, base):
        return cached

    class VersionBInferenceWindowDataModule(base):
        """Version-B's `_assert_contract` freezes the held-in query to [10,210) because that is its training/selection window; this inference-only subclass relaxes ONLY that one check to the three pre-declared M1 internal-LOSO windows and enforces every other Version-B contract check unchanged. It must never be used for training."""

        def _assert_contract(self, stage):
            hparams = self.hparams
            raw_start = hparams.heldin_query_start_trial
            raw_end = hparams.heldin_query_end_trial
            start = int(raw_start)
            end = None if raw_end is None else int(raw_end)
            _require(
                (start, end) in plan.FULL_QUERY_ALLOWED_WINDOWS,
                f"query window {(start, end)!r} not pre-declared",
            )
            hparams.heldin_query_start_trial = plan.QUERY_START
            hparams.heldin_query_end_trial = plan.QUERY_STOP_EXCLUSIVE
            try:
                super()._assert_contract(stage)
            finally:
                hparams.heldin_query_start_trial = raw_start
                hparams.heldin_query_end_trial = raw_end
            restored = (
                int(hparams.heldin_query_start_trial),
                None if hparams.heldin_query_end_trial is None else int(hparams.heldin_query_end_trial),
            )
            _require(restored == (start, end), "held-in query window drifted after Version-B contract check")
            self.version_b_contract_relaxation = {
                "field": "heldin_query_window",
                "frozen_window": [10, 210],
                "used_window": [start, end],
                "all_other_contract_checks_enforced": True,
                "inference_only": True,
            }

        def get_split_manifest(self):
            raise DatamoduleError("inference window datamodule has no training split manifest")

    bound = VersionBInferenceWindowDataModule
    bound.__module__ = __name__
    bound.__qualname__ = "VersionBInferenceWindowDataModule"
    globals()["VersionBInferenceWindowDataModule"] = bound
    return bound


def make_datamodule(
    repo_root: Path,
    bank: dict[str, Any],
    arm: str,
    *,
    query_window: tuple[int, int | None] = (plan.QUERY_START, plan.QUERY_STOP_EXCLUSIVE),
):
    query_window = tuple(query_window)
    _require(query_window in plan.FULL_QUERY_ALLOWED_WINDOWS, f"query window {query_window!r} not pre-declared")
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceLOSODataModule

    _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
    start, end = query_window
    data_dir = Path(repo_root) / plan.DATA_DIR_RELATIVE
    _require(data_dir.is_dir(), f"missing data dir {data_dir}")
    default_window = (plan.QUERY_START, plan.QUERY_STOP_EXCLUSIVE)
    if query_window == default_window:
        module_cls = M1VersionBSourceLOSODataModule
    else:
        module_cls = _version_b_inference_window_class(M1VersionBSourceLOSODataModule)
    datamodule = module_cls(
        task="m1",
        data_dir=str(data_dir),
        source_session_names=list(plan.FOLD0_SOURCE_SESSIONS),
        heldin_session_names=list(plan.FOLD0_SOURCE_SESSIONS),
        batch_size=plan.STAGE1_BATCH_SIZE,
        window_size=plan.WINDOW_SIZE,
        calibration_n_trials=plan.SUPPORT_TRIALS,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=plan.TRIAL_LENGTH,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=start,
        heldin_query_end_trial=end,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=plan.SEED,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        afc4_arm="none",
    )
    datamodule.setup("fit")
    if query_window == default_window:
        datamodule.version_b_contract_relaxation = None
    else:
        relaxation = getattr(datamodule, "version_b_contract_relaxation", None)
        _require(isinstance(relaxation, dict), "inference window missing contract relaxation receipt")
        datamodule.version_b_contract_relaxation = relaxation
    datamodule.train_dataset = RSyn3CarrierDataset(datamodule.train_dataset, bank, arm)
    datamodule.val_heldin_dataset = RSyn3CarrierDataset(datamodule.val_heldin_dataset, bank, arm)
    return datamodule
