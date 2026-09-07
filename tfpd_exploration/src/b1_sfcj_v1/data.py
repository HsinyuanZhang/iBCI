"""NWB loading, position-law binning, rate view, LODO manifest, Stage 0A inventory."""
from __future__ import annotations

import datetime as dt
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from pynwb import NWBHDF5IO

from .constants import (
    BOX_MS,
    CALIB_TRIAL_COUNTS,
    DATA_ROOT,
    EVALAI_OVERVIEW_URL,
    EXPECTED_FALCON_SHA256,
    FALCON_README_URL,
    FOLDS,
    FS_NEURAL,
    HELD_IN_DATES,
    HELD_OUT_DATES,
    MINIVAL_TRIAL_COUNTS,
    N_CHANNELS,
    N_FREQ,
    N_MS_BINS,
    N_NEURAL_SAMPLES,
    N_SPEC_FRAMES,
    N_VALID,
    NEURAL_DT,
    RAW_SPECTROGRAM_MIN,
    RESULTS_ROOT,
    SAMPLES_PER_MS,
    SPEC_DT,
    SPEC_T0,
    VALID_END,
    VALID_START,
)
from .metric import valid_frame_mask_freq_major
from .util import sha256_array, sha256_bytes, sha256_file, write_json


class DataContractError(AssertionError):
    pass


@dataclass(frozen=True)
class Trial:
    date: str
    split: str
    trial_index: int
    path: str
    tx: np.ndarray
    timestamps: np.ndarray
    spectrogram: np.ndarray
    spectrogram_times: np.ndarray
    spectrogram_frequencies: np.ndarray
    eval_mask: np.ndarray
    start_time: float
    stop_time: float


@dataclass
class FileRecord:
    path: Path
    split: str
    date: str
    n_trials: int
    size: int
    sha256: str
    trials: list[Trial] = field(default_factory=list)


def spec_frame_to_bin(frame: int) -> int:
    return int(np.floor(SPEC_T0 * 1000.0 + frame))


def alignment_table() -> list[dict]:
    keys = (0, 1, VALID_START - 1, VALID_START, VALID_END - 1, N_SPEC_FRAMES - 1)
    rows = []
    for k in keys:
        t_s = SPEC_T0 + k * SPEC_DT
        rows.append(
            {
                "frame": int(k),
                "spectrogram_time_s": float(t_s),
                "bin": spec_frame_to_bin(k),
                "formula": "floor(10.24 + k)",
            }
        )
    return rows


def discover_nwb_files(root: Path = DATA_ROOT) -> list[Path]:
    files = sorted(root.rglob("*.nwb"))
    if len(files) != 9:
        raise DataContractError(f"expected 9 NWB files, found {len(files)} under {root}")
    return files


def parse_split_date(path: Path) -> tuple[str, str]:
    name = path.name
    parent = path.parent.name
    if "held-in-calib" in parent:
        split = "held-in-calib"
    elif "held-in-minival" in parent:
        split = "held-in-minival"
    elif "held-out-calib" in parent:
        split = "held-out-calib"
    else:
        raise DataContractError(f"unrecognized split in {path}")
    if "ses-20210626" in name:
        date = "20210626"
    elif "ses-20210627" in name:
        date = "20210627"
    elif "ses-20210628" in name:
        date = "20210628"
    elif "ses-20210630" in name:
        date = "20210630"
    elif "ses-20210701" in name:
        date = "20210701"
    elif "ses-20210705" in name:
        date = "20210705"
    else:
        raise DataContractError(f"unrecognized date in {path}")
    return split, date


@lru_cache(maxsize=16)
def load_file(path_str: str) -> FileRecord:
    path = Path(path_str)
    split, date = parse_split_date(path)
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        tx_all = np.asarray(nwb.get_acquisition("tx").data, dtype=np.float64)
        ts_all = np.asarray(nwb.get_acquisition("tx").timestamps, dtype=np.float64)
        table = nwb.trials.to_dataframe().reset_index()
    n_trials = len(table)
    if tx_all.shape != (N_NEURAL_SAMPLES * n_trials, N_CHANNELS):
        raise DataContractError(f"{path}: tx shape {tx_all.shape} != ({N_NEURAL_SAMPLES * n_trials}, {N_CHANNELS})")
    trials = []
    for i, row in table.iterrows():
        sl = slice(i * N_NEURAL_SAMPLES, (i + 1) * N_NEURAL_SAMPLES)
        spec = np.asarray(row["spectrogram_values"], dtype=np.float64)
        mask = np.asarray(row["spectrogram_eval_mask"]).astype(bool)
        times = np.asarray(row["spectrogram_times"], dtype=np.float64)
        freqs = np.asarray(row["spectrogram_frequencies"], dtype=np.float64)
        trials.append(
            Trial(
                date=date,
                split=split,
                trial_index=int(i),
                path=str(path),
                tx=tx_all[sl],
                timestamps=ts_all[sl],
                spectrogram=spec,
                spectrogram_times=times,
                spectrogram_frequencies=freqs,
                eval_mask=mask,
                start_time=float(row["start_time"]),
                stop_time=float(row["stop_time"]),
            )
        )
    return FileRecord(
        path=path,
        split=split,
        date=date,
        n_trials=n_trials,
        size=int(path.stat().st_size),
        sha256=sha256_file(path),
        trials=trials,
    )


def load_all_files() -> dict[tuple[str, str], FileRecord]:
    out = {}
    for path in discover_nwb_files():
        rec = load_file(str(path))
        out[(rec.split, rec.date)] = rec
    return out


def bin_tx_counts(tx: np.ndarray) -> np.ndarray:
    """sample i -> bin i//30. tx [27000, 85] -> counts [900, 85]."""
    tx = np.asarray(tx)
    if tx.shape[0] != N_NEURAL_SAMPLES:
        raise DataContractError(f"expected {N_NEURAL_SAMPLES} samples, got {tx.shape}")
    return tx.reshape(N_MS_BINS, SAMPLES_PER_MS, tx.shape[1]).sum(axis=1)


def box_smooth_5ms(counts: np.ndarray) -> np.ndarray:
    kernel = np.ones(BOX_MS, dtype=np.float64) / float(BOX_MS)
    counts = np.asarray(counts, dtype=np.float64)
    out = np.empty_like(counts)
    for ch in range(counts.shape[1]):
        out[:, ch] = np.convolve(counts[:, ch], kernel, mode="same")
    return out


def rate_view(tx: np.ndarray) -> np.ndarray:
    return box_smooth_5ms(bin_tx_counts(tx))


def predict_layout_to_counts(neural: np.ndarray) -> np.ndarray:
    """decoder.predict layout [1, 85, 27000] -> [900, 85] counts. No timestamps."""
    arr = np.asarray(neural)
    if arr.ndim != 3 or arr.shape[-2:] != (N_CHANNELS, N_NEURAL_SAMPLES) and arr.shape[1:] != (N_CHANNELS, N_NEURAL_SAMPLES):
        if arr.shape == (1, N_CHANNELS, N_NEURAL_SAMPLES):
            pass
        else:
            raise DataContractError(f"predict neural must be [1,85,27000], got {arr.shape}")
    tx = np.transpose(arr[0], (1, 0))
    return bin_tx_counts(tx)


def calib_trials(date: str) -> list[Trial]:
    rec = load_all_files()[("held-in-calib" if date in HELD_IN_DATES else "held-out-calib", date)]
    return list(rec.trials)


def minival_trials(date: str) -> list[Trial]:
    if date not in MINIVAL_TRIAL_COUNTS:
        raise DataContractError(f"no minival for {date}")
    return list(load_all_files()[("held-in-minival", date)].trials)


def first_m3(date: str) -> list[Trial]:
    trials = calib_trials(date)
    if len(trials) < 3:
        raise DataContractError(f"{date} has {len(trials)} calib trials, need 3")
    return trials[:3]


def next_m3(date: str) -> list[Trial]:
    trials = calib_trials(date)
    if len(trials) < 6:
        raise DataContractError(f"{date} has {len(trials)} calib trials, need 6 for next M3")
    return trials[3:6]


def query_trials(date: str) -> list[Trial]:
    """Pre-registered stream: calib 4..N then minival 1..2."""
    calib = calib_trials(date)
    rest = list(calib[3:])
    if date in MINIVAL_TRIAL_COUNTS:
        rest.extend(minival_trials(date))
    return rest


def lodo_manifest() -> dict:
    rows = []
    for fold in FOLDS:
        val = fold["val_date"]
        queries = query_trials(val)
        in_range = [i < fold["n_in_range"] for i in range(len(queries))]
        rows.append(
            {
                "fold": fold["fold"],
                "train_dates": list(fold["train_dates"]),
                "val_date": val,
                "k_train_max": fold["k_train_max"],
                "n_query": len(queries),
                "n_in_range": int(sum(in_range)),
                "n_ood": int(len(queries) - sum(in_range)),
                "query_order": [
                    {
                        "stream_index": i,
                        "split": q.split,
                        "trial_index": q.trial_index,
                        "in_range": bool(in_range[i]),
                        "pre_decode_k": 3 + i,
                        "ood": bool(not in_range[i]),
                    }
                    for i, q in enumerate(queries)
                ],
            }
        )
    return {
        "calib_to_minival_convention": "calib 4..N -> minival 1..2",
        "cross_file_chronology": "unrecoverable; timestamps restart at 0 per file",
        "k_train_max": {str(f["fold"]): f["k_train_max"] for f in FOLDS},
        "in_range_mask": {str(f["fold"]): f["n_in_range"] for f in FOLDS},
        "folds": rows,
    }


def _fetch_url(url: str, timeout: int = 30) -> dict:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
        return {
            "url": url,
            "ok": True,
            "fetched_at_utc": stamp,
            "sha256": sha256_bytes(body),
            "n_bytes": len(body),
            "text": body.decode("utf-8", errors="replace"),
        }
    except Exception as exc:  # noqa: BLE001 — inventory must record unreachability
        return {
            "url": url,
            "ok": False,
            "fetched_at_utc": stamp,
            "error": f"{type(exc).__name__}: {exc}",
            "sha256": None,
            "n_bytes": 0,
            "text": None,
        }


def _assert(cond: bool, message: str, failures: list[str]) -> None:
    if not cond:
        failures.append(message)


def run_stage0a(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    files = discover_nwb_files()
    records = []
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from falcon_challenge.evaluator import DATASET_HELDINOUT_MAP
    import falcon_challenge

    falcon_dir = Path(falcon_challenge.__file__).resolve().parent
    falcon_sha = {name: sha256_file(falcon_dir / name) for name in EXPECTED_FALCON_SHA256}
    for name, expected in EXPECTED_FALCON_SHA256.items():
        _assert(falcon_sha[name] == expected, f"falcon {name} SHA {falcon_sha[name]} != {expected}", failures)

    cfg = FalconConfig(FalconTask.b1)
    dotted_map = DATASET_HELDINOUT_MAP["b1"]
    _assert(
        dotted_map["held_in"] == ["2021.06.26", "2021.06.27", "2021.06.28"],
        f"DATASET_HELDINOUT_MAP b1 held_in unexpected: {dotted_map['held_in']}",
        failures,
    )

    counts = {"held-in-calib": {}, "held-in-minival": {}, "held-out-calib": {}}
    tx_value_set = set()
    raw_minima = []
    hash_tags = {}

    for path in files:
        rec = load_file(str(path))
        split, date = rec.split, rec.date
        counts[split][date] = rec.n_trials
        neural, spec_t, trial_change, eval_mask = load_nwb(path, dataset=FalconTask.b1)
        tag = cfg.hash_dataset(path)
        hash_tags[str(path)] = tag
        _assert(tag == date, f"hash_dataset({path.name})={tag!r} != undotted {date}", failures)
        _assert(neural.shape == (N_NEURAL_SAMPLES * rec.n_trials, N_CHANNELS), f"{path.name} neural shape {neural.shape}", failures)
        _assert(neural.dtype == np.float64 or np.asarray(neural).dtype == neural.dtype, f"{path.name} dtype {neural.dtype}", failures)
        uniq = np.unique(rec.trials[0].tx)
        tx_value_set.update(float(x) for x in uniq)
        _assert(set(np.unique(neural).tolist()).issubset({0.0, 1.0}), f"{path.name} tx not in {{0,1}}", failures)
        tc_idx = np.where(np.asarray(trial_change))[0]
        expected_tc = np.array([26999 + 27000 * k for k in range(rec.n_trials)])
        _assert(
            np.array_equal(tc_idx, expected_tc),
            f"{path.name} trial_change {tc_idx.tolist()} != {expected_tc.tolist()}",
            failures,
        )
        with NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            ts_file = np.asarray(nwb.get_acquisition("tx").timestamps, dtype=np.float64)
        dt = np.diff(ts_file)
        _assert(np.allclose(dt, NEURAL_DT, rtol=0, atol=1e-12), f"{path.name} timestamp step not 1/30000", failures)
        _assert(ts_file[0] == 0.0, f"{path.name} timestamp[0]={ts_file[0]} != 0", failures)

        for trial in rec.trials:
            _assert(trial.tx.shape == (N_NEURAL_SAMPLES, N_CHANNELS), f"{path.name} trial tx {trial.tx.shape}", failures)
            _assert(trial.spectrogram.shape == (N_FREQ, N_SPEC_FRAMES), f"{path.name} spec {trial.spectrogram.shape}", failures)
            _assert(trial.eval_mask.shape == (N_FREQ, N_SPEC_FRAMES), f"{path.name} mask {trial.eval_mask.shape}", failures)
            _assert(float(trial.spectrogram_times[0]) == SPEC_T0, f"{path.name} spectrogram_times[0]={trial.spectrogram_times[0]}", failures)
            _assert(
                np.allclose(np.diff(trial.spectrogram_times), SPEC_DT, rtol=0, atol=1e-12),
                f"{path.name} spectrogram step",
                failures,
            )
            tmask = trial.eval_mask[0]
            true_idx = np.where(tmask)[0]
            _assert(true_idx[0] == VALID_START and true_idx[-1] == VALID_END - 1 and len(true_idx) == N_VALID, f"{path.name} mask not [90,790)", failures)
            _assert(all(np.array_equal(trial.eval_mask[f], tmask) for f in range(N_FREQ)), f"{path.name} mask not shared across frequencies", failures)
            _assert(float(trial.spectrogram.min()) == RAW_SPECTROGRAM_MIN, f"{path.name} raw min {trial.spectrogram.min()} != 1.0", failures)
            raw_minima.append(float(trial.spectrogram.min()))
            counts_bins = bin_tx_counts(trial.tx)
            _assert(counts_bins.shape == (N_MS_BINS, N_CHANNELS), "bin shape", failures)
            _assert(np.array_equal(counts_bins.sum(axis=0), trial.tx.sum(axis=0)), f"{path.name} spike conservation failed", failures)
            # each 1 ms bin has exactly 30 samples by construction of i//30; prove with timestamps
            _assert(trial.tx.shape[0] == N_NEURAL_SAMPLES, "trial samples", failures)
            for b in range(N_MS_BINS):
                sl = slice(b * SAMPLES_PER_MS, (b + 1) * SAMPLES_PER_MS)
                _assert(sl.stop - sl.start == SAMPLES_PER_MS, "bin width", failures)
            # position law: sample i -> bin i//30, proved from timestamps
            sample_index = np.arange(N_NEURAL_SAMPLES)
            local_ms = (trial.timestamps - trial.start_time) * 1000.0
            bins_from_ts = np.floor(local_ms + 1e-9).astype(int)
            _assert(np.array_equal(bins_from_ts, sample_index // SAMPLES_PER_MS), f"{path.name} timestamp bin law", failures)
            _assert(np.all(sample_index // SAMPLES_PER_MS == np.repeat(np.arange(N_MS_BINS), SAMPLES_PER_MS)), "i//30 law", failures)
            _assert(all(int((bins_from_ts == b).sum()) == SAMPLES_PER_MS for b in range(N_MS_BINS)), f"{path.name} bin occupancy != 30", failures)

        records.append(
            {
                "path": str(path),
                "repo_relative": str(path.relative_to(DATA_ROOT.parents[2])),
                "split": split,
                "session": date,
                "n_trials": rec.n_trials,
                "size_bytes": rec.size,
                "sha256": rec.sha256,
                "tx_shape": [N_NEURAL_SAMPLES * rec.n_trials, N_CHANNELS],
                "tx_dtype": str(neural.dtype),
                "tx_value_set": [0.0, 1.0],
                "timestamp_start": float(ts_file[0]),
                "timestamp_step": float(NEURAL_DT),
                "timestamp_end": float(ts_file[-1]),
                "trial_change_indices": expected_tc.tolist(),
                "spectrogram_shape_per_trial": [N_FREQ, N_SPEC_FRAMES],
                "spectrogram_times0": SPEC_T0,
                "spectrogram_step_s": SPEC_DT,
                "mask": "[90,790)",
                "raw_spectrogram_min": RAW_SPECTROGRAM_MIN,
                "log_contract": "log(raw); raw already includes +offset so min==1.0",
                "hash_dataset": tag,
                "n_channels": N_CHANNELS,
                "n_freq": N_FREQ,
                "mask_true_per_trial": N_VALID * N_FREQ,
            }
        )

    expected_counts = {
        "held-in-calib": dict(CALIB_TRIAL_COUNTS),
        "held-in-minival": dict(MINIVAL_TRIAL_COUNTS),
        "held-out-calib": {d: 3 for d in HELD_OUT_DATES},
    }
    # held-in-calib counts only the three held-in dates
    expected_counts["held-in-calib"] = {d: CALIB_TRIAL_COUNTS[d] for d in HELD_IN_DATES}
    for split, mapping in expected_counts.items():
        for date, n in mapping.items():
            got = counts.get(split, {}).get(date)
            _assert(got == n, f"{split}/{date} trials {got} != {n}", failures)

    _assert(tx_value_set <= {0.0, 1.0}, f"tx value set {tx_value_set}", failures)
    _assert(all(m == 1.0 for m in raw_minima), "raw spectrogram min not identically 1.0", failures)

    # calib vs minival restart
    chrono_note = (
        "held-in-calib and held-in-minival each restart timestamps at 0.0; "
        "cross-file chronology is unrecoverable. Fixed convention: calib 4..N -> minival 1..2."
    )
    for date in HELD_IN_DATES:
        c0 = load_all_files()[("held-in-calib", date)].trials[0].timestamps[0]
        m0 = load_all_files()[("held-in-minival", date)].trials[0].timestamps[0]
        _assert(c0 == 0.0 and m0 == 0.0, f"{date} calib/minival timestamps do not both start at 0", failures)

    readme = _fetch_url(FALCON_README_URL)
    evalai = _fetch_url(EVALAI_OVERVIEW_URL)
    snap_dir = out_dir / "rulesnapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    if readme["ok"]:
        (snap_dir / "falcon_readme.md").write_text(readme["text"], encoding="utf-8")
    if evalai["ok"] and evalai["text"] is not None:
        (snap_dir / "evalai_overview.html").write_text(evalai["text"], encoding="utf-8")

    evaluate_mismatch = {
        "hash_dataset_local": "undotted 20210626",
        "DATASET_HELDINOUT_MAP_b1": dotted_map,
        "installed_evaluate_mismatch": (
            "FalconConfig.hash_dataset returns undotted 20210626 while "
            "DATASET_HELDINOUT_MAP['b1'] uses dotted 2021.06.26; installed evaluate() "
            "session lookup fails on local files."
        ),
    }

    inventory = {
        "n_files": 9,
        "total_bytes": int(sum(r["size_bytes"] for r in records)),
        "files": records,
        "trial_counts": counts,
        "alignment": {
            "samples_per_trial": N_NEURAL_SAMPLES,
            "ms_bins": N_MS_BINS,
            "samples_per_ms": SAMPLES_PER_MS,
            "position_law": "sample i -> bin i//30",
            "frame_k_to_bin": "floor(10.24 + k)",
            "table": alignment_table(),
        },
        "mask": {"start": VALID_START, "end": VALID_END, "n_valid": N_VALID, "shared_by_all_frequencies": True},
        "raw_spectrogram_min": RAW_SPECTROGRAM_MIN,
        "log_transform": "log(raw) with no extra log1p",
        "trial_change_law": "26999 + 27000 k",
        "calib_minival_chronology": chrono_note,
        "hash_dataset_tags": hash_tags,
        "evaluate_mismatch": evaluate_mismatch,
        "falcon_challenge_sha256": falcon_sha,
        "falcon_sha_match_design_s18": {k: falcon_sha[k] == EXPECTED_FALCON_SHA256[k] for k in falcon_sha},
        "rule_snapshot": {
            "falcon_readme": {k: v for k, v in readme.items() if k != "text"},
            "evalai_overview": {k: v for k, v in evalai.items() if k != "text"},
        },
        "lodo_manifest": lodo_manifest(),
        "assertions_failed": failures,
        "gate_pass": len(failures) == 0,
        "input_sha256s": {r["repo_relative"]: r["sha256"] for r in records},
    }
    write_json(out_dir / "stage0a_inventory.json", inventory)
    lines = [
        "# Stage 0A receipt",
        "",
        f"gate_pass: {inventory['gate_pass']}",
        f"n_files: 9",
        f"total_bytes: {inventory['total_bytes']}",
        "",
        "## Assertions",
    ]
    if failures:
        lines.extend(f"- FAIL: {f}" for f in failures)
    else:
        lines.append("- all listed Stage 0A assertions passed")
    lines.extend(
        [
            "",
            "## hash_dataset vs map",
            evaluate_mismatch["installed_evaluate_mismatch"],
            "",
            "## chronology",
            chrono_note,
            "",
            "## falcon_challenge SHA",
        ]
    )
    for name, digest in falcon_sha.items():
        lines.append(f"- {name}: {digest} match={digest == EXPECTED_FALCON_SHA256[name]}")
    lines.extend(
        [
            "",
            "## rule snapshot",
            f"- README ok={readme['ok']} sha={readme.get('sha256')} at {readme.get('fetched_at_utc')}",
            f"- EvalAI ok={evalai['ok']} sha={evalai.get('sha256')} at {evalai.get('fetched_at_utc')}"
            + ("" if evalai["ok"] else f" unreachable: {evalai.get('error')}"),
        ]
    )
    (out_dir / "stage0a_receipt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise DataContractError("Stage 0A assertions failed:\n" + "\n".join(failures))
    return inventory
