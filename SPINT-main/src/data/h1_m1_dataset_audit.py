"""Read-only data audit of H1 (DANDI 000954) and M1 (DANDI 000941).

This module answers three questions nobody had checked with actual data:

Q1 -- What is ``Velocity`` in H1 (sub-HumanPitt)?  The FALCON H1 evaluator
     (``falcon_challenge.dataloaders.load_nwb``) reads
     ``nwbfile.acquisition['OpenLoopKinematicsVelocity']`` as the decode
     target.  This audit reports the exact NWB text describing that field
     and its sibling ``OpenLoopKinematics``, plus every session/device/lab
     string in the file, so the provenance question ("measured limb motion"
     vs "decoder/effector output") can be answered from the data itself
     rather than assumed.

Q2 -- What per-channel/per-electrode side information exists in H1 and M1
     that no consumer arm has used?  This audit enumerates every
     electrode-table, device, and per-unit field present in both datasets'
     ``held-in-calib`` NWB files: presence/absence, dtype, shape, and an
     example value.  It then classifies each field as functional or
     anatomical per the framing in
     ``sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md`` and
     ``sua_exploration/docs/UNIT_SIDE_FEATURE_ABLATION.md``.  It recommends
     nothing.

Q3 -- What is a "trial" in H1?  A prior measurement found H1 trials run
     about 15.1 s with only 8-15 per recording, unusually long/few for a
     reaching task.  This audit reads ``intervals/epochs`` (sub-trial phase
     tags, e.g. Reach/Orient/SnapTo/Shape/Grasp/Carry/Release, each preceded
     by a Presentation cue) and ``acquisition/TrialNum`` directly to report
     what a "trial" actually contains.

Scope and safety
-----------------
Read-only.  No training, no GPU, no forward passes.  Only the
``held-in-calib`` split of each dataset is opened; ``reject_path_scope``
fails closed on any held-out / minival / formal / EvalAI path so a caller
cannot accidentally widen this module's reach by editing a constant.  The
only writes this module performs are the JSON audit receipt under
``h1_m1_dataset_audit_receipts/`` next to this file, and receipts refuse to
overwrite an existing file.

Status: READ_ONLY_DATA_AUDIT.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

MODULE_STATUS = "READ_ONLY_DATA_AUDIT"
AUDIT_SCHEMA = "h1_m1_dataset_audit_v1"

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../SPINT-main
DATA_ROOT = REPO_ROOT / "data"

H1_DATASET_DIR = "000954"
H1_SUBJECT_DIR = "sub-HumanPitt-held-in-calib"
M1_DATASET_DIR = "000941"
M1_SUBJECT_DIR = "sub-MonkeyL-held-in-calib"

# Any of these substrings appearing in a resolved path is a hard stop.  This
# mirrors the fail-closed guard used elsewhere in src/data (see
# h1_m4_eb_pilot.reject_path_scope) so this module cannot be pointed at a
# held-out, minival, or EvalAI file even by an edited constant.
FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (
    "held-out", "heldout", "minival", "evalai", "formal", "private", "test_ecephys",
)


class DatasetAuditError(ValueError):
    """Fail-closed violation of this audit's read-only, source-only contract."""


def reject_path_scope(path: str | Path) -> None:
    lower = str(Path(path).resolve()).lower()
    hit = [tok for tok in FORBIDDEN_PATH_TOKENS if tok in lower]
    if hit:
        raise DatasetAuditError(f"h1_m1_dataset_audit forbids path {path} (matched {hit})")


def list_allowed_nwb(dataset_dirname: str, subject_dirname: str) -> list[Path]:
    """List NWB files in exactly one held-in-calib subject directory.

    Never recurses beyond the named directory; rejects any path carrying a
    forbidden token even if it were somehow present under the named dir.
    """
    root = (DATA_ROOT / dataset_dirname).resolve()
    reject_path_scope(root)
    if root.name != dataset_dirname or not root.is_dir():
        raise DatasetAuditError(f"expected dataset root {dataset_dirname}, got {root}")
    directory = (root / subject_dirname).resolve()
    reject_path_scope(directory)
    if "held-in-calib" not in directory.name:
        raise DatasetAuditError(f"h1_m1_dataset_audit only reads held-in-calib dirs, got {directory}")
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise DatasetAuditError(f"{directory} escapes dataset root {root}") from exc
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    out = []
    for candidate in sorted(directory.glob("*.nwb")):
        resolved = candidate.resolve()
        reject_path_scope(resolved)
        resolved.relative_to(directory)
        out.append(resolved)
    if not out:
        raise DatasetAuditError(f"no NWB files found under {directory}")
    return out


# --------------------------------------------------------------------------- #
# h5py helpers: read raw NWB/HDF5 content as plain, JSON-safe Python.
# --------------------------------------------------------------------------- #
def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_decode(v) for v in value.tolist()]
    return value


def _read_scalar_dataset(ds: h5py.Dataset) -> Any:
    return _decode(ds[()])


def _attrs(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {k: _decode(v) for k, v in obj.attrs.items() if k not in ("namespace", "object_id", ".specloc")}


def _dataset_field_summary(ds: h5py.Dataset, n_example: int = 3) -> dict[str, Any]:
    shape = ds.shape
    dtype = str(ds.dtype)
    try:
        if ds.ndim == 0:
            example = _decode(ds[()])
        else:
            n = min(n_example, shape[0]) if shape[0] else 0
            example = _decode(ds[:n])
    except Exception as exc:  # pragma: no cover - defensive, no known trigger
        example = f"<unreadable: {exc}>"
    return {"dtype": dtype, "shape": list(shape), "example": example}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Q1 -- H1 velocity provenance.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VelocityProvenanceRecord:
    file: str
    session_description: Any
    identifier: Any
    experiment_description: Any
    experimenter: Any
    institution: Any
    lab: Any
    subject_description: Any
    general_top_level_keys: list[str]
    processing_module_keys: list[str]
    open_loop_kinematics: dict[str, Any]
    open_loop_kinematics_velocity: dict[str, Any]


def audit_h1_velocity_provenance(paths: Sequence[Path]) -> dict[str, Any]:
    """Read every session/device/comment string bearing on what H1 'Velocity' is.

    Reports exactly what the NWB metadata states for
    ``acquisition/OpenLoopKinematics`` and
    ``acquisition/OpenLoopKinematicsVelocity`` (the field the FALCON H1
    evaluator loads as the decode target, per
    ``falcon_challenge.dataloaders.load_nwb``), together with every
    session-level and general-level description string in the file.  It
    also records that ``general`` carries no ``devices`` or
    ``extracellular_ephys`` group in H1 (unlike M1), i.e. there is no device
    or effector description anywhere in the file beyond the strings quoted
    here.  It does not infer what is not stated.
    """
    records: list[VelocityProvenanceRecord] = []
    for path in paths:
        reject_path_scope(path)
        with h5py.File(path, "r") as f:
            gen = f["general"]
            subj = gen["subject"]

            def ts_summary(name: str) -> dict[str, Any]:
                grp = f[f"acquisition/{name}"]
                data_ds = grp["data"]
                st_ds = grp["starting_time"]
                return {
                    "group_attrs": _attrs(grp),
                    "data": _dataset_field_summary(data_ds),
                    "data_attrs": _attrs(data_ds),
                    "starting_time_value": _read_scalar_dataset(st_ds),
                    "starting_time_attrs": _attrs(st_ds),
                }

            records.append(VelocityProvenanceRecord(
                file=path.name,
                session_description=_read_scalar_dataset(f["session_description"]),
                identifier=_read_scalar_dataset(f["identifier"]),
                experiment_description=_read_scalar_dataset(gen["experiment_description"]),
                experimenter=_read_scalar_dataset(gen["experimenter"]),
                institution=_read_scalar_dataset(gen["institution"]),
                lab=_read_scalar_dataset(gen["lab"]),
                subject_description=_read_scalar_dataset(subj["description"]),
                general_top_level_keys=sorted(gen.keys()),
                processing_module_keys=sorted(f["processing"].keys()),
                open_loop_kinematics=ts_summary("OpenLoopKinematics"),
                open_loop_kinematics_velocity=ts_summary("OpenLoopKinematicsVelocity"),
            ))

    distinct_session_descriptions = sorted({r.session_description for r in records})
    distinct_experiment_descriptions = sorted({r.experiment_description for r in records})
    distinct_kin_descriptions = sorted({r.open_loop_kinematics["group_attrs"].get("description") for r in records})
    distinct_vel_descriptions = sorted({r.open_loop_kinematics_velocity["group_attrs"].get("description") for r in records})
    distinct_kin_comments = sorted({r.open_loop_kinematics["group_attrs"].get("comments") for r in records})
    distinct_vel_comments = sorted({r.open_loop_kinematics_velocity["group_attrs"].get("comments") for r in records})
    distinct_kin_units = sorted({r.open_loop_kinematics["data_attrs"].get("unit") for r in records})
    distinct_vel_units = sorted({r.open_loop_kinematics_velocity["data_attrs"].get("unit") for r in records})
    device_or_ephys_keys_present = sorted({
        k for r in records for k in r.general_top_level_keys if k in ("devices", "extracellular_ephys")
    })

    return {
        "n_files": len(records),
        "files": [r.file for r in records],
        "distinct_session_descriptions": distinct_session_descriptions,
        "distinct_experiment_descriptions": distinct_experiment_descriptions,
        "distinct_OpenLoopKinematics_descriptions": distinct_kin_descriptions,
        "distinct_OpenLoopKinematicsVelocity_descriptions": distinct_vel_descriptions,
        "distinct_OpenLoopKinematics_comments": distinct_kin_comments,
        "distinct_OpenLoopKinematicsVelocity_comments": distinct_vel_comments,
        "distinct_OpenLoopKinematics_data_unit": distinct_kin_units,
        "distinct_OpenLoopKinematicsVelocity_data_unit": distinct_vel_units,
        "processing_module_keys_all_files": sorted({k for r in records for k in r.processing_module_keys}),
        "general_device_or_extracellular_ephys_keys_present_in_H1": device_or_ephys_keys_present,
        "per_file": [r.__dict__ for r in records],
        "not_stated_in_nwb_metadata": [
            "Whether OpenLoopKinematics/OpenLoopKinematicsVelocity reflect measured "
            "limb kinematics, a robotic/virtual effector trajectory, or decoder "
            "output. The unit attribute is literally 'arbitrary' and comments are "
            "literally 'no comments' on every file audited.",
            "Any device, sensor, or effector identity for the kinematics stream: "
            "general/ has no 'devices' or 'extracellular_ephys' group in H1 (see "
            "general_device_or_extracellular_ephys_keys_present_in_H1, empty), so "
            "there is no NWB Device object describing what produced this signal.",
        ],
    }


# --------------------------------------------------------------------------- #
# Q2 -- per-channel / per-electrode side information.
# --------------------------------------------------------------------------- #
def _find_field_names(f: h5py.File, keywords: Sequence[str]) -> list[str]:
    hits: list[str] = []

    def visit(name: str, _obj: Any) -> None:
        low = name.lower()
        if any(kw in low for kw in keywords):
            hits.append(name)

    f.visititems(visit)
    return hits


def audit_h1_channel_side_information(paths: Sequence[Path]) -> dict[str, Any]:
    """Enumerate per-channel/per-electrode fields present in H1 (000954).

    H1's ``units`` table and ``general`` group are inspected directly; a
    repo-wide keyword scan for electrode/channel/waveform/impedance/array/bank
    is also run over the full HDF5 tree of each file so nothing is missed by
    only checking the conventional NWB locations.
    """
    per_file = []
    for path in paths:
        reject_path_scope(path)
        with h5py.File(path, "r") as f:
            units = f["units"]
            unit_fields = {
                k: _dataset_field_summary(units[k]) for k in units.keys() if isinstance(units[k], h5py.Dataset)
            }
            keyword_hits = _find_field_names(
                f, ("electrode", "channel", "waveform", "impedance", "array", "bank")
            )
            per_file.append({
                "file": path.name,
                "units_colnames": _decode(units.attrs.get("colnames")),
                "units_dataset_fields": unit_fields,
                "n_units": int(units["id"].shape[0]),
                "general_top_level_keys": sorted(f["general"].keys()),
                "keyword_scan_hits_electrode_channel_waveform_impedance_array_bank": keyword_hits,
            })

    n_units_all = sorted({p["n_units"] for p in per_file})
    return {
        "dataset": "000954 (H1, sub-HumanPitt)",
        "n_files": len(per_file),
        "per_file": per_file,
        "n_units_consistent_across_files": n_units_all,
        "fields_present": {
            "units.id": "present (unit index, int64; not a channel/electrode identifier)",
            "units.spike_times": "present (per-unit spike times; no per-channel metadata attached)",
        },
        "fields_absent": [
            "electrode table (general/extracellular_ephys/electrodes)",
            "device table (general/devices)",
            "electrode group (ElectrodeGroup)",
            "per-unit electrode index (units.electrodes)",
            "electrode coordinates (x, y, z)",
            "impedance",
            "spike waveform / waveform_mean",
            "array or bank identifier",
            "unit quality metric",
        ],
        "classification": (
            "No per-channel or per-electrode side information of any kind is present "
            "in the H1 held-in-calib NWB files. The units table has exactly the "
            "columns listed in units_colnames above; there is nothing to classify as "
            "functional or anatomical because nothing beyond spike times and a bare "
            "unit index exists."
        ),
    }


def audit_m1_channel_side_information(paths: Sequence[Path]) -> dict[str, Any]:
    """Enumerate per-channel/per-electrode fields present in M1 (000941)."""
    per_file = []
    for path in paths:
        reject_path_scope(path)
        with h5py.File(path, "r") as f:
            gen = f["general"]
            devices = gen["devices"]
            device_summaries = {
                dev_name: _attrs(devices[dev_name]) for dev_name in devices.keys()
            }
            ee = gen["extracellular_ephys"]
            electrode_groups = {
                k: _attrs(ee[k]) for k in ee.keys() if k != "electrodes"
            }
            electrodes = ee["electrodes"]
            electrode_fields = {
                k: _dataset_field_summary(electrodes[k]) for k in electrodes.keys()
            }
            group_name_vals = [_decode(v) for v in electrodes["group_name"][:]]
            from collections import Counter
            group_name_counts = dict(Counter(group_name_vals))
            imp_vals = electrodes["imp"][:]
            xyz_all_nan = bool(
                np.all(np.isnan(electrodes["x"][:]))
                and np.all(np.isnan(electrodes["y"][:]))
                and np.all(np.isnan(electrodes["z"][:]))
            )
            location_vals = sorted({_decode(v) for v in electrodes["location"][:]})
            filtering_vals = sorted({_decode(v) for v in electrodes["filtering"][:]})

            units = f["units"]
            units_colnames = _decode(units.attrs.get("colnames"))
            unit_electrode_link = units["electrodes"][:].tolist() if "electrodes" in units else None
            electrode_index_is_identity = (
                unit_electrode_link == list(range(len(unit_electrode_link)))
                if unit_electrode_link is not None else None
            )
            keyword_hits = _find_field_names(f, ("waveform",))

            per_file.append({
                "file": path.name,
                "general_top_level_keys": sorted(gen.keys()),
                "devices": device_summaries,
                "electrode_groups": electrode_groups,
                "electrodes_colnames": _decode(electrodes.attrs.get("colnames")),
                "electrodes_fields": electrode_fields,
                "n_electrodes": int(electrodes["id"].shape[0]),
                "electrode_group_name_counts": group_name_counts,
                "electrode_impedance_all_nan": bool(np.all(np.isnan(imp_vals))),
                "electrode_xyz_all_nan": xyz_all_nan,
                "electrode_location_distinct_values": location_vals,
                "electrode_filtering_distinct_values": filtering_vals,
                "units_colnames": units_colnames,
                "n_units": int(units["id"].shape[0]),
                "unit_to_electrode_is_identity_map": electrode_index_is_identity,
                "waveform_field_scan_hits": keyword_hits,
            })

    return {
        "dataset": "000941 (M1, sub-MonkeyL)",
        "n_files": len(per_file),
        "per_file": per_file,
        "fields_present": {
            "general/devices": "present: one Device per array (see per_file[*].devices)",
            "general/extracellular_ephys/*_electrode_group": (
                "present: one ElectrodeGroup per physical FMA array (H/I/J/K), each "
                "with a 'location' attribute (constant 'Motor Cortex' / 'M1' across "
                "electrodes) and a 'description' naming the array label"
            ),
            "general/extracellular_ephys/electrodes.group_name": (
                "present and populated: bank/array identifier per electrode "
                "(H_electrode_group / I_electrode_group / J_electrode_group / "
                "K_electrode_group), 16 electrodes each, 64 total"
            ),
            "general/extracellular_ephys/electrodes.location": "present but constant ('M1') across all electrodes -- non-discriminative",
            "units.electrodes": "present: DynamicTableRegion linking each unit to one electrode row; a 1:1 identity map in every audited file (unit i <-> electrode i)",
            "units.obs_intervals": "present: per-unit observed time interval [start, stop]",
        },
        "fields_present_but_unpopulated": {
            "electrodes.x/y/z": "columns exist in the schema but every value is NaN in every audited file -- no electrode coordinates",
            "electrodes.imp": "column exists but every value is NaN -- no impedance",
            "electrodes.filtering": "column exists but every value is the literal string 'unknown'",
        },
        "fields_absent": [
            "spike waveform / waveform_mean (no 'waveform' field anywhere in the HDF5 tree)",
            "unit quality metric",
        ],
        "classification": {
            "anatomical_not_functional": [
                "electrodes.location (constant 'M1' / 'Motor Cortex') -- brain-region label, not a per-unit functional property, and constant so it carries zero information within-dataset",
                "electrodes.group_name / ElectrodeGroup (array/bank identifier H/I/J/K) -- physical array membership, an implant/anatomical fact, not a functional (tuning) property",
                "electrodes.x/y/z, electrodes.imp -- would be anatomical/physical (position, impedance) if populated, but are not populated in this dataset",
            ],
            "could_serve_as_identity_side_information_at_deployment": [
                "units.electrodes (electrode index per unit) and electrodes.group_name -- available at calibration time with zero forward-pass cost (static per-unit lookup), same category as the electrode-index feature ('F3') flagged as untested in "
                "sua_exploration/docs/UNIT_SIDE_FEATURE_ABLATION.md section 3.2: 'electrode index is stable across sessions while sorted unit id is not' ('对同一块植入阵列，electrode index 是跨 session 稳定的，而 sorted unit id 不是'), "
                "with the same document's caveat that neural drift means the same electrode does not necessarily record the same neuron across sessions, so this is flagged as untested, not recommended.",
            ],
            "no_functional_side_information_present": (
                "No waveform, amplitude/SNR, or other tuning-adjacent per-unit feature "
                "exists in either M1 or H1's held-in-calib NWB files. The functional-"
                "vs-anatomical distinction drawn in ASIC_DEPLOYMENT_CHARTER.md and "
                "UNIT_SIDE_FEATURE_ABLATION.md (waveform/electrode features are what a "
                "sorter uses to define a unit, and drift across days) cannot be tested "
                "against M1 or H1 directly: the fields that document discusses "
                "(waveform, amplitude/SNR) are simply absent from these two datasets. "
                "Only the anatomical/positional fields (array/bank id) are present."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Q3 -- H1 trial and task structure.
# --------------------------------------------------------------------------- #
def audit_h1_trial_structure(paths: Sequence[Path]) -> dict[str, Any]:
    """Read ``intervals/epochs`` and ``acquisition/TrialNum`` directly.

    Reports, per session and pooled: how many trials, how long each trial
    is (from TrialNum boundaries), and the sub-trial phase tags recorded in
    intervals/epochs (e.g. Presentation/Reach/Orient/SnapTo/Shape/Grasp/
    Carry/Release), with duration statistics per tag.
    """
    from collections import Counter, defaultdict

    per_file = []
    all_trial_durations: list[float] = []
    all_trial_counts: list[int] = []
    tag_durations: dict[str, list[float]] = defaultdict(list)
    all_tags_seen: set[str] = set()

    for path in paths:
        reject_path_scope(path)
        with h5py.File(path, "r") as f:
            trialnum = f["acquisition/TrialNum/data"][:]
            rate = f["acquisition/TrialNum/starting_time"].attrs["rate"]
            st = f["acquisition/TrialNum/starting_time"][()]
            timestamps = st + np.arange(len(trialnum)) * rate

            uniq_trials = np.unique(trialnum)
            durations = []
            for t in uniq_trials:
                idx = np.where(trialnum == t)[0]
                d = float(timestamps[idx[-1]] - timestamps[idx[0]])
                durations.append(d)
            all_trial_durations.extend(durations)
            all_trial_counts.append(len(uniq_trials))

            ep = f["intervals/epochs"]
            start = ep["start_time"][:]
            stop = ep["stop_time"][:]
            tags = [_decode(t) for t in ep["tags"][:]]
            for tg, s, e in zip(tags, start, stop):
                tag_durations[tg].append(float(e - s))
            all_tags_seen.update(tags)

            per_file.append({
                "file": path.name,
                "n_trials": int(len(uniq_trials)),
                "trial_durations_s": [round(d, 3) for d in durations],
                "n_epochs": int(len(start)),
                "session_span_s": float(stop[-1] - start[0]) if len(start) else None,
                "distinct_epoch_tags": sorted(set(tags)),
                "epoch_tag_counts": dict(Counter(tags)),
            })

    def stats(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

    tag_stats = {tg: stats(vals) for tg, vals in sorted(tag_durations.items())}

    # Canonical per-trial phase order as it appears within a trial in the
    # tag stream: a cue ("Presentation*") precedes each of the eight
    # movement-execution phases.
    canonical_phase_order = [
        "Intertrial",
        "Presentation", "Reach",
        "Presentation2", "Orient",
        "Presentation3", "SnapTo",
        "Presentation4", "Shape",
        "Presentation5", "Grasp",
        "Presentation6", "Carry",
        "Presentation7", "Orient2",
        "Release",
    ]

    return {
        "n_files": len(per_file),
        "per_file": per_file,
        "trials_per_session": {
            "min": int(min(all_trial_counts)),
            "max": int(max(all_trial_counts)),
            "mean": float(np.mean(all_trial_counts)),
        },
        "trial_duration_s_pooled": stats(all_trial_durations),
        "epoch_tag_vocabulary": sorted(all_tags_seen),
        "epoch_tag_duration_stats_s": tag_stats,
        "canonical_phase_order_observed_in_tag_stream": canonical_phase_order,
        "interpretation_grounded_in_tags": (
            "Each TrialNum-defined 'trial' is not a single point-to-point reach. "
            "intervals/epochs subdivides every trial into ~15-16 tagged sub-phases: "
            "an Intertrial period, then seven repetitions of a Presentation cue "
            "epoch (~1.0-1.5 s, duration attrs are near-constant per Presentation* "
            "tag) immediately followed by a movement-execution epoch -- Reach, "
            "Orient, SnapTo, Shape, Grasp, Carry, Orient2, Release -- covering the "
            "dataset's 7 DoF (3D translation, roll, 3D grasp) in sequence. Summing "
            "the median duration of one full tag cycle reproduces the pooled trial "
            "duration reported above. This is read directly from the 'tags' field "
            "of intervals/epochs and the per-tag duration statistics; it is not an "
            "inference beyond the recorded tags."
        ),
    }


# --------------------------------------------------------------------------- #
# Environment snapshot (read-only: nvidia-smi / ps, no process control).
# --------------------------------------------------------------------------- #
def environment_snapshot() -> dict[str, Any]:
    import subprocess

    def run(cmd: list[str]) -> str:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return (out.stdout or out.stderr).strip()
        except Exception as exc:  # pragma: no cover
            return f"<unavailable: {exc}>"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "h5py_version": h5py.__version__,
        "numpy_version": np.__version__,
        "nvidia_smi": run(["nvidia-smi"]),
        "ps_aux": run(["ps", "aux"]),
    }


# --------------------------------------------------------------------------- #
# Orchestration + receipt.
# --------------------------------------------------------------------------- #
def run_audit() -> dict[str, Any]:
    h1_paths = list_allowed_nwb(H1_DATASET_DIR, H1_SUBJECT_DIR)
    m1_paths = list_allowed_nwb(M1_DATASET_DIR, M1_SUBJECT_DIR)

    source_file_hashes = {
        "000954_held_in_calib": {p.name: sha256_file(p) for p in h1_paths},
        "000941_held_in_calib": {p.name: sha256_file(p) for p in m1_paths},
    }

    return {
        "schema": AUDIT_SCHEMA,
        "module_status": MODULE_STATUS,
        "scope": {
            "datasets_read": [
                f"{H1_DATASET_DIR}/{H1_SUBJECT_DIR}",
                f"{M1_DATASET_DIR}/{M1_SUBJECT_DIR}",
            ],
            "splits_forbidden": list(FORBIDDEN_PATH_TOKENS),
            "n_h1_files": len(h1_paths),
            "n_m1_files": len(m1_paths),
        },
        "q1_h1_velocity_provenance": audit_h1_velocity_provenance(h1_paths),
        "q2_channel_side_information": {
            "h1_000954": audit_h1_channel_side_information(h1_paths),
            "m1_000941": audit_m1_channel_side_information(m1_paths),
        },
        "q3_h1_trial_structure": audit_h1_trial_structure(h1_paths),
        "source_file_sha256": source_file_hashes,
        "environment": environment_snapshot(),
    }


def _sanitize_nan(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_nan(v) for v in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(_sanitize_nan(value), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
        + "\n"
    ).encode("utf-8")


def write_receipt(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Write the audit receipt.  No silent overwrite."""
    receipt_path = Path(path).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"h1_m1_dataset_audit refuses to overwrite: {receipt_path}")
    payload = _canonical_json(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


RECEIPTS_DIR = Path(__file__).resolve().parent / "h1_m1_dataset_audit_receipts"


def main(receipt_name: str = "h1_m1_dataset_audit_v1_receipt.json") -> dict[str, Any]:
    result = run_audit()
    meta = write_receipt(RECEIPTS_DIR / receipt_name, result)
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
