"""Root-run audit of M10 carrier bin clock; never reads post-M10 EMG values."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "btransform_unified_v2/scripts/cross_session_v1")]
from falcon_challenge import dataloaders
from m1_data import FOLDS, nwb_path
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as legacy


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    results = {}
    for session in FOLDS:
        path = nwb_path(session)
        old = legacy.load_support_bins(path, emg_trial_stop=10, neural_trial_stop=10)
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            trials = nwb.trials.to_dataframe().iloc[:10]
            series = nwb.acquisition["preprocessed_emg"]
            ts = np.asarray(series.get_timeseries(next(iter(series.time_series))).timestamps[:])
            end = float(trials.iloc[-1].stop_time)
            prefix_stop = int(np.searchsorted(ts, end, side="right"))
            prefix_ts = ts[:prefix_stop]
            ids = np.asarray(nwb.units.id[:])
            spikes = [np.asarray(nwb.units.get_unit_spike_times(i, in_interval=(float(ts[0] - .02), end))) for i in range(len(ids))]
            frame = pd.DataFrame({"spike_times": spikes}, index=ids)
            online = dataloaders.bin_units(frame, bin_size_s=.02, bin_timestamps=prefix_ts)
            indices = []
            for _, row in trials.iterrows():
                onset, stop = legacy._onset_stop(row)
                ix = np.arange(np.searchsorted(ts, onset, side="left"), np.searchsorted(ts, stop, side="right"))
                indices.extend(ix.tolist())
            ix = np.asarray(indices, dtype=np.int64)
            aligned = online[ix].astype(np.float64)
            old_counts = np.rint(old.rates * .02)
            assert aligned.shape == old_counts.shape and aligned.shape[1] == 64
            assert np.array_equal(ids, frame.index.to_numpy())
            # Interior rows exclude support clipping and noncontiguous clock edges.
            interior = (ix + 1 < len(online)) & (ix > 0)
            safe = ix[interior]
            gaps = ts[safe + 1] - ts[safe]
            interior[np.flatnonzero(interior)[~np.isclose(gaps, .02, rtol=0, atol=1e-8)]] = False
            interior &= (ts[ix] >= legacy._onset_stop(trials.iloc[0])[0]) & (ts[ix] + .02 < end)
            shifted = online[ix[interior] + 1].astype(np.float64)
            legacy_interior = old_counts[interior]
            results[session] = {
                "path": str(path), "path_sha256": old.path_sha256,
                "support_trials": [0, 10], "post_M10_EMG_values_read": False,
                "unit_ids": ids.tolist(), "unit_order_sha256": sha(ids),
                "unit_table_and_online_order_equal": True, "shape": list(aligned.shape),
                "legacy_counts_sha256": sha(old_counts), "official_end_counts_sha256": sha(aligned),
                "same_timestamp_unequal_cell_fraction": float(np.mean(old_counts != aligned)),
                "same_timestamp_absolute_count_difference_mean": float(np.mean(np.abs(old_counts - aligned))),
                "interior_bins": int(interior.sum()),
                "legacy_vs_official_next_bin_unequal_cell_fraction": float(np.mean(legacy_interior != shifted)),
                "global_timestamp_indices_sha256": sha(ix),
                "legacy_definition": "[timestamp, timestamp + 0.02), with global movement-support clipping",
                "online_definition": "official bin_units(..., is_timestamp_bin_start=False)",
            }
            assert results[session]["same_timestamp_unequal_cell_fraction"] > 0
        print(json.dumps({"session": session, **{k: v for k, v in results[session].items() if "fraction" in k or k == "interior_bins"}}), flush=True)
    out = ROOT / "btransform_unified_v2/results/m1_carrier_refinement_v1/bin_clock_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "m1_carrier_refinement_clock_audit_v1", "status": "PASSED", "created_unix": time.time(), "sessions": results,
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(legacy.__file__), Path(dataloaders.__file__))},
               "interpretation": "Observed phase difference is a computational discrepancy; no claim that correcting it improves held-out decoding."}
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
