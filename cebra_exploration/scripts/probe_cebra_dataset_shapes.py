"""Probe CEBRA hippocampus preload shapes. Downloads <1 MB total. CPU only."""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import requests

ROOT = pathlib.Path("/home/xinyuan/Work_host/SPINT/cebra_exploration")
sys.path.insert(0, str(ROOT / "third_party" / "cebra"))

URLS = {
    "achilles": (
        "https://cebra.fra1.digitaloceanspaces.com/data/rat_hippocampus/achilles.jl.gz",
        "c52f9b55cbc23c66d57f3842214058b8",
        "5d7b243e07b24c387e5412cd5ff46f0b",
    ),
    "buddy": (
        "https://cebra.fra1.digitaloceanspaces.com/data/rat_hippocampus/buddy.jl.gz",
        "36341322907708c466871bf04bc133c2",
        "339290585be2188f48a176f05aaf5df6",
    ),
    "cicero": (
        "https://cebra.fra1.digitaloceanspaces.com/data/rat_hippocampus/cicero.jl.gz",
        "a83b02dbdc884fdd7e53df362499d42f",
        "f262a87d2e59f164cb404cd410015f3a",
    ),
    "gatsby": (
        "https://cebra.fra1.digitaloceanspaces.com/data/rat_hippocampus/gatsby.jl.gz",
        "2b889da48178b3155011c12555342813",
        "564e431c19e55db2286a9d64c86a94c4",
    ),
}


def main() -> None:
    import cebra.data.assets as assets

    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="cebra_hc_probe_"))
    print(f"probe_dir={out_dir}")
    proxies = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    session = requests.Session()
    session.proxies.update(proxies)

    # Patch requests.get used by CEBRA downloader to go through the proxy.
    orig_get = requests.get

    def proxied_get(*args, **kwargs):
        kwargs.setdefault("proxies", proxies)
        return orig_get(*args, **kwargs)

    requests.get = proxied_get
    try:
        for name, (url, checksum, gz_checksum) in URLS.items():
            print(f"\n=== {name} ===")
            assets.download_file_with_progress_bar(
                url=url,
                expected_checksum=checksum,
                location=str(out_dir),
                file_name=f"{name}.jl",
                gzipped_checksum=gz_checksum,
            )
            local_path = out_dir / f"{name}.jl"
            print(f"path={local_path} exists={local_path.exists()} size={local_path.stat().st_size if local_path.exists() else 'NA'}")
            import joblib

            data = joblib.load(local_path)
            print(f"keys={sorted(data.keys())}")
            for k, v in data.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape} dtype={getattr(v, 'dtype', type(v))}")
                else:
                    print(f"  {k}: type={type(v)} value={v!r}"[:200])
            spikes = data["spikes"]
            pos = data["position"]
            print(f"n_samples={spikes.shape[0]} n_units={spikes.shape[1]}")
            print(f"position_shape={pos.shape} pos_min={pos.min(axis=0)} pos_max={pos.max(axis=0)}")
            if pos.shape[1] >= 3:
                print(f"col0 unique-ish range; col1 unique={set(pos[:, 1].tolist()[:20])}...")
                print(f"col1 unique={sorted(set(pos[:, 1].round(6).tolist()))}")
                print(f"col2 unique={sorted(set(pos[:, 2].round(6).tolist()))}")
    finally:
        requests.get = orig_get


if __name__ == "__main__":
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    main()
