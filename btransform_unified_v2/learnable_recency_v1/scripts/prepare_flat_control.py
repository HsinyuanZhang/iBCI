#!/usr/bin/env python3
"""Prepare, but never execute, FULL_FLAT M1/M2/H1 training commands.

This utility writes reproducible command manifests only.  It contains no
training or scoring dispatch branch and deliberately never creates a target
results directory named by a future training command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
for candidate in (
    PKG / "src",
    PKG.parent / "src",
    PKG,
    PKG.parent.parent / "btransform_unified_v1" / "src",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from learnable_recency_v1.flat_control import TASKS, build_manifest

SCHEMA = "learnable_recency_flat_control_prepare_v1"
STATUS = "PREPARED_NOT_TRAINED"
OWN_FILES = {"manifest.json", "run_commands.sh", *(f"{task}.json" for task in TASKS)}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _is_own_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, Mapping) and data.get("prepare_schema") == SCHEMA


def _check_destination(dest: Path) -> None:
    if not dest.exists():
        return
    unknown = [path.name for path in dest.iterdir() if path.name not in OWN_FILES]
    if unknown:
        raise FileExistsError(
            f"refusing to modify non-prepare destination entries: {unknown}"
        )
    for path in dest.glob("*.json"):
        if not _is_own_json(path):
            raise FileExistsError(f"refusing to overwrite non-prepare JSON: {path}")
    shell = dest / "run_commands.sh"
    if shell.exists() and "PREPARED_NOT_TRAINED" not in shell.read_text(
        encoding="utf-8"
    ):
        raise FileExistsError(f"refusing to overwrite non-prepare shell file: {shell}")


def _prepared(task: str, python_executable: str, device: str) -> dict[str, Any]:
    control = dict(
        build_manifest(task, python_executable=python_executable, device=device)
    )
    if control.get("effective_bias") != "allzero":
        raise RuntimeError(f"{task}: FULL_FLAT manifest does not declare allzero bias")
    for field in (
        "train_argv",
        "score_argv",
        "results_dir",
        "selection_dir",
        "reference",
    ):
        if field not in control:
            raise RuntimeError(f"{task}: missing flat-control manifest field {field}")
    if not isinstance(control["train_argv"], list) or not isinstance(
        control["score_argv"], list
    ):
        raise RuntimeError(f"{task}: future argv fields must be lists")
    return {
        "prepare_schema": SCHEMA,
        "status": STATUS,
        "training_started": False,
        "task": task,
        "control": control,
    }


def _shell(tasks: list[Mapping[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# PREPARED_NOT_TRAINED: this file records future commands only.",
        "# Review and run one command manually when training is explicitly authorized.",
        "",
    ]
    for item in tasks:
        task = str(item["task"])
        control = item["control"]
        lines.append(f"# {task} FULL_FLAT train")
        lines.append(shlex.join([str(token) for token in control["train_argv"]]))
        lines.append(f"# {task} FULL_FLAT score")
        lines.append(shlex.join([str(token) for token in control["score_argv"]]))
        lines.append("")
    return "\n".join(lines)


def prepare(
    *, tasks: tuple[str, ...], dest: Path, python_executable: str, device: str
) -> dict[str, Any]:
    selected = [_prepared(task, python_executable, device) for task in tasks]
    _check_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for item in selected:
        _atomic_text(dest / f"{item['task']}.json", _json_text(item))
    manifest = {
        "prepare_schema": SCHEMA,
        "status": STATUS,
        "training_started": False,
        "tasks": selected,
    }
    _atomic_text(dest / "manifest.json", _json_text(manifest))
    _atomic_text(dest / "run_commands.sh", _shell(selected))
    return {
        "prepare_schema": SCHEMA,
        "status": STATUS,
        "training_started": False,
        "dest": str(dest),
        "files": {
            name: _sha(dest / name)
            for name in sorted(OWN_FILES & {path.name for path in dest.iterdir()})
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--dest", type=Path, default=PKG / "results/flat_control_setup")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    print(
        json.dumps(
            prepare(
                tasks=tasks,
                dest=args.dest.resolve(),
                python_executable=args.python_executable,
                device=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
