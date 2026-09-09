"""Wait for the temporal controller, then audit all cells and render the figure."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import OUT, ROOT, atomic_json, sha

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    result = OUT / "finalization_pipeline.json"
    if result.exists():
        raise FileExistsError("existing finalization pipeline: inspect its status and process first")
    program_path = OUT / "program.json"
    program = read(program_path)
    if program.get("status") != "RUNNING" or not Path(f"/proc/{program['controller_pid']}/cmdline").is_file():
        raise RuntimeError("this waiting pipeline must attach to a live temporal controller")
    controller_pid = program["controller_pid"]
    expected_command = Path(f"/proc/{controller_pid}/cmdline").read_bytes()
    if str(HERE / "run_program.py").encode() not in expected_command and b"scripts/chronological_last2_v1/run_program.py" not in expected_command:
        raise RuntimeError("unexpected controller command")
    files = [HERE / name for name in ("protocol.py", "collect_results.py", "m1_audit.py", "h1_audit.py", "plot_results.py", "finalize_when_ready.py")]
    bindings = {str(path): sha(path) for path in files}
    state = {"schema": "chronological_last2_finalization_pipeline_v1", "status": "WAITING_FOR_NINE_CELLS",
             "pid": os.getpid(), "controller_pid": controller_pid, "created_at": now(),
             "protocol_sha256": sha(OUT / "protocol.json"), "implementation_sha256": bindings,
             "paper_integration": "root review, visual inspection and manuscript update follow successful artifact generation"}

    def save():
        state["updated_at"] = now()
        atomic_json(result, state)

    def verify_code():
        if sha(OUT / "protocol.json") != state["protocol_sha256"] or any(sha(Path(raw)) != digest for raw, digest in bindings.items()):
            raise RuntimeError("finalization implementation/protocol changed after attaching")

    save()
    try:
        previous_counts = None
        while True:
            program = read(program_path)
            if program.get("controller_pid") != controller_pid or program.get("protocol_sha256") != state["protocol_sha256"]:
                raise RuntimeError("controller identity or protocol changed")
            if program.get("status") not in ("RUNNING", "COMPLETED"):
                raise RuntimeError(f"temporal controller stopped without completion: {program.get('status')}")
            command_path = Path(f"/proc/{controller_pid}/cmdline")
            live = command_path.is_file()
            if live and command_path.read_bytes() != expected_command:
                raise RuntimeError("controller PID now belongs to another command")
            counts = program.get("counts", {})
            if counts != previous_counts:
                state["observed_counts"] = counts
                save()
                print(json.dumps({"event": "waiting", "counts": counts}), flush=True)
                previous_counts = counts
            if program["status"] == "COMPLETED" and not live:
                if len(program.get("cells", [])) != 9 or any(c.get("status") != "COMPLETED" for c in program["cells"]):
                    raise RuntimeError("completed controller lacks nine completed cells")
                break
            if not live:
                raise RuntimeError("temporal controller disappeared before writing completion")
            time.sleep(5)
        verify_code()
        state.update(status="AUDITING_AND_COLLECTING", completed_program_sha256=sha(program_path))
        save()
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", PYTHONWARNINGS="ignore", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", MPLBACKEND="Agg")
        with (OUT / "logs/final_collect.log").open("x") as stream:
            subprocess.run([sys.executable, "-u", str(HERE / "collect_results.py"), "--dest", str(OUT)], cwd=ROOT.parent, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if read(OUT / "summary.json").get("status") != "PASSED" or read(OUT / "collection_receipt.json").get("status") != "PASSED":
            raise RuntimeError("full collection did not pass")
        verify_code()
        state.update(status="RENDERING_FIGURE", summary_sha256=sha(OUT / "summary.json"), collection_receipt_sha256=sha(OUT / "collection_receipt.json"))
        save()
        with (OUT / "logs/final_plot.log").open("x") as stream:
            subprocess.run([sys.executable, "-u", str(HERE / "plot_results.py"), "--summary", str(OUT / "summary.json"), "--dest", str(OUT / "figures")], cwd=ROOT.parent, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        receipt_path = OUT / "figures/chronological_last2_figure_receipt.json"
        receipt = read(receipt_path)
        if receipt.get("status") != "PASSED" or receipt["input_summary"]["sha256"] != state["summary_sha256"]:
            raise RuntimeError("figure receipt is not bound to the audited summary")
        for name, digest in receipt["outputs"].items():
            if sha(receipt_path.parent / name) != digest:
                raise RuntimeError("rendered figure/CSV checksum mismatch")
        state.update(status="COMPLETED", completed_at=now(), figure_receipt_sha256=sha(receipt_path))
        save()
        print("All nine temporal cells audited; figure and CSV ready for root visual review and manuscript integration.", flush=True)
    except BaseException as error:
        state.update(status="FAILED", failed_at=now(), error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
