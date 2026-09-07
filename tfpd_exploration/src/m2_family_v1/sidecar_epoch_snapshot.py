"""Non-invasive snapshotter for an already-running formal M2 pair."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]/"tfpd_exploration/results/m2/family_v1/paired_source_only_v1"
PID=1702813
def snapshot_once() -> list[str]:
    made=[]
    for arm in ("FLAT","ROUTE"):
        live=ROOT/arm/"epoch_metrics.json"
        if not live.is_file(): continue
        record=json.loads(live.read_text())
        epoch=int(record["epoch"]); destination=ROOT/arm/f"epoch_{epoch:03d}_metrics.json"
        if destination.exists(): continue
        payload={"arm":arm,"epoch":epoch,"raw_equal_session_r2":float(record["raw_equal_session_r2"]),"ema_equal_session_r2":float(record["ema_equal_session_r2"]),"elapsed_seconds_pair_start_cumulative":float(record["elapsed_seconds"]),"warning":"elapsed_seconds is pair-start cumulative, not arm-only epoch duration.","snapshot_role":"sidecar_immutable_epoch_snapshot","source":"epoch_metrics.json"}
        body=(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode(); destination.write_bytes(body); destination.with_suffix(destination.suffix+".sha256").write_text(hashlib.sha256(body).hexdigest()+"  "+destination.name+"\n"); made.append(str(destination))
    return made
if __name__=="__main__":
    # Bounded polling on the exact known process, never name-based pgrep.
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline and os.path.exists(f"/proc/{PID}"):
        created=snapshot_once()
        if created: print(json.dumps({"created":created},sort_keys=True),flush=True)
        time.sleep(15)
    print(json.dumps({"done":True,"pid_alive":os.path.exists(f"/proc/{PID}")},sort_keys=True),flush=True)
