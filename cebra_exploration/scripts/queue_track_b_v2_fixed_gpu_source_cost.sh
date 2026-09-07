#!/usr/bin/env bash
# Wait-only launcher for the reviewed Track-B fixed-GPU engineering cost fit.
#
# Default invocation is a no-write/no-data dry plan.  A live queue requires
# both explicit flags and a review-time SHA of this exact script in
# TRACK_B_QUEUE_EXPECTED_SHA256.  It never kills a process or claims GPU 1
# while a compute process is present.
set -u
set -o pipefail

readonly REPO_ROOT="/home/xinyuan/Work_host/SPINT"
readonly QUEUE_SCRIPT="$REPO_ROOT/cebra_exploration/scripts/queue_track_b_v2_fixed_gpu_source_cost.sh"
readonly PYTHON_BIN="/home/xinyuan/miniconda3/envs/spint/bin/python"
readonly FIXED_CORE="$REPO_ROOT/cebra_exploration/src/track_b_v2_fixed_gpu_engineering.py"
readonly PREFLIGHT="$REPO_ROOT/cebra_exploration/scripts/preflight_track_b_v2_fixed_gpu_engineering.py"
readonly RUNNER="$REPO_ROOT/cebra_exploration/scripts/run_track_b_v2_fixed_gpu_source_cost.py"
readonly CEBRA_PROVENANCE="$REPO_ROOT/cebra_exploration/third_party/CEBRA_PROVENANCE.txt"
readonly MASK_RECEIPT_CANONICAL="$REPO_ROOT/sua_exploration/results/carrier_value_mask_v1/terminal_mask_aggregate.json"
readonly OUTPUT_CANONICAL="$REPO_ROOT/cebra_exploration/results/track_b_v2_fixed_gpu_cost_sua_firstfold_d8it250_s42_gpu1_v1/receipt.json"
readonly PHYSICAL_GPU_INDEX="1"
readonly POLL_SECONDS_CANONICAL="1800"

readonly FIXED_CORE_SHA256="6a195e195807430c67d84f353bbdc8cca5bef536874cf510d296c3fecab00fed"
readonly PREFLIGHT_SHA256="93207f17db04738482d6f07eb3ab29288d7aaccf687d050cb42d3c3f8b2aeaba"
readonly RUNNER_SHA256="460969623d7e8389110bce19b3d0d38690830fde377bf082c906953f55424823"
readonly CEBRA_PROVENANCE_SHA256="d3619c956fb57ccf698b49185d558f20423b29ebf4251b516b5289e2e3ebf6d9"
readonly FIXED_RUNTIME_CLOSURE_SHA256="15ec4f8b3a1e9ea46da9e377d86969c7f73a42bf16932fbcd21d1f54d6bf746d"

execute=0
authorised=0
for arg in "$@"; do
    case "$arg" in
        --execute) execute=1 ;;
        --i-have-authorization) authorised=1 ;;
        *) printf 'ERROR: unsupported queue argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

if [[ "$execute" -ne "$authorised" ]]; then
    printf '%s\n' 'ERROR: live queue requires both --execute and --i-have-authorization' >&2
    exit 2
fi

if [[ "$execute" -eq 0 ]]; then
    printf '%s\n' \
        'DRY_PLAN_ONLY__NO_WAIT__NO_DATA__NO_GPU__NO_WRITE' \
        "mask_receipt=$MASK_RECEIPT_CANONICAL" \
        "physical_gpu_index=$PHYSICAL_GPU_INDEX" \
        "poll_seconds=$POLL_SECONDS_CANONICAL" \
        "canonical_output=$OUTPUT_CANONICAL" \
        'live_requires=--execute --i-have-authorization and TRACK_B_QUEUE_EXPECTED_SHA256'
    exit 0
fi

test_mode="${TRACK_B_QUEUE_TEST_MODE:-0}"
if [[ "$test_mode" != "0" && "$test_mode" != "1" ]]; then
    printf '%s\n' 'ERROR: TRACK_B_QUEUE_TEST_MODE must be 0 or 1' >&2
    exit 2
fi
if [[ "$test_mode" == "0" ]]; then
    for forbidden in TRACK_B_QUEUE_MASK_RECEIPT TRACK_B_QUEUE_NVIDIA_SMI \
                     TRACK_B_QUEUE_POLL_SECONDS TRACK_B_QUEUE_MAX_POLLS \
                     TRACK_B_QUEUE_TEST_NO_EXEC; do
        if [[ -n "${!forbidden:-}" ]]; then
            printf 'ERROR: %s is forbidden outside test mode\n' "$forbidden" >&2
            exit 2
        fi
    done
fi

mask_receipt="$MASK_RECEIPT_CANONICAL"
nvidia_smi_bin="/usr/bin/nvidia-smi"
poll_seconds="$POLL_SECONDS_CANONICAL"
max_polls="0"
test_no_exec="0"
if [[ "$test_mode" == "1" ]]; then
    mask_receipt="${TRACK_B_QUEUE_MASK_RECEIPT:-$mask_receipt}"
    nvidia_smi_bin="${TRACK_B_QUEUE_NVIDIA_SMI:-$nvidia_smi_bin}"
    poll_seconds="${TRACK_B_QUEUE_POLL_SECONDS:-$poll_seconds}"
    max_polls="${TRACK_B_QUEUE_MAX_POLLS:-$max_polls}"
    test_no_exec="${TRACK_B_QUEUE_TEST_NO_EXEC:-$test_no_exec}"
fi

is_decimal() { [[ "$1" =~ ^[0-9]+$ ]]; }
if ! is_decimal "$poll_seconds" || ! is_decimal "$max_polls"; then
    printf '%s\n' 'ERROR: poll interval and max polls must be decimal integers' >&2
    exit 2
fi
if [[ "$test_mode" == "0" && "$poll_seconds" != "$POLL_SECONDS_CANONICAL" ]]; then
    printf '%s\n' 'ERROR: production poll interval drift' >&2
    exit 2
fi

sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_sha() {
    local path="$1"
    local expected="$2"
    [[ -f "$path" && ! -L "$path" ]] || {
        printf 'ERROR: reviewed implementation file invalid: %s\n' "$path" >&2
        exit 1
    }
    local actual
    actual="$(sha_file "$path")" || exit 1
    [[ "$actual" == "$expected" ]] || {
        printf 'ERROR: reviewed implementation SHA drift: %s expected=%s actual=%s\n' \
            "$path" "$expected" "$actual" >&2
        exit 1
    }
}

validate_live_implementation() {
    local expected_self="${TRACK_B_QUEUE_EXPECTED_SHA256:-}"
    local executing_script
    executing_script="$(readlink -f -- "${BASH_SOURCE[0]}")" || exit 1
    [[ "$executing_script" == "$QUEUE_SCRIPT" && ! -L "${BASH_SOURCE[0]}" ]] || {
        printf 'ERROR: queue must execute the canonical regular script: %s\n' "$QUEUE_SCRIPT" >&2
        exit 1
    }
    [[ "$expected_self" =~ ^[0-9a-f]{64}$ ]] || {
        printf '%s\n' 'ERROR: live queue requires a 64-hex TRACK_B_QUEUE_EXPECTED_SHA256 review anchor' >&2
        exit 2
    }
    require_sha "$QUEUE_SCRIPT" "$expected_self"
    require_sha "$FIXED_CORE" "$FIXED_CORE_SHA256"
    require_sha "$PREFLIGHT" "$PREFLIGHT_SHA256"
    require_sha "$RUNNER" "$RUNNER_SHA256"
    require_sha "$CEBRA_PROVENANCE" "$CEBRA_PROVENANCE_SHA256"

    local preflight_json closure_sha
    preflight_json="$(
        env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
            PYTHONPATH="$REPO_ROOT/cebra_exploration/src:$REPO_ROOT/cebra_exploration/third_party/cebra" \
            "$PYTHON_BIN" "$PREFLIGHT" --expected-cuda-visible-device "$PHYSICAL_GPU_INDEX"
    )" || {
        printf '%s\n' 'ERROR: fixed no-data implementation preflight failed' >&2
        exit 1
    }
    closure_sha="$(printf '%s' "$preflight_json" | env PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
        'import json,sys; print(json.load(sys.stdin)["vendored_cuda_static_audit"]["closure_sha256"])')" || exit 1
    [[ "$closure_sha" == "$FIXED_RUNTIME_CLOSURE_SHA256" ]] || {
        printf 'ERROR: fixed runtime closure drift expected=%s actual=%s\n' \
            "$FIXED_RUNTIME_CLOSURE_SHA256" "$closure_sha" >&2
        exit 1
    }
}

output_pair_is_fresh() {
    [[ ! -e "$OUTPUT_CANONICAL" && ! -L "$OUTPUT_CANONICAL" &&
       ! -e "$OUTPUT_CANONICAL.sha256" && ! -L "$OUTPUT_CANONICAL.sha256" ]]
}

validate_mask_receipt() {
    env PYTHONNOUSERSITE=1 "$PYTHON_BIN" - "$mask_receipt" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

body = Path(sys.argv[1]).absolute()
side = body.with_name(body.name + ".sha256")

def read_verified(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if stat.S_IMODE(info.st_mode) != 0o444:
            raise ValueError(f"{label} mode is not 0444")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)

try:
    body_bytes = read_verified(body, "mask receipt body")
    side_bytes = read_verified(side, "mask receipt sidecar")
    digest = hashlib.sha256(body_bytes).hexdigest()
    expected_side = f"{digest}  {body.name}\n".encode("ascii")
    if side_bytes != expected_side:
        raise ValueError("mask receipt standard sidecar/SHA mismatch")
    payload = json.loads(body_bytes)
    if not isinstance(payload, dict):
        raise ValueError("mask receipt body is not a JSON object")
    if payload.get("receipt_kind") != "carrier_value_mask_terminal_aggregate":
        raise ValueError("mask terminal receipt_kind mismatch")
    if payload.get("formal_subc_test_nwb_opened") is not False:
        raise ValueError("mask terminal formal flag is not exactly false")
    if payload.get("screen_id") != "carrier_value_mask_v1":
        raise ValueError("mask terminal screen_id mismatch")
    if type(payload.get("seed")) is not int or payload["seed"] != 42:
        raise ValueError("mask terminal seed mismatch")
    if type(payload.get("parameter_delta")) is not int or payload["parameter_delta"] != 0:
        raise ValueError("mask terminal parameter_delta mismatch")
    if type(payload.get("positive_forward_mask_signal")) is not bool:
        raise ValueError("mask terminal signal must be exactly boolean")
    allowed_verdicts = {
        "VALUE_WEIGHTED_LOW_GAIN_MASK_POSITIVE__DESIGN_SEPARATE_TRAINED_ROUTE",
        "VALUE_WEIGHTED_LOW_GAIN_MASK_FLAT_OR_NONSPECIFIC__ADVANCE_QUEUE",
    }
    if payload.get("verdict") not in allowed_verdicts:
        raise ValueError("mask terminal verdict mismatch")
    routing = payload.get("routing_checks")
    expected_checks = {
        "external_low_t4_absolute_delta_positive",
        "external_low_mask_carrier_interaction_positive",
        "external_low_t4_beats_random_same_count",
        "external_low_t4_beats_high_gain_mask",
        "within_low_t4_delta_at_least_minus_0p03",
    }
    if (not isinstance(routing, dict) or set(routing) != expected_checks or
            any(type(value) is not bool for value in routing.values())):
        raise ValueError("mask terminal routing_checks schema mismatch")
    domains = payload.get("domains")
    if (not isinstance(domains, dict) or
            not {"within_subject", "external_subject_M"}.issubset(domains)):
        raise ValueError("mask terminal domains schema mismatch")
except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
    print(f"WAIT_MASK_INVALID: {exc}", file=sys.stderr)
    raise SystemExit(75)
print(digest)
PY
}

gpu_is_idle() {
    local processes
    processes="$("$nvidia_smi_bin" -i "$PHYSICAL_GPU_INDEX" \
        --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" || return 1
    [[ -z "${processes//[[:space:]]/}" ]]
}

validate_live_implementation
if ! output_pair_is_fresh; then
    printf '%s\n' 'ERROR: canonical GPU engineering output body/sidecar is not fresh' >&2
    exit 1
fi

poll_count=0
while true; do
    poll_count=$((poll_count + 1))
    mask_sha=""
    mask_ready=0
    if mask_sha="$(validate_mask_receipt)"; then
        mask_ready=1
    fi
    gpu_ready=0
    if gpu_is_idle; then
        gpu_ready=1
    fi

    if [[ "$mask_ready" -eq 1 && "$gpu_ready" -eq 1 ]]; then
        # Revalidate every decisive condition immediately before exec.
        validate_live_implementation
        if ! output_pair_is_fresh; then
            printf '%s\n' 'ERROR: canonical GPU engineering output appeared while queue waited' >&2
            exit 1
        fi
        final_mask_sha="$(validate_mask_receipt)" || {
            printf '%s\n' 'WAIT: mask receipt changed or became invalid before launch' >&2
            mask_ready=0
        }
        if [[ "$mask_ready" -eq 1 && "$final_mask_sha" != "$mask_sha" ]]; then
            printf '%s\n' 'WAIT: mask receipt SHA changed between validations' >&2
            mask_ready=0
        fi
        if [[ "$mask_ready" -eq 1 ]] && ! gpu_is_idle; then
            printf '%s\n' 'WAIT: physical GPU 1 became occupied before launch' >&2
            gpu_ready=0
        fi
        if [[ "$mask_ready" -eq 1 && "$gpu_ready" -eq 1 ]]; then
            if [[ "$test_mode" == "1" && "$test_no_exec" == "1" ]]; then
                printf 'READY_TEST_ONLY__NO_EXEC mask_sha256=%s polls=%s\n' "$mask_sha" "$poll_count"
                exit 0
            fi
            printf 'READY: exec reviewed fixed runner mask_sha256=%s polls=%s\n' "$mask_sha" "$poll_count" >&2
            cd "$REPO_ROOT" || exit 1
            exec env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_INDEX" \
                PYTHONPATH="$REPO_ROOT/cebra_exploration/src:$REPO_ROOT/cebra_exploration/third_party/cebra" \
                "$PYTHON_BIN" "$RUNNER" \
                --expected-cuda-visible-device "$PHYSICAL_GPU_INDEX" \
                --output "$OUTPUT_CANONICAL" \
                --execute --i-have-authorization
        fi
    fi

    if [[ "$mask_ready" -ne 1 ]]; then
        printf 'WAIT poll=%s reason=mask_missing_or_invalid next_check_seconds=%s\n' \
            "$poll_count" "$poll_seconds" >&2
    elif [[ "$gpu_ready" -ne 1 ]]; then
        printf 'WAIT poll=%s reason=physical_gpu1_compute_process_present_or_query_failed next_check_seconds=%s\n' \
            "$poll_count" "$poll_seconds" >&2
    fi
    if [[ "$max_polls" -gt 0 && "$poll_count" -ge "$max_polls" ]]; then
        exit 75
    fi
    sleep "$poll_seconds"
    validate_live_implementation
    if ! output_pair_is_fresh; then
        printf '%s\n' 'ERROR: canonical GPU engineering output appeared while queue waited' >&2
        exit 1
    fi
done
