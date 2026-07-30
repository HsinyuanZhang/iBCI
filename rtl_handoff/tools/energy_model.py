#!/usr/bin/env python3
"""Analytic (pre-PDK) energy / efficiency estimator for the B3+decoder engine.

This is a TRANSPARENT bottom-up model, not a substitute for post-synthesis or
post-layout power. It combines:

    E_total = E_mac + E_sram + E_leak

  E_mac   = mac_count * e_mac_pj
  E_sram  = reads  * e_sram_rd_pj + writes * e_sram_wr_pj
  E_leak  = leak_pw_mw * active_time_ms      (per phase)

and reports SESSION energy and FRAME energy separately, then a duty-cycle
weighted average power at a target frame rate (default 50 Hz).

All per-op energy coefficients are PLACEHOLDERS you must overwrite from the
target PDK / SRAM-compiler datasheet or published silicon before quoting mW.
Edit COEFFS below or pass a --coeffs JSON. Nothing here should be reported as a
credible chip power number until coefficients come from a real macro.

Usage
-----
  python energy_model.py --preset b3_m2
  python energy_model.py --preset decoder_m2_baseline
  python energy_model.py --preset spint_dim256_cached   # real-time frame path
  python energy_model.py --preset spint_dim256_full     # session (ID+decode)
  python energy_model.py --preset b3_m2 --coeffs my_pdk.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

# --- PLACEHOLDER coefficients (OVERWRITE from PDK/SRAM datasheet) ------------
COEFFS = {
    "e_mac_pj": 0.20,        # per INT8x INT8 MAC (logic)
    "e_sram_rd_pj": 1.0,     # per SRAM read access (per logical word)
    "e_sram_wr_pj": 1.2,     # per SRAM write access (per logical word)
    "leak_mw": 0.5,          # total array+SRAM leakage while powered
    "freq_mhz": 100.0,       # array clock
    "pe": 64,                # PE count (utilisation-independent ideal cycles)
}


@dataclass
class Workload:
    name: str
    mac: float          # MAC operations for this phase
    sram_reads: float   # logical read accesses
    sram_writes: float  # logical write accesses
    phase: str          # "session" or "frame"


# --- Analytic MAC counts (from hardware_pe_sram docs) ------------------------
# B3 M2 encoder:  M*N*(T*D) + N*(2*D^2 + D*W)
def b3_mac(M=33, N=96, T=100, D=64, W=50):
    return M * N * (T * D) + N * (2 * D * D + D * W)


PRESETS = {
    # session-rate encoder
    "b3_m2": Workload("B3 M2 encoder", b3_mac(), sram_reads=b3_mac() / 8,
                      sram_writes=96 * 64, phase="session"),
    # frame-rate decoder (cached-query baseline, from 00_SYSTEM_OVERVIEW)
    "decoder_m2_baseline": Workload("Decoder M2 cached-query", 82.87e6,
                                    sram_reads=82.87e6 / 8, sram_writes=96 * 512,
                                    phase="frame"),
    # frame-rate decoder (static-query compiled candidate, from 02_PE_ARRAY)
    "decoder_m2_compiled": Workload("Decoder M2 compiled", 63.90e6,
                                    sram_reads=63.90e6 / 8, sram_writes=96 * 512,
                                    phase="frame"),
    # --- Full SPINT Dim256/TopK64 A8W8 target (see 05_SPINT_DIM256_SPEC.md) ---
    # Full mode = ID generation + one decode; dominated by fc_id_in (94.66%),
    # but this is a low-frequency session/calibration-update task.
    "spint_dim256_full": Workload("SPINT Dim256 full (ID+decode)", 524_322_816,
                                  sram_reads=524_322_816 / 8,
                                  sram_writes=96 * 50 + 96 * 256, phase="session"),
    # Cached mode = M1-M4 per 50-bin decode window, ID_MAP reused. This tiny
    # 14.17M-MAC path is what drives the real-time frame deadline.
    "spint_dim256_cached": Workload("SPINT Dim256 cached decode", 14_174_208,
                                    sram_reads=14_174_208 / 8,
                                    sram_writes=96 * 256, phase="frame"),
}


def estimate(w: Workload, c: dict, frame_hz: float) -> dict:
    e_mac = w.mac * c["e_mac_pj"]
    e_sram = w.sram_reads * c["e_sram_rd_pj"] + w.sram_writes * c["e_sram_wr_pj"]
    ideal_cycles = w.mac / c["pe"]
    active_ms = ideal_cycles / (c["freq_mhz"] * 1e3)  # cycles / (MHz*1e3) = ms
    e_leak = c["leak_mw"] * active_ms * 1e6            # mW*ms = 1e-6 J = 1e6 pJ
    e_total_pj = e_mac + e_sram + e_leak
    out = {
        "workload": w.name,
        "phase": w.phase,
        "mac": w.mac,
        "ideal_cycles": ideal_cycles,
        "active_ms": active_ms,
        "E_mac_nJ": e_mac / 1e3,
        "E_sram_nJ": e_sram / 1e3,
        "E_leak_nJ": e_leak / 1e3,
        "E_total_nJ": e_total_pj / 1e3,
    }
    if w.phase == "frame":
        # average power if this frame runs frame_hz times per second
        out["avg_power_mW_at_rate"] = (e_total_pj * 1e-12) * frame_hz * 1e3
        out["deadline_margin_ms_at_20ms"] = 20.0 - out["active_ms"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="b3_m2")
    ap.add_argument("--coeffs", type=str, default=None, help="JSON overriding COEFFS")
    ap.add_argument("--frame-hz", type=float, default=50.0)
    args = ap.parse_args()

    c = dict(COEFFS)
    if args.coeffs:
        c.update(json.loads(open(args.coeffs).read()))

    res = estimate(PRESETS[args.preset], c, args.frame_hz)
    print(json.dumps(res, indent=2))
    print("\nNOTE: coefficients are PLACEHOLDERS. Replace with PDK/SRAM-macro "
          "numbers before reporting any mW/energy as credible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
