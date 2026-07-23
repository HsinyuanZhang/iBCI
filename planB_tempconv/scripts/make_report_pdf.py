#!/usr/bin/env python3
"""Generate a multi-page PDF report summarizing all TCN scaling experiments.

Pages:
  1. Title + results table
  2. Pareto: R² vs log(params) scatter
  3. Paired delta bar chart (all configs vs baseline)
  4. Per-session grouped bar (plain vs SE)
  5. Budget curve (plain vs SE vs chnosqz vs grouped_se)
  6. SE gain decomposition (deconfound)
"""
import csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "outputs" / "results"
OUT = HERE / "outputs" / "figures" / "tcn_scaling_report.pdf"

# ---- load data ----
def load_csv(fname):
    return list(csv.DictReader(open(RES / fname)))

c1 = load_csv("c1_capacity_probe.csv")
c2 = load_csv("c2_dilated_probe.csv")
c3 = load_csv("c3_film_probe.csv")
c4 = load_csv("c4_se_glu_probe.csv")
c5 = load_csv("c5_grouped_se_deconfound.csv")

def sat_rows(rows, config_col="config", config_val=None, r2_col="ref_raw_r2",
             budget_col="budget_frames", seed_col="seed", session_col="session"):
    out = [r for r in rows if int(r[budget_col]) == 10**9]
    if config_val is not None and config_col in out[0]:
        out = [r for r in out if r[config_col] == config_val]
    return out

def stats(rows, r2_col="ref_raw_r2"):
    """Returns per-seed session-mean sat R2 array."""
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(float(r[r2_col]))
    arr = np.array([np.mean(by_seed[s]) for s in sorted(by_seed)])
    return arr

# Compute summary stats for each config
# c1: hidden 32/64/128
c1_32 = stats(sat_rows(c1, "hidden", "32"))
c1_64 = stats(sat_rows(c1, "hidden", "64"))
c1_128 = stats(sat_rows(c1, "hidden", "128"))
# c2: d1, d3_plain, d3_dil, d3_res
c2_d1 = stats(sat_rows(c2, "config", "d1"))
c2_d3p = stats(sat_rows(c2, "config", "d3_plain"))
c2_d3d = stats(sat_rows(c2, "config", "d3_dil"))
c2_d3r = stats(sat_rows(c2, "config", "d3_res"))
# c3: plain, film
c3_plain = stats(sat_rows(c3, "config", "plain"))
c3_film = stats(sat_rows(c3, "config", "film"))
# c4: plain, se, glu
c4_plain = stats(sat_rows(c4, "config", "plain"))
c4_se = stats(sat_rows(c4, "config", "se"))
c4_glu = stats(sat_rows(c4, "config", "glu"))
# c5: plain, se, glu, grouped_se, chnosqz
c5_plain = stats(sat_rows(c5, "config", "plain"))
c5_se = stats(sat_rows(c5, "config", "se"))
c5_glu = stats(sat_rows(c5, "config", "glu"))
c5_gse = stats(sat_rows(c5, "config", "grouped_se"))
c5_chs = stats(sat_rows(c5, "config", "chnosqz"))

# params
P = {"plain": 4130, "hidden64": 7298, "hidden128": 13634,
     "d3": 6050, "film": 16642, "se": 6542, "glu": 7234,
     "grouped_se": 7757, "chnosqz": 6542, "ridge7": 1538, "ridge50": 9794,
     "spint": 4569288}

# ---- plot style ----
plt.rcParams.update({"font.size": 9, "axes.titlesize": 13, "axes.labelsize": 11,
                     "figure.dpi": 150})
C_BASE = "#888888"
C_BAD = "#d62728"
C_GOOD = "#2ca02c"
C_SE = "#1f77b4"
C_NEUTRAL = "#ff7f0e"

with PdfPages(OUT) as pdf:

    # ===== PAGE 1: Title + table =====
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.suptitle("TCN Student Scaling — Full Experiment Summary",
                 fontsize=18, fontweight="bold", y=0.96)
    ax.text(0.5, 0.90, "Plan-B tempconv: closing the gap to SPINT (0.20 -> 0.26)\n"
            "All student numbers: 6 held-out sessions, 3 seeds, ref_raw saturated R²",
            ha="center", va="top", fontsize=11, color="#444", transform=ax.transAxes)

    table_data = [
        ["Model / Config", "Params", "sat R²", "paired Δ", "verdict"],
        ["[anchor] LOO oracle", "—", "0.268", "—", "ceiling"],
        ["[anchor] SPINT teacher (cross-attn)", "4,569,288", "0.26*", "—", "target (single seed)"],
        ["[anchor] ridge W50", "9,794", "0.116", "—", "linear, matched window"],
        ["[baseline] plain TCN (causal pad)", "4,130", "0.168±0.032", "—", "seeded baseline"],
        ["", "", "", "", ""],
        ["c1: hidden=64", "7,298", "0.152", "-0.016", "width: NO"],
        ["c1: hidden=128", "13,634", "0.137", "-0.031", "width: NO"],
        ["c2: d3_plain (no residual)", "6,050", "0.034", "-0.135", "depth: underfit collapse"],
        ["c2: d3_dil [1,2,4] (no residual)", "6,050", "0.008", "-0.160", "depth: underfit collapse"],
        ["c2: d3_res [1,2,4]+residual", "6,050", "0.153", "-0.016", "RF 9->57: NO"],
        ["c3: FiLM-from-CORAL", "16,642", "0.133", "-0.035", "session FiLM: redundant"],
        ["c4: GLU (hidden-level gate)", "7,234", "0.136", "-0.032", "gate too downstream"],
        ["c5: chnosqz (channel gate, no squeeze)", "6,542", "0.165", "+0.002", "deconfound: squeeze matters"],
        ["c5: grouped_se (per-output gate)", "7,757", "0.179", "+0.016", "per-output: NO"],
        ["c4/5: SETCN (shared SE, r=8)", "6,542", "0.196", "+0.033±0.007", "BEST STUDENT (4.7 sigma)"],
    ]
    tbl = ax.table(cellText=table_data, loc="center", cellLoc="left",
                   colWidths=[0.35, 0.13, 0.12, 0.16, 0.20])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    # color the SE row
    for j in range(5):
        tbl[len(table_data)-1, j].set_facecolor("#c8e6c9")
    for j in range(5):
        tbl[0, j].set_facecolor("#e0e0e0")
    pdf.savefig(fig); plt.close()

    # ===== PAGE 2: Pareto scatter =====
    fig, ax = plt.subplots(figsize=(10, 7))
    # anchors
    ax.scatter([P["spint"]], [0.26], marker="*", s=400, c=C_BAD, zorder=5, edgecolors="k")
    ax.annotate("SPINT teacher\n(4.57M, 0.26)", (P["spint"], 0.26), textcoords="offset points",
                xytext=(-80, 15), fontsize=9, fontweight="bold", color=C_BAD)
    ax.axhline(0.268, color="#ccc", ls="--", lw=1)
    ax.text(0.5, 0.272, "LOO oracle 0.268", fontsize=8, color="#999", transform=ax.get_yaxis_transform())
    ax.scatter([P["ridge50"]], [0.116], marker="s", s=80, c=C_BASE, zorder=4)
    ax.annotate("ridge W50 (9.8k, 0.116)", (P["ridge50"], 0.116), textcoords="offset points",
                xytext=(8, -3), fontsize=8, color=C_BASE)
    # students
    pts = [
        (P["plain"], c1_32.mean(), "plain TCN (4.1k)", C_BASE, "o", 100),
        (P["hidden64"], c1_64.mean(), "h64", C_BAD, "x", 60),
        (P["hidden128"], c1_128.mean(), "h128", C_BAD, "x", 60),
        (P["d3"], c2_d3r.mean(), "d3_res (6.0k)", C_BASE, "^", 80),
        (P["d3"], c2_d3p.mean(), "d3_plain", C_BAD, "v", 50),
        (P["d3"], c2_d3d.mean(), "d3_dil", C_BAD, "v", 50),
        (P["film"], c3_film.mean(), "FiLM (16.6k)", C_BAD, "x", 60),
        (P["glu"], c5_glu.mean(), "GLU (7.2k)", C_BAD, "x", 60),
        (P["grouped_se"], c5_gse.mean(), "grouped SE", C_NEUTRAL, "D", 70),
        (P["chnosqz"], c5_chs.mean(), "chnosqz", C_BASE, "o", 70),
        (P["se"], c5_se.mean(), "SETCN (6.5k)", C_GOOD, "o", 200),
    ]
    for x, y, label, c, m, s in pts:
        ax.scatter([x], [y], c=c, marker=m, s=s, zorder=5, edgecolors="k", linewidths=0.5)
        off = (8, 8) if "SETCN" in label else (6, -8)
        if "d3" in label or "h6" in label or "h1" in label:
            off = (8, -12)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=off, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel(r"Held-out saturated $R^2$ (ref_raw)")
    ax.set_title("Pareto Frontier: $R^2$ vs Parameters")
    ax.set_ylim(-0.02, 0.30)
    ax.set_xlim(800, 1e7)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#999", lw=0.5)
    pdf.savefig(fig); plt.close()

    # ===== PAGE 3: Paired delta bar chart =====
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = [
        ("h=64", c1_64 - c1_32, P["hidden64"]),
        ("h=128", c1_128 - c1_32, P["hidden128"]),
        ("d3_plain", c2_d3p - c2_d1, P["d3"]),
        ("d3_dil", c2_d3d - c2_d1, P["d3"]),
        ("d3_res", c2_d3r - c2_d1, P["d3"]),
        ("FiLM", c3_film - c3_plain, P["film"]),
        ("GLU", c5_glu - c5_plain, P["glu"]),
        ("chnosqz", c5_chs - c5_plain, P["chnosqz"]),
        ("grouped_se", c5_gse - c5_plain, P["grouped_se"]),
        ("SETCN (SE)", c5_se - c5_plain, P["se"]),
    ]
    labels = [c[0] for c in configs]
    means = [np.mean(c[1]) for c in configs]
    stds = [np.std(c[1]) for c in configs]
    colors = [C_GOOD if m > 0.01 else (C_BAD if m < -0.01 else C_BASE) for m in means]
    colors[-1] = C_GOOD  # highlight SE
    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=colors,
                  edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"Paired $\Delta R^2$ vs plain TCN baseline")
    ax.set_title(r"All Configs: Paired $\Delta$ vs Baseline (error bars = paired std)")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.003,
                f"{m:+.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylim(-0.18, 0.06)
    pdf.savefig(fig); plt.close()

    # ===== PAGE 4: Per-session plain vs SE =====
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # use c5 data for both plain and se per session per seed
    c5_sat = [r for r in c5 if int(r["budget_frames"]) == 10**9]
    sessions_order = ["ses-2020-10-30-Run1", "ses-2020-10-30-Run2",
                      "ses-2020-11-18-Run1", "ses-2020-11-19-Run1",
                      "ses-2020-11-24-Run1", "ses-2020-11-24-Run2"]
    short_names = ["10-30 R1", "10-30 R2", "11-18 R1", "11-19 R1", "11-24 R1", "11-24 R2"]
    plain_ps, se_ps, gse_ps = [], [], []
    for sk in sessions_order:
        pv = [float(r["ref_raw_r2"]) for r in c5_sat if r["session"] == sk and r["config"] == "plain"]
        sv = [float(r["ref_raw_r2"]) for r in c5_sat if r["session"] == sk and r["config"] == "se"]
        gv = [float(r["ref_raw_r2"]) for r in c5_sat if r["session"] == sk and r["config"] == "grouped_se"]
        plain_ps.append(np.mean(pv)); se_ps.append(np.mean(sv)); gse_ps.append(np.mean(gv))
    x = np.arange(len(short_names))
    w = 0.25
    ax.bar(x - w, plain_ps, w, label="plain TCN", color=C_BASE, edgecolor="k", lw=0.5)
    ax.bar(x, se_ps, w, label="SETCN (shared SE)", color=C_GOOD, edgecolor="k", lw=0.5)
    ax.bar(x + w, gse_ps, w, label="grouped SE", color=C_NEUTRAL, edgecolor="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=20)
    ax.set_ylabel(r"Saturated $R^2$")
    ax.set_title("Per-Session Breakdown: plain vs SE vs grouped-SE")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="k", lw=0.5)
    pdf.savefig(fig); plt.close()

    # ===== PAGE 5: Budget curve =====
    fig, ax = plt.subplots(figsize=(10, 5.5))
    budgets_plot = [200, 400, 800, 1600, 3200, 10**9]
    blabels = ["200", "400", "800", "1600", "3200", "sat"]
    for cfg, color, label in [("plain", C_BASE, "plain TCN"), ("se", C_GOOD, "SETCN"),
                               ("chnosqz", "#9467bd", "chnosqz"), ("grouped_se", C_NEUTRAL, "grouped SE")]:
        means = []
        for b in budgets_plot:
            vals = [float(r["ref_raw_r2"]) for r in c5
                    if int(r["budget_frames"]) == b and r["config"] == cfg]
            means.append(np.mean(vals))
        ax.plot(range(len(budgets_plot)), means, "o-", color=color, label=label, lw=2, ms=6)
    ax.set_xticks(range(len(budgets_plot))); ax.set_xticklabels(blabels)
    ax.set_xlabel("Calibration budget (frames)")
    ax.set_ylabel(r"Mean $R^2$ (6 sessions x 3 seeds)")
    ax.set_title("Few-Shot Budget Curve")
    ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close()

    # ===== PAGE 6: SE gain decomposition =====
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # Left: deconfound waterfall
    ax = axes[0]
    vals = [c5_plain.mean(), c5_chs.mean(), c5_se.mean()]
    labels_w = ["plain TCN\n(baseline)", "channel gate\nNO squeeze", "SETCN\n(channel gate\n+ squeeze)"]
    colors_w = [C_BASE, "#9467bd", C_GOOD]
    bars = ax.bar(range(3), vals, color=colors_w, edgecolor="k", lw=0.5, width=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_w, fontsize=8)
    ax.set_ylabel(r"Saturated $R^2$")
    ax.set_title("SE Deconfound: location vs squeeze")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    # annotate deltas
    ax.annotate("", xy=(1, vals[1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=2))
    ax.text(0.5, (vals[0]+vals[1])/2 + 0.005, f"location only\n+{vals[1]-vals[0]:.3f}",
            ha="center", fontsize=8, color=C_BAD)
    ax.annotate("", xy=(2, vals[2]), xytext=(1, vals[1]),
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=2))
    ax.text(1.5, (vals[1]+vals[2])/2 + 0.005, f"+squeeze\n+{vals[2]-vals[1]:.3f}",
            ha="center", fontsize=8, color=C_GOOD)
    ax.set_ylim(0.14, 0.22)

    # Right: summary text box
    ax2 = axes[1]; ax2.axis("off")
    summary_text = (
        "KEY FINDINGS\n\n"
        "1. SE is the only positive lever: +0.033 ± 0.007 (4.7σ)\n"
        "   3/3 seeds positive, 5/6 sessions positive\n\n"
        "2. Deconfound: the temporal squeeze (global\n"
        "   window context) accounts for ~0.031 of the\n"
        "   +0.033 gain. Channel-axis location: ~0.002.\n\n"
        "3. Grouped SE (per-output gates) does NOT help.\n"
        "   At n_out=2, shared gate captures routing.\n\n"
        "4. Static capacity (width c1, depth c2) and\n"
        "   session-level FiLM (c3) all failed.\n\n"
        "BEST OPERATING POINT:\n"
        "   SETCN: 6,542 params, R²=0.196\n"
        "   = 1/700 of SPINT, closes ~30% of gap\n\n"
        "NEXT: multi-scale squeeze, SE + dilated stack,\n"
        "   r-sweep (r=4)"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#ccc"))
    fig.suptitle("SE Gain Decomposition & Conclusions", fontsize=14, fontweight="bold")
    pdf.savefig(fig); plt.close()

print(f"PDF saved to {OUT}")
