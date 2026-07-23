#!/usr/bin/env python3
"""Generate a plain-language PDF explaining the differences between all the
network architectures tested. Visual block diagrams + intuitive explanations.
Uses DroidSansFallback for CJK rendering."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

CJK_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(CJK_PATH)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "network_diff_guide.pdf"

C_INPUT = "#bbdefb"
C_CONV = "#c8e6c9"
C_FC = "#fff9c4"
C_GATE = "#ffcdd2"
C_OUT = "#e1bee7"
C_SPINT = "#ffccbc"

def box(ax, x, y, w, h, text, color=C_FC, fontsize=8, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor="#555", lw=1.2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)

def arrow(ax, x1, y1, x2, y2, color="#666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


with PdfPages(OUT) as pdf:

    # ===== PAGE 1: Big picture =====
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fig.suptitle("Core Differences Between Networks \u2014 Plain Language",
                 fontsize=20, fontweight="bold", y=0.96)
    ax.text(5, 8.8, "Task: from 96-channel neural signals -> predict finger x/y velocity (20ms per frame)",
            ha="center", fontsize=12, color="#333")
    ax.text(5, 8.2, "All networks take the same input (B,96,50) and produce the same output (B,2).\nThe ONLY difference is what happens in the middle.",
            ha="center", fontsize=10, color="#666")

    box(ax, 0.5, 6.8, 1.5, 0.8, "Neural input\n96ch x 50f", C_INPUT, 8)
    arrow(ax, 2.0, 7.2, 2.8, 7.2)
    box(ax, 2.8, 6.8, 2.5, 0.8, "??? difference here ???", "#ffffff", 9, True)
    arrow(ax, 5.3, 7.2, 6.1, 7.2)
    box(ax, 6.1, 6.8, 1.5, 0.8, "Finger vel\nx, y", C_OUT, 8)

    ax.text(5, 5.8, "One-sentence summary: every method answers the same question",
            ha="center", fontsize=12, fontweight="bold", color="#222")
    ax.text(5, 5.3, '"Which neurons matter, and how much?"',
            ha="center", fontsize=13, color="#d32f2f")
    ax.text(5, 4.95, "...they differ only in the TIMESCALE of context used to answer it  (per-frame / per-window / per-session)",
            ha="center", fontsize=9, color="#555")

    items = [
        ("plain TCN", "Fixed answer:\nweights are frozen\nafter training", C_CONV),
        ("SPINT (teacher)", "Per-window vote over\nneurons: softmax\nattention (fixed queries)", C_SPINT),
        ("SE TCN [BEST]", "Gate from the window\nsummary scales each\nchannel (squeeze)", C_GATE),
        ("FiLM TCN", "Adjust once per session:\nuses calib data to set\nparams, frozen in session", C_FC),
        ("GLU TCN", "Internal switch:\ngating at the 32-dim\nhidden layer", C_CONV),
        ("grouped SE", "Separate votes:\ndifferent gates for\nx-output vs y-output", C_GATE),
    ]
    for i, (name, desc, color) in enumerate(items):
        col = i % 3; row = i // 3
        x = 0.5 + col * 3.2; y = 3.0 - row * 2.2
        box(ax, x, y, 2.8, 1.8, f"{name}\n\n{desc}", color, 8.5)
    pdf.savefig(fig); plt.close()

    # ===== PAGE 2: plain TCN vs SPINT =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"hspace": 0.4})
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    # --- plain TCN ---
    ax = axes[0]
    ax.set_title("(1) plain TCN -- Fixed Weights (4,130 params, R2=0.168)",
                 fontsize=12, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.3, 1.0, "Input\n96x50", C_INPUT, 8)
    arrow(ax, 1.6, 3.0, 2.2, 3.0)
    box(ax, 2.2, 2.5, 1.8, 1.0, "Temporal conv\n9 taps/channel\n(depthwise FIR)", C_CONV, 8)
    arrow(ax, 4.0, 3.0, 4.6, 3.0)
    box(ax, 4.6, 2.5, 1.5, 1.0, "Take last\nframe (96d)", "#ffffff", 8)
    arrow(ax, 6.1, 3.0, 6.7, 3.0)
    box(ax, 6.7, 2.5, 2.0, 1.0, "Linear mix\n96->32->2\n(FIXED weights)", C_FC, 8)
    arrow(ax, 8.7, 3.0, 9.3, 3.0)
    box(ax, 9.3, 2.5, 0.6, 1.0, "x,y", C_OUT, 8)
    ax.text(5, 1.3, "How it works: each channel gets a 9-tap temporal filter, then a FIXED linear layer mixes\n"
                    "the 96 channels into 2 outputs.",
            ha="center", fontsize=9, color="#444")
    ax.text(5, 0.6, "Weakness: weights are frozen after training. The mixing ratio of neuron 3 vs neuron 47\n"
                    "NEVER changes, regardless of what the neural activity looks like this frame.",
            ha="center", fontsize=9, color="#d32f2f")

    # --- SPINT ---
    ax = axes[1]
    ax.set_title("(2) SPINT teacher -- Dynamic Attention over Neurons, per window (4,569,288 params, R2=0.26)",
                 fontsize=12, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.3, 1.0, "Input\n96x50", C_INPUT, 8)
    arrow(ax, 1.6, 3.0, 2.2, 3.0)
    box(ax, 2.2, 2.5, 1.5, 1.0, "Read-in\n50f->512\n(per neuron)", C_FC, 8)
    arrow(ax, 3.7, 3.0, 4.3, 3.0)
    box(ax, 4.3, 2.5, 2.2, 1.0, "Cross-attention\nover 96 neurons\n(fixed queries, softmax)", C_SPINT, 8, True)
    arrow(ax, 6.5, 3.0, 7.1, 3.0)
    box(ax, 7.1, 2.5, 1.5, 1.0, "FFN\n512->2048\n->512", C_FC, 8)
    arrow(ax, 8.6, 3.0, 9.2, 3.0)
    box(ax, 9.2, 2.5, 0.6, 1.0, "x,y", C_OUT, 8)
    ax.text(5, 1.35, "How it works: each neuron's 50-frame trace is embedded to 512; then a few FIXED learned queries attend\n"
                     "over the 96 neurons (softmax) to pick 'which neurons to read, and how much' for THIS WINDOW.\n"
                     "An ID encoder also gives each neuron a session-specific fingerprint.",
            ha="center", fontsize=9, color="#444")
    ax.text(5, 0.5, "Cost: 4.6M params, needs softmax + LayerNorm -- extremely expensive on-chip. This is what Plan B avoids.",
            ha="center", fontsize=9, color="#d32f2f")
    pdf.savefig(fig); plt.close()

    # ===== PAGE 3: SE vs FiLM vs GLU =====
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"hspace": 0.5})
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    # --- SE TCN ---
    ax = axes[0]
    ax.set_title("(3) SETCN -- Window-Summary Channel Gate [BEST] (6,542 params, R2=0.196, +0.033, 4.7σ)",
                 fontsize=11, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.2, 1.0, "Input\n96x50", C_INPUT, 7.5)
    arrow(ax, 1.5, 3.0, 2.1, 3.0)
    box(ax, 2.1, 2.5, 1.5, 1.0, "Temporal\nconv 96xT", C_CONV, 7.5)
    arrow(ax, 3.6, 3.0, 4.2, 3.0)
    box(ax, 4.2, 2.5, 2.2, 1.0, "SE gate\nsqueeze -> MLP\n-> sigmoid -> scale", C_GATE, 7.5, True)
    arrow(ax, 6.4, 3.0, 7.0, 3.0)
    box(ax, 7.0, 2.5, 1.5, 1.0, "Linear\n96->32->2", C_FC, 7.5)
    arrow(ax, 8.5, 3.0, 9.1, 3.0)
    box(ax, 9.1, 2.5, 0.6, 1.0, "x,y", C_OUT, 7.5)
    ax.text(5, 1.3, "What it does: averages over the window (squeeze) to get a per-channel context summary, then a small MLP\n"
                    "computes a per-channel scale (0~1) that boosts/suppresses each neuron based on the WINDOW summary.",
            ha="center", fontsize=8.5, color="#333")
    ax.text(5, 0.5, "Key finding (+0.033, 4.7σ, 3/3 seeds): the squeeze (window context) is the active ingredient.\n"
                    "Removing it (see chnosqz, next page) drops back to baseline -- the channel location alone adds ~nothing.",
            ha="center", fontsize=8.5, color="#2e7d32")

    # --- FiLM ---
    ax = axes[1]
    ax.set_title("(4) FiLM TCN -- Adjust Once Per Session (16,642 params, R2=0.133, -0.035 FAILED)",
                 fontsize=11, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.2, 1.0, "Input\n96x50", C_INPUT, 7.5)
    arrow(ax, 1.5, 3.0, 2.1, 3.0)
    box(ax, 2.1, 2.5, 1.5, 1.0, "Temporal\nconv", C_CONV, 7.5)
    arrow(ax, 3.6, 3.0, 4.2, 3.0)
    box(ax, 4.2, 2.5, 1.5, 1.0, "Last frame\n96d", "#fff", 7.5)
    arrow(ax, 5.7, 3.0, 6.3, 3.0)
    box(ax, 6.3, 2.5, 2.0, 1.0, "FiLM modulate\ny = gamma*h+beta\n(ONCE per session)", C_GATE, 7.5)
    arrow(ax, 8.3, 3.0, 8.9, 3.0)
    box(ax, 8.9, 2.5, 0.8, 1.0, "Linear\n->x,y", C_FC, 7.5)
    ax.text(5, 1.3, "What it does: uses calib stats (CORAL a,b) to compute gamma,beta -- an affine transform\n"
                    "applied identically to EVERY frame in the same session.",
            ha="center", fontsize=8.5, color="#333")
    ax.text(5, 0.5, "Why it fails: CORAL already did the linear alignment. FiLM re-infers modulation from the\n"
                    "SAME (a,b) = redundant. On well-aligned sessions it just adds noise.",
            ha="center", fontsize=8.5, color="#d32f2f")

    # --- GLU ---
    ax = axes[2]
    ax.set_title("(5) GLU TCN -- Hidden-Layer Switch (7,234 params, R2=0.136, -0.032 FAILED)",
                 fontsize=11, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.2, 1.0, "Input\n96x50", C_INPUT, 7.5)
    arrow(ax, 1.5, 3.0, 2.1, 3.0)
    box(ax, 2.1, 2.5, 1.5, 1.0, "Temporal\nconv", C_CONV, 7.5)
    arrow(ax, 3.6, 3.0, 4.2, 3.0)
    box(ax, 4.2, 2.5, 1.5, 1.0, "Last frame\n96d", "#fff", 7.5)
    arrow(ax, 5.7, 3.0, 6.3, 3.0)
    box(ax, 6.3, 2.5, 2.2, 1.0, "GLU gate\na * sigmoid(g)\nat 32-dim hidden", C_GATE, 7.5)
    arrow(ax, 8.5, 3.0, 9.1, 3.0)
    box(ax, 9.1, 2.5, 0.6, 1.0, "x,y", C_OUT, 7.5)
    ax.text(5, 1.3, "What it does: replaces ReLU in the linear layer with 'value x gate', giving the\n"
                    "32-dim hidden representation a dynamic switch.",
            ha="center", fontsize=8.5, color="#333")
    ax.text(5, 0.5, "Why it fails: the gate is at the 32-dim hidden layer (AFTER channels are already mixed).\n"
                    "Too downstream to re-route neuron information. SE gates at 96-dim channel level -- the right place.",
            ha="center", fontsize=8.5, color="#d32f2f")
    pdf.savefig(fig); plt.close()

    # ===== PAGE 4 (NEW): the deconfound + the timescale axis =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"hspace": 0.5})
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    # --- chnosqz deconfound ---
    ax = axes[0]
    ax.set_title("(*) chnosqz -- The Deconfound: channel gate WITHOUT the squeeze (6,542 params, R2=0.165, +0.002)",
                 fontsize=11, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.2, 1.0, "Input\n96x50", C_INPUT, 7.5)
    arrow(ax, 1.5, 3.0, 2.1, 3.0)
    box(ax, 2.1, 2.5, 1.5, 1.0, "Temporal\nconv", C_CONV, 7.5)
    arrow(ax, 3.6, 3.0, 4.2, 3.0)
    box(ax, 4.2, 2.5, 2.2, 1.0, "Channel gate\nfrom LAST frame\n(NO squeeze)", "#e0e0e0", 7.5, True)
    arrow(ax, 6.4, 3.0, 7.0, 3.0)
    box(ax, 7.0, 2.5, 1.5, 1.0, "Linear\n96->32->2", C_FC, 7.5)
    arrow(ax, 8.5, 3.0, 9.1, 3.0)
    box(ax, 9.1, 2.5, 0.6, 1.0, "x,y", C_OUT, 7.5)
    ax.text(5, 1.3, "What it does: IDENTICAL to SE, except the gate is computed from the single last frame instead of\n"
                    "the time-averaged window summary. This isolates 'gating at the channel axis' from 'the squeeze'.",
            ha="center", fontsize=8.5, color="#333")
    ax.text(5, 0.5, "Why it matters: it gives +0.002 = plain. So SE's +0.033 is NOT from gating at the channel axis --\n"
                    "it is ENTIRELY from the squeeze (the window-average context). The per-frame part contributes nothing.",
            ha="center", fontsize=8.5, color="#2e7d32")

    # --- timescale axis ---
    ax = axes[1]
    ax.set_title("The Real Axis: the TIMESCALE of context that drives the gate",
                 fontsize=13, fontweight="bold", loc="left")
    ax.text(5, 3.6, "Every gate answers the SAME question ('which neurons matter, how much'); the only thing that\n"
                    "changes is the timescale of context it reads. The gap-closer is the WINDOW timescale -- not per-frame, not per-session.",
            ha="center", fontsize=9, color="#333")
    ts = [
        (1.4, "PER-FRAME\n(chnosqz)\n\nΔ = +0.002\nNULL", "#e0e0e0", "too fast / noisy"),
        (4.4, "PER-WINDOW\n(SE squeeze)\n\nΔ = +0.033\nWIN", "#a5d6a7", "the sweet spot"),
        (7.4, "PER-SESSION\n(FiLM)\n\nΔ = -0.035\nFAIL", "#ef9a9a", "too slow / CORAL-redundant"),
    ]
    for x, txt, col, note in ts:
        box(ax, x, 1.6, 2.2, 1.7, txt, col, 9, True)
        ax.text(x + 1.1, 1.35, note, ha="center", fontsize=8, color="#555", style="italic")
    ax.annotate("", xy=(9.6, 0.9), xytext=(0.9, 0.9),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))
    ax.text(0.9, 0.55, "faster", ha="left", fontsize=8, color="#888")
    ax.text(9.6, 0.55, "slower", ha="right", fontsize=8, color="#888")
    ax.text(5.25, 0.55, "context timescale", ha="center", fontsize=8, color="#888")
    pdf.savefig(fig); plt.close()

    # ===== PAGE 5: grouped SE + summary =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"hspace": 0.4})
    for ax in axes[:2]:
        ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    # --- grouped SE ---
    ax = axes[0]
    ax.set_title("(6) grouped SE -- Separate Votes for x and y (7,757 params, R2=0.179, worse than shared SE)",
                 fontsize=11, fontweight="bold", loc="left")
    box(ax, 0.3, 2.5, 1.2, 1.0, "Input\n96x50", C_INPUT, 7.5)
    arrow(ax, 1.5, 3.0, 2.1, 3.0)
    box(ax, 2.1, 2.5, 1.5, 1.0, "Temporal\nconv", C_CONV, 7.5)
    arrow(ax, 3.6, 3.0, 4.2, 3.0)
    box(ax, 4.2, 2.5, 2.5, 1.0, "Two gate groups\n96 gates for x\n96 gates for y", C_GATE, 7.5, True)
    arrow(ax, 6.7, 3.0, 7.3, 3.0)
    box(ax, 7.3, 2.5, 1.5, 1.0, "Separate\nreadout\n->x, ->y", C_FC, 7.5)
    arrow(ax, 8.8, 3.0, 9.4, 3.0)
    box(ax, 9.4, 2.5, 0.4, 1.0, "x,y", C_OUT, 7.5)
    ax.text(5, 1.3, "Motivation: x-direction and y-direction tuned neurons are different subpopulations,\n"
                    "so give x and y their own separate gates. Prediction: if it improves, output-specific routing matters.",
            ha="center", fontsize=8.5, color="#333")
    ax.text(5, 0.5, "Result: it does NOT help. The two outputs' dynamic neuron subsets overlap enough that\n"
                    "a shared gate suffices. Separate gates just add estimation noise (2x params from same data).",
            ha="center", fontsize=8.5, color="#d32f2f")

    # --- summary ---
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Summary: What Worked, What Didn't, and Why",
                 fontsize=13, fontweight="bold", loc="left")
    summary = (
        "KEY INSIGHT\n\n"
        "Every method answers the SAME question -- 'which neurons matter, and how much?'\n"
        "-- and differs ONLY in the TIMESCALE of context it uses to answer:\n\n"
        "  plain TCN : never answers -- weights are frozen, same every window\n"
        "  SPINT     : re-weights per WINDOW over neurons (softmax) -- strongest but most expensive\n"
        "  SE TCN *  : gates from the WINDOW summary (squeeze -> sigmoid scale) -- best value\n"
        "  chnosqz   : same gate but from the last frame (no squeeze) -- NULL (+0.002), proves the squeeze is the load\n"
        "  FiLM      : adjusts once per SESSION -- too coarse, redundant with CORAL\n"
        "  GLU       : switches at the hidden layer -- too late (after mixing)\n"
        "  grouped SE: separate votes for x/y -- unnecessary (subsets overlap)\n\n"
        "CONCLUSION\n"
        "  [YES] Gating driven by the WINDOW-level summary -- not per-frame (chnosqz null), not per-session (FiLM redundant)\n"
        "  [NO]  Static widening / deepening / session-level / per-frame gating all failed\n"
        "  -> The active ingredient is WINDOW-level adaptive gain (normalization), not per-frame routing"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=8.5,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#ccc"))
    pdf.savefig(fig); plt.close()

print(f"PDF saved to {OUT}")
