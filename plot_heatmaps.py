"""
Recreate the heatmap figures from toy_runs_analysis_fast.m using Python.

Reads CSVs from data_large/ (produced by batch_run_large_exploratory_stochastic_obsv.jl)
and generates 4 figures:
  1. Scores (Entropic risk measure)
  2. 5th percentile of landing values
  3. 10th percentile of landing values
  4. 25th percentile of landing values

Each figure has 5 subplots (one per sigma), with strategies on the y-axis
and reward transition k on the x-axis.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_large")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Strategy definitions: (display_name, csv_pattern_fragment)
# These match the CSV filenames produced by batch_run_large_exploratory_stochastic_obsv.jl
STRATEGIES = [
    # Percentile strategies (RiskEntropicSigma)
    ("Max σ",       "stochastic_max_perc"),
    ("Min σ",       "stochastic_min_perc"),
    ("Mean σ",      "stochastic_mean_perc"),
    ("C-wise σ",    "stochastic_cellwise_perc"),
    # CVaR strategies
    ("CVaR α 10",   "stochastic_CVaR_alpha10"),
    ("CVaR α 30",   "stochastic_CVaR_alpha30"),
    ("CVaR α 50",   "stochastic_CVaR_alpha50"),
    ("CVaR α 60",   "stochastic_CVaR_alpha60"),
    ("CVaR α 70",   "stochastic_CVaR_alpha70"),
    ("CVaR α 80",   "stochastic_CVaR_alpha80"),
    # EVaR strategies
    ("EVaR α 0.01", "stochastic_EVaR_alpha1"),
    ("EVaR α 0.05", "stochastic_EVaR_alpha5"),
    ("EVaR α 0.1",  "stochastic_EVaR_alpha10"),
    ("EVaR α 0.2",  "stochastic_EVaR_alpha20"),
    # Percentile + exploration
    ("Max σ expl",      "stochastic_max_perc_expl"),
    ("Min σ expl",      "stochastic_min_perc_expl"),
    ("Mean σ expl",     "stochastic_mean_perc_expl"),
    ("C-wise σ expl",   "stochastic_cellwise_perc_expl"),
    # CVaR + exploration
    ("CVaR α 10 expl",  "stochastic_CVaR_alpha10_expl"),
    ("CVaR α 30 expl",  "stochastic_CVaR_alpha30_expl"),
    ("CVaR α 50 expl",  "stochastic_CVaR_alpha50_expl"),
    ("CVaR α 60 expl",  "stochastic_CVaR_alpha60_expl"),
    ("CVaR α 70 expl",  "stochastic_CVaR_alpha70_expl"),
    ("CVaR α 80 expl",  "stochastic_CVaR_alpha80_expl"),
    # EVaR + exploration
    ("EVaR α 0.01 expl", "stochastic_EVaR_alpha1_expl"),
    ("EVaR α 0.05 expl", "stochastic_EVaR_alpha5_expl"),
    ("EVaR α 0.1 expl",  "stochastic_EVaR_alpha10_expl"),
    ("EVaR α 0.2 expl",  "stochastic_EVaR_alpha20_expl"),
]

K_VALUES = [("k=0", "k0"), ("k=2", "k2"), ("k=4", "k4"), ("k=7", "k7"), ("k=15", "k15")]
SIGMA_VALUES = [("σ=0.5", "0_5"), ("σ=1", "1_0"), ("σ=2", "2_0"), ("σ=3", "3_0"), ("σ=5", "5_0")]

# Metric definitions
BETA = 0.5

def entropic(v):
    """Entropic risk measure: -(1/beta) * log(E[exp(-beta * v)])"""
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan
    return -(1.0 / BETA) * np.log(np.mean(np.exp(-BETA * v)) + 1e-15)

def percentile_5(v):
    v = v[~np.isnan(v)]
    return np.percentile(v, 5) if len(v) > 0 else np.nan

def percentile_10(v):
    v = v[~np.isnan(v)]
    return np.percentile(v, 10) if len(v) > 0 else np.nan

def percentile_25(v):
    v = v[~np.isnan(v)]
    return np.percentile(v, 25) if len(v) > 0 else np.nan

# ─────────────────────────────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────────────────────────────

def load_landing_values(strat_csv_frag, k_frag, sigma_frag):
    """Load landing values (column 8) from the matching CSV."""
    pattern = f"batch_{strat_csv_frag}_{k_frag}_sigma{sigma_frag}.csv"
    filepath = os.path.join(DATA_DIR, pattern)
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        return df.iloc[:, 7].values  # column 8 (0-indexed: 7) = landing_value
    except Exception:
        return None

def compute_matrix(metric_fn, sigma_frag):
    """Compute metric for all strategies × k values at a given sigma."""
    M = np.full((len(STRATEGIES), len(K_VALUES)), np.nan)
    for si, (_, strat_frag) in enumerate(STRATEGIES):
        for ki, (_, k_frag) in enumerate(K_VALUES):
            v = load_landing_values(strat_frag, k_frag, sigma_frag)
            if v is not None and len(v) > 0:
                M[si, ki] = metric_fn(v)
    return M

# ─────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_heatmap_figure(metric_fn, metric_name, filename):
    """Generate a figure with 5 subplots (one per sigma)."""
    strat_names = [s[0] for s in STRATEGIES]
    k_names = [k[0] for k in K_VALUES]

    fig, axes = plt.subplots(1, 5, figsize=(24, max(8, len(STRATEGIES) * 0.4)),
                              sharey=True)
    fig.suptitle(f"{metric_name}", fontsize=14, fontweight='bold')

    # Compute global min/max for consistent colormap
    all_vals = []
    matrices = {}
    for sig_name, sig_frag in SIGMA_VALUES:
        M = compute_matrix(metric_fn, sig_frag)
        matrices[sig_frag] = M
        all_vals.extend(M[~np.isnan(M)].tolist())

    if len(all_vals) == 0:
        print(f"  No data found for {metric_name}, skipping.")
        plt.close(fig)
        return

    vmin, vmax = min(all_vals), max(all_vals)

    for idx, (sig_name, sig_frag) in enumerate(SIGMA_VALUES):
        ax = axes[idx]
        M = matrices[sig_frag]

        im = ax.imshow(M, aspect='auto', cmap='YlGnBu', vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(k_names)))
        ax.set_xticklabels(k_names, fontsize=8)
        ax.set_xlabel('Reward Transition', fontsize=9)
        ax.set_title(sig_name, fontsize=11)

        if idx == 0:
            ax.set_yticks(range(len(strat_names)))
            ax.set_yticklabels(strat_names, fontsize=7)
            ax.set_ylabel('Strategy', fontsize=9)

        # Annotate cells
        for si in range(M.shape[0]):
            for ki in range(M.shape[1]):
                if not np.isnan(M[si, ki]):
                    color = 'white' if M[si, ki] < (vmin + vmax) / 2 else 'black'
                    ax.text(ki, si, f'{M[si, ki]:.2f}', ha='center', va='center',
                            fontsize=6, color=color)

    fig.colorbar(im, ax=axes, shrink=0.4, pad=0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(OUT_DIR, filename)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")

# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating heatmap figures from data_large/ CSVs...")

    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\n  ERROR: {DATA_DIR} does not exist.")
        print("  Run batch_run_large_exploratory_stochastic_obsv.jl first to generate CSVs.")
        exit(1)

    csv_count = len(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"  Found {csv_count} CSVs in {DATA_DIR}")

    plot_heatmap_figure(entropic,       "Scores - Metric: Entropic",        "scores_entropic.png")
    plot_heatmap_figure(percentile_5,   "5th %ile - Metric: Entropic",      "percentile_5th.png")
    plot_heatmap_figure(percentile_10,  "10th %ile - Metric: Entropic",     "percentile_10th.png")
    plot_heatmap_figure(percentile_25,  "25th %ile - Metric: Entropic",     "percentile_25th.png")

    print("\nDone!")
