import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ── Paste your results or load from CSV ───────────────────────────
# results_df = pd.read_csv("zone_train_test_metrics.csv")

ZONE_METRICS = {
    "SW_MONSOON":  {
        "train": dict(rmse=8.228,  mae=3.559, mse=67.707,  r2=0.702, nse=0.702),
        "test":  dict(rmse=11.986, mae=4.947, mse=143.658, r2=0.495, nse=0.495),
    },
    "NE_MONSOON":  {
        "train": dict(rmse=5.916,  mae=2.904, mse=34.994,  r2=0.830, nse=0.830),
        "test":  dict(rmse=8.529,  mae=4.402, mse=72.745,  r2=0.241, nse=0.241),
    },
    "LOW_MONSOON": {
        "train": dict(rmse=3.123,  mae=1.241, mse=9.752,   r2=0.834, nse=0.834),
        "test":  dict(rmse=4.973,  mae=1.954, mse=24.729,  r2=0.560, nse=0.560),
    },
}

# ── Or build directly from your df ───────────────────────────────
def build_zone_dict(df: pd.DataFrame) -> dict:
    out = {}
    for zone in df["zone"].unique():
        out[zone] = {}
        for split in ["train", "test"]:
            row = df[(df["zone"] == zone) & (df["split"] == split)].iloc[0]
            out[zone][split] = row[["rmse", "mae", "mse", "r2", "nse"]].to_dict()
    return out


# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc", "axes.labelcolor": "#333333",
    "xtick.color": "#555555", "ytick.color": "#555555",
    "text.color": "#333333", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e0e0e0", "grid.linestyle": "--", "grid.alpha": 0.7,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})

TRAIN_COLOR = "#378ADD"
TEST_COLOR  = "#D4537E"
ZONE_COLORS = {"SW_MONSOON": "#534AB7", "NE_MONSOON": "#1D9E75", "LOW_MONSOON": "#BA7517"}
ZONE_SHORT  = {"SW_MONSOON": "SW", "NE_MONSOON": "NE", "LOW_MONSOON": "LOW"}

os.makedirs("report_plots", exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  FIG 1 — Grouped bar: RMSE & MAE per zone
# ══════════════════════════════════════════════════════════════════
def plot_error_metrics(zone_dict: dict):
    zones  = list(zone_dict.keys())
    labels = [ZONE_SHORT[z] for z in zones]
    x      = np.arange(len(zones))
    w      = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Error Metrics — Train vs Test by Zone", fontsize=14, fontweight="bold", y=1.02)

    for ax, metric, ylabel, title in zip(
        axes,
        ["rmse", "mae"],
        ["RMSE (mm)", "MAE (mm)"],
        ["Root Mean Squared Error", "Mean Absolute Error"],
    ):
        train_vals = [zone_dict[z]["train"][metric] for z in zones]
        test_vals  = [zone_dict[z]["test"][metric]  for z in zones]

        bars_tr = ax.bar(x - w/2, train_vals, width=w, color=TRAIN_COLOR, label="Train", zorder=3, linewidth=0)
        bars_te = ax.bar(x + w/2, test_vals,  width=w, color=TEST_COLOR,  label="Test",  zorder=3,
                         linewidth=1, edgecolor=TEST_COLOR, alpha=0.75)

        for bar in bars_tr:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color=TRAIN_COLOR)
        for bar in bars_te:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color=TEST_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.set_ylim(0, max(test_vals) * 1.25)

    plt.tight_layout()
    fig.savefig("report_plots/fig1_error_metrics.png", dpi=300, bbox_inches="tight")
    print("Saved: fig1_error_metrics.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  FIG 2 — R² & NSE skill scores
# ══════════════════════════════════════════════════════════════════
def plot_skill_scores(zone_dict: dict):
    zones  = list(zone_dict.keys())
    labels = [ZONE_SHORT[z] for z in zones]
    x      = np.arange(len(zones))
    w      = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Model Skill Scores — Train vs Test by Zone", fontsize=14, fontweight="bold", y=1.02)

    for ax, metric, title in zip(axes, ["r2", "nse"], ["R² Score", "Nash–Sutcliffe Efficiency (NSE)"]):
        train_vals = [zone_dict[z]["train"][metric] for z in zones]
        test_vals  = [zone_dict[z]["test"][metric]  for z in zones]

        ax.bar(x - w/2, train_vals, width=w, color=TRAIN_COLOR, label="Train", zorder=3)
        ax.bar(x + w/2, test_vals,  width=w, color=TEST_COLOR,  label="Test",  zorder=3, alpha=0.75)
        ax.axhline(1.0, color="#aaaaaa", lw=1, ls="--", label="Perfect = 1.0")

        for i, (tv, ev) in enumerate(zip(train_vals, test_vals)):
            ax.text(i - w/2, tv + 0.01, f"{tv:.3f}", ha="center", va="bottom", fontsize=9, color=TRAIN_COLOR)
            ax.text(i + w/2, ev + 0.01, f"{ev:.3f}", ha="center", va="bottom", fontsize=9, color=TEST_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_ylim(0, 1.12)
        ax.legend(framealpha=0.9)

    plt.tight_layout()
    fig.savefig("report_plots/fig2_skill_scores.png", dpi=300, bbox_inches="tight")
    print("Saved: fig2_skill_scores.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  FIG 3 — Overfit gap (RMSE test − train)
# ══════════════════════════════════════════════════════════════════
def plot_overfit_gap(zone_dict: dict):
    zones  = list(zone_dict.keys())
    labels = [ZONE_SHORT[z] for z in zones]
    gaps   = [zone_dict[z]["test"]["rmse"] - zone_dict[z]["train"]["rmse"] for z in zones]
    colors = [ZONE_COLORS[z] for z in zones]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, gaps, color=colors, zorder=3, width=0.45)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f"+{gap:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("RMSE gap (mm)")
    ax.set_title("Generalisation Gap — Test RMSE minus Train RMSE\n(smaller = better generalisation)", fontsize=13)
    ax.set_ylim(0, max(gaps) * 1.3)

    patches = [mpatches.Patch(color=ZONE_COLORS[z], label=ZONE_SHORT[z]) for z in zones]
    ax.legend(handles=patches, framealpha=0.9)

    plt.tight_layout()
    fig.savefig("report_plots/fig3_overfit_gap.png", dpi=300, bbox_inches="tight")
    print("Saved: fig3_overfit_gap.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  FIG 4 — All 5 metrics heatmap (zones × metric × split)
# ══════════════════════════════════════════════════════════════════
def plot_heatmap(zone_dict: dict):
    zones   = list(zone_dict.keys())
    metrics = ["rmse", "mae", "mse", "r2", "nse"]
    splits  = ["train", "test"]

    row_labels = [f"{ZONE_SHORT[z]} {s.capitalize()}" for z in zones for s in splits]
    matrix     = np.array([
        [zone_dict[z][s][m] for m in metrics]
        for z in zones for s in splits
    ])

    # Normalise each column 0–1 for colour, but show raw values as text
    norm_matrix = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        col_data = matrix[:, col]
        mn, mx   = col_data.min(), col_data.max()
        norm_matrix[:, col] = (col_data - mn) / (mx - mn + 1e-9)

    # For error metrics lower=better → invert colour
    for col_idx in [0, 1, 2]:
        norm_matrix[:, col_idx] = 1 - norm_matrix[:, col_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(norm_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("All Metrics Heatmap — green = better performance", fontsize=13, pad=12)

    for i in range(len(row_labels)):
        for j, metric in enumerate(metrics):
            val = matrix[i, j]
            txt = f"{val:.3f}" if metric in ("r2","nse") else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < norm_matrix[i,j] < 0.85 else "white")

    # Draw zone separator lines
    for k in [1, 3]:
        ax.axhline(k + 0.5, color="white", lw=2)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised score (1 = best)")
    plt.tight_layout()
    fig.savefig("report_plots/fig4_heatmap.png", dpi=300, bbox_inches="tight")
    print("Saved: fig4_heatmap.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  FIG 5 — Overall averaged metric bar (single summary figure)
# ══════════════════════════════════════════════════════════════════
def plot_overall_summary(zone_dict: dict):
    zones   = list(zone_dict.keys())
    metrics = ["rmse", "mae", "r2", "nse"]
    labels  = ["RMSE", "MAE", "R²", "NSE"]

    avg_train = {m: np.mean([zone_dict[z]["train"][m] for z in zones]) for m in metrics}
    avg_test  = {m: np.mean([zone_dict[z]["test"][m]  for z in zones]) for m in metrics}

    x = np.arange(len(metrics))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_tr = ax.bar(x - w/2, [avg_train[m] for m in metrics], width=w,
                     color=TRAIN_COLOR, label="Train (avg)", zorder=3)
    bars_te = ax.bar(x + w/2, [avg_test[m]  for m in metrics], width=w,
                     color=TEST_COLOR,  label="Test (avg)",  zorder=3, alpha=0.8)

    for bar in list(bars_tr) + list(bars_te):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("Overall Average Metrics Across All Three Zones", fontsize=13)
    ax.set_ylabel("Metric value")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(avg_test["rmse"], avg_train["rmse"]) * 1.25)

    plt.tight_layout()
    fig.savefig("report_plots/fig5_overall_summary.png", dpi=300, bbox_inches="tight")
    print("Saved: fig5_overall_summary.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  RUN ALL
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # If loading from saved CSV:
    # df = pd.read_csv("zone_train_test_metrics.csv")
    # zone_dict = build_zone_dict(df)

    zone_dict = ZONE_METRICS   # or pass build_zone_dict(df)

    plot_error_metrics(zone_dict)
    plot_skill_scores(zone_dict)
    plot_overfit_gap(zone_dict)
    plot_heatmap(zone_dict)
    plot_overall_summary(zone_dict)

    print("\nAll 5 figures saved to report_plots/")