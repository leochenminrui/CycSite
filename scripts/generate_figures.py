#!/usr/bin/env python3
"""Generate publication figures from the study result files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Muted versions of the conceptual colors used in Figure 1.
CHEMISTRY = "#C87812"
SITE = "#2F7D4A"
CONTEXT = "#A64D79"
NEUTRAL = "#6B7280"
MOLECULAR = "#40566E"
SITE_SUBBLOCKS = {"B1": "#1F6B3A", "B2": "#4D8F62", "B3": "#79A984"}
COLORS = {"A": CHEMISTRY, **SITE_SUBBLOCKS, "C": CONTEXT}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

LABELS = {
    "edit_n_methylation": "$N$-methylation count",
    "logp": "LogP",
    "anchor_pos_std": "SD of normalized anchor position",
    "num_rotatable_bonds": "Rotatable-bond count",
    "mol_weight": "Molecular weight",
    "tpsa": "TPSA",
    "anchor_pos_mean": "Mean normalized anchor position",
    "anchor_c_terminal_count": "C-terminal edit count",
    "anchor_res_L": "Leucine fraction at anchors",
    "num_h_donors": "Hydrogen-bond donors",
    "edit_d_amino_acid": "$D$-amino-acid count",
    "anchor_res_P": "Proline fraction at anchors",
    "anchor_res_F": "Phenylalanine fraction at anchors",
    "num_aliphatic_rings": "Aliphatic rings",
    "anchor_density": "Anchor density",
    "cyclization_ring_size": "Cyclization ring size",
}


def style_axis(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, color="#555555")
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D7DCE2", linewidth=0.55, alpha=0.7)


def panel_label(ax, label):
    ax.text(
        -0.10, 1.03, label, transform=ax.transAxes,
        fontsize=11, fontweight="bold", ha="left", va="bottom",
    )


def save(fig, name, apply_tight_layout=True):
    if apply_tight_layout:
        fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def shap_top15():
    data = pd.read_csv(RESULTS / "shap/feature_summary_with_ci.csv")
    data = data[data.representation == "A+B+C"].nlargest(15, "mean").sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    labels = [LABELS.get(name, name.replace("_", " ")) for name in data.feature]
    ax.barh(
        labels, data["mean"], height=0.68,
        color=[COLORS[group] for group in data.subblock],
        edgecolor="white", linewidth=0.45,
    )
    ax.errorbar(
        data["mean"], labels,
        xerr=[
            data["mean"] - data.bootstrap_bca_ci95_low,
            data.bootstrap_bca_ci95_high - data["mean"],
        ],
        fmt="none", ecolor="#30343B", capsize=2, capthick=0.7, linewidth=0.7,
    )
    legend = {
        "A": "whole-peptide Chemistry",
        "B1": "Site position",
        "B2": "Site residue identity",
        "B3": "Site residue properties/terminal context",
        "C": "scaffold/multi-edit Context",
    }
    ax.legend(
        handles=[Patch(facecolor=COLORS[key], label=f"{key}: {legend[key]}") for key in COLORS],
        ncol=2, frameon=False, loc="lower right", handlelength=1.2,
        columnspacing=1.0, labelspacing=0.35,
    )
    ax.set_xlabel("Mean absolute TreeSHAP attribution proportion")
    ax.set_xlim(left=0)
    style_axis(ax, "x")
    save(fig, "figure_shap_top15_features")


def predicted_observed():
    seeds = pd.read_csv(RESULTS / "seed_level_results.csv")
    representations = ["A", "A+B", "A+B+C"]
    rows = []
    for split in ("random", "sequence_cluster"):
        for representation in representations:
            paths = seeds.query(
                "split == @split and representation == @representation"
            ).sort_values("seed").prediction_file
            frames = [pd.read_csv(ROOT / path) for path in paths]
            base = frames[0][["sample_id", "y_true"]].copy()
            base["y_pred"] = np.mean([frame.y_pred.to_numpy() for frame in frames], axis=0)
            base["split"] = split
            base["representation"] = representation
            rows.append(base)
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(OUT / "predicted_vs_observed_source.csv", index=False)
    low = min(data.y_true.min(), data.y_pred.min())
    high = max(data.y_true.max(), data.y_pred.max())
    padding = 0.03 * (high - low)
    limits = [low - padding, high + padding]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), sharex=True, sharey=True)
    for row, split in enumerate(("random", "sequence_cluster")):
        for column, representation in enumerate(representations):
            ax = axes[row, column]
            frame = data.query("split == @split and representation == @representation")
            ax.scatter(
                frame.y_true, frame.y_pred, s=7, alpha=0.18,
                color=MOLECULAR, edgecolors="none", rasterized=True,
            )
            ax.plot(limits, limits, color="#222222", linewidth=0.8, linestyle="--")
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            if row == 0:
                ax.set_title(representation, fontsize=9, pad=5)
            style_axis(ax)
        axes[row, 0].text(
            -0.30, 0.5, "Random split" if split == "random" else "Strict sequence split",
            transform=axes[row, 0].transAxes, rotation=90, va="center", ha="center",
            fontsize=9,
        )
    for ax in axes[-1]:
        ax.set_xlabel("Observed permeability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean prediction across 30 seeds")
    save(fig, "figure_predicted_vs_observed")


def time_forward():
    performance = pd.read_csv(RESULTS / "time_forward/model_performance_with_ci.csv")
    performance = performance[performance.metric == "spearman"]
    similarity = pd.read_csv(RESULTS / "time_forward/similarity_diagnostics.csv")
    cutoff = pd.read_csv(RESULTS / "time_forward/cutoff_statistics.csv")
    styles = {
        "Sequence": (NEUTRAL, "o", ":"),
        "Chemistry": (CHEMISTRY, "s", "-"),
        "Chemistry + Site": (SITE, "o", "-"),
        "Chemistry + Site + Context": (CONTEXT, "D", "-"),
        "RDKit": ("#708090", "^", "--"),
        "ECFP": (MOLECULAR, "v", "--"),
    }
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1], "hspace": 0.12},
    )
    for model, (color, marker, linestyle) in styles.items():
        frame = performance[performance.model == model].sort_values("cutoff")
        axes[0].errorbar(
            frame.cutoff, frame.estimate,
            yerr=[frame.estimate - frame.ci95_low, frame.ci95_high - frame.estimate],
            marker=marker, markersize=3.8, linestyle=linestyle, linewidth=1.1,
            color=color, ecolor=color, elinewidth=0.6, capsize=1.8, alpha=0.95,
            label=model,
        )
    axes[0].axhline(0, color="#333333", linewidth=0.7)
    axes[0].set_ylabel("Spearman correlation\n(95% grouped-bootstrap CI)")
    axes[0].legend(
        ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2, handlelength=2.0,
    )
    style_axis(axes[0], "y")
    panel_label(axes[0], "A")

    axes[1].plot(
        similarity.cutoff, similarity.median_max_tanimoto,
        marker="o", markersize=4, linewidth=1.2, color=MOLECULAR,
        label="Median maximum Tanimoto",
    )
    right = axes[1].twinx()
    right.bar(cutoff.cutoff, cutoff.test_n, width=0.56, color=NEUTRAL, alpha=0.22)
    axes[1].set_ylabel("Median max Tanimoto")
    right.set_ylabel("Test-set size", color=NEUTRAL)
    right.tick_params(axis="y", colors=NEUTRAL, width=0.7)
    right.spines["top"].set_visible(False)
    right.spines["right"].set_linewidth(0.7)
    axes[1].set_xlabel("Cutoff year")
    axes[1].set_xticks(sorted(cutoff.cutoff.unique()))
    style_axis(axes[1], "y")
    panel_label(axes[1], "B")
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.10, top=0.89, hspace=0.16)
    save(fig, "figure_time_forward_shift_diagnostics", apply_tight_layout=False)


def scaffold():
    paired = pd.read_csv(
        RESULTS / "scaffold_ranking/paired_comparison_AB_vs_A_family_values.csv"
    ).dropna(subset=["difference"]).sort_values("difference")
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    colors = np.where(paired.difference >= 0, SITE, CHEMISTRY)
    ax.bar(np.arange(len(paired)), paired.difference, width=0.82, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.75)
    ax.set_xlim(-0.7, len(paired) - 0.3)
    ax.set_xticks([])
    ax.set_xlabel("Peptide families ordered by Chemistry+Site minus Chemistry")
    ax.set_ylabel("Pairwise-accuracy difference")
    style_axis(ax, "y")
    save(fig, "figure_scaffold_family_differences")

    summary = pd.read_csv(RESULTS / "scaffold_ranking/summary_with_ci.csv")
    order = ["Sequence", "Chemistry", "Chemistry + Site", "Chemistry + Site + Context", "ECFP"]
    frame = summary.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(frame))
    bar_colors = [NEUTRAL, CHEMISTRY, SITE, CONTEXT, MOLECULAR]
    ax.bar(x, frame.mean_family_pairwise_accuracy, width=0.68, color=bar_colors)
    ax.errorbar(
        x, frame.mean_family_pairwise_accuracy,
        yerr=[
            frame.mean_family_pairwise_accuracy - frame.family_bootstrap_ci95_low,
            frame.family_bootstrap_ci95_high - frame.mean_family_pairwise_accuracy,
        ],
        fmt="none", ecolor="#30343B", capsize=2.5, capthick=0.75, linewidth=0.75,
    )
    lower = min(0.48, float(frame.family_bootstrap_ci95_low.min()) - 0.015)
    upper = float(frame.family_bootstrap_ci95_high.max()) + 0.015
    ax.set_ylim(lower, upper)
    ax.axhline(0.5, color="#777777", linewidth=0.65, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Sequence", "Chemistry", "Chemistry\n+ Site", "Chemistry + Site\n+ Context", "ECFP"])
    ax.set_ylabel("Mean family pairwise accuracy\n(95% family-bootstrap CI)")
    style_axis(ax, "y")
    save(fig, "figure_scaffold_ranking_summary")


def incremental_effects():
    primary = pd.read_csv(RESULTS / "paired_comparisons.csv")
    primary = primary[primary.metric == "r2"].copy()
    secondary = pd.read_csv(
        RESULTS / "secondary_incremental_analysis/cross_estimator_paired_effects.csv"
    )

    def primary_row(split, contrast, reverse=False):
        row = primary.query("split == @split and contrast == @contrast").iloc[0]
        if reverse:
            return {
                "mean": -row.mean_difference,
                "low": -row.bootstrap_bca_ci95_high,
                "high": -row.bootstrap_bca_ci95_low,
            }
        return {
            "mean": row.mean_difference,
            "low": row.bootstrap_bca_ci95_low,
            "high": row.bootstrap_bca_ci95_high,
        }

    panel_a = pd.DataFrame([
        {"split": "Random", "effect": "Site", **primary_row("random", "A+B minus A")},
        {"split": "Random", "effect": "Context", **primary_row("random", "A+B+C minus A+B")},
        {"split": "Strict sequence", "effect": "Site", **primary_row("sequence_cluster", "A+B minus A")},
        {
            "split": "Strict sequence", "effect": "Context",
            **primary_row("sequence_cluster", "A+B minus A+B+C", reverse=True),
        },
    ])
    panel_a["contrast"] = panel_a.effect.map({
        "Site": "A+B - A",
        "Context": "A+B+C - A+B",
    })

    secondary_plot = secondary.rename(columns={
        "Estimator": "estimator",
        "Contrast": "contrast",
        "Mean_Delta_R2": "mean",
        "BCa_CI_Lower": "low",
        "BCa_CI_Upper": "high",
    })[["estimator", "contrast", "mean", "low", "high"]]
    panel_a.assign(panel="A").to_csv(OUT / "incremental_primary_effects_source.csv", index=False)
    secondary_plot.assign(panel="B").to_csv(OUT / "incremental_estimator_effects_source.csv", index=False)
    combined_source = pd.concat([
        panel_a.rename(columns={
            "split": "category", "effect": "series", "contrast": "Contrast",
            "mean": "Mean_Delta_R2", "low": "BCa_CI_Lower", "high": "BCa_CI_Upper",
        }).assign(Panel="A", Estimator=""),
        secondary_plot.rename(columns={
            "estimator": "category", "contrast": "Contrast",
            "mean": "Mean_Delta_R2", "low": "BCa_CI_Lower", "high": "BCa_CI_Upper",
        }).assign(Panel="B", series=lambda frame: frame["Contrast"], Estimator=lambda frame: frame["category"]),
    ], ignore_index=True, sort=False)
    combined_source.to_csv(OUT / "incremental_paired_effects_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [0.88, 1.35]})

    ax = axes[0]
    splits = ["Random", "Strict sequence"]
    x = np.arange(len(splits))
    width = 0.32
    for effect, offset, color, hatch in [
        ("Site", -width / 2, SITE, "///"),
        ("Context", width / 2, CONTEXT, "\\\\"),
    ]:
        frame = panel_a[panel_a.effect == effect].set_index("split").loc[splits]
        ax.bar(
            x + offset, frame["mean"], width=width, color=color, alpha=0.92,
            hatch=hatch, edgecolor="white", linewidth=0.4, label=effect,
        )
        ax.errorbar(
            x + offset, frame["mean"],
            yerr=[frame["mean"] - frame.low, frame.high - frame["mean"]],
            fmt="none", ecolor="#30343B", capsize=2.5, capthick=0.75, linewidth=0.75,
        )
    ax.axhline(0, color="#333333", linewidth=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel(r"Mean paired $\Delta R^2$")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax, "y")
    panel_label(ax, "A")

    ax = axes[1]
    estimators = ["Ridge", "ElasticNet", "RBF-SVR", "Random Forest", "XGBoost"]
    x = np.arange(len(estimators))
    for contrast, color, marker, label, linestyle in [
        ("A+B - A", SITE, "o", "Site beyond Chemistry", "-"),
        ("A+B+C - A+B", CONTEXT, "s", "Context beyond Chemistry+Site", "--"),
    ]:
        frame = secondary_plot[secondary_plot.contrast == contrast].set_index("estimator").loc[estimators]
        ax.errorbar(
            x, frame["mean"],
            yerr=[frame["mean"] - frame.low, frame.high - frame["mean"]],
            color=color, marker=marker, markersize=4.5, linestyle=linestyle,
            linewidth=1.2, ecolor=color, elinewidth=0.7, capsize=2.2,
            label=label,
        )
    ax.axhline(0, color="#333333", linewidth=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators, rotation=20, ha="right")
    ax.set_ylabel(r"Mean paired $\Delta R^2$")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, "y")
    panel_label(ax, "B")

    save(fig, "figure_incremental_paired_effects")


def main():
    shap_top15()
    predicted_observed()
    if (RESULTS / "time_forward/model_performance_with_ci.csv").exists():
        time_forward()
    if (RESULTS / "scaffold_ranking/summary_with_ci.csv").exists():
        scaffold()
    if (RESULTS / "secondary_incremental_analysis/cross_estimator_paired_effects.csv").exists():
        incremental_effects()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
