#!/usr/bin/env python3
"""Recompute family pairwise accuracy from predictions and summarize it."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("CASPER_RESULTS_ROOT", ROOT / "results")) / "scaffold_ranking"


def family_metrics(frame):
    y = frame.y_true.to_numpy(dtype=float)
    pred = frame.y_pred.to_numpy(dtype=float)
    valid = label_ties = predicted_ties = concordant = 0
    score = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                label_ties += 1
                continue
            valid += 1
            if pred[i] == pred[j]:
                predicted_ties += 1
                score += 0.5
            elif (y[i] > y[j]) == (pred[i] > pred[j]):
                concordant += 1
                score += 1.0
    return {
        "valid_pair_count": valid,
        "correct_pair_count": score,
        "strictly_concordant_pair_count": concordant,
        "predicted_tie_count": predicted_ties,
        "label_tie_count": label_ties,
        "pairwise_accuracy": score / valid if valid else np.nan,
        "within_family_spearman": stats.spearmanr(y, pred).statistic,
    }


def main():
    predictions = pd.read_csv(OUT / "all_predictions.csv")
    old = pd.read_csv(OUT / "family_level_results.csv")
    metadata_columns = ["family_id", "source", "model", "n_historical", "n_support", "n_test"]
    metadata = old[metadata_columns].drop_duplicates(["family_id", "model"])
    rows = []
    for (family_id, model), frame in predictions.groupby(["family_id", "model"], sort=False):
        rows.append({"family_id": family_id, "model": model, **family_metrics(frame)})
    metrics = pd.DataFrame(rows)
    data = metadata.merge(metrics, on=["family_id", "model"], how="inner", validate="one_to_one")
    data.to_csv(OUT / "family_level_results.csv", index=False)

    rng = np.random.default_rng(20260717)
    summary_rows = []
    for model, frame in data.groupby("model"):
        values = frame.pairwise_accuracy.dropna().to_numpy()
        bootstrap = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(5000)])
        summary_rows.append({"model": model, "n_families_total": frame.family_id.nunique(), "n_families_with_valid_pairs": len(values), "overall_pairwise_accuracy": frame.correct_pair_count.sum() / frame.valid_pair_count.sum(), "mean_family_pairwise_accuracy": values.mean(), "family_bootstrap_ci95_low": np.quantile(bootstrap, 0.025), "family_bootstrap_ci95_high": np.quantile(bootstrap, 0.975), "bootstrap_replicates": 5000, "bootstrap_unit": "peptide family", "interval": "percentile", "prediction_tie_score": 0.5})
    pd.DataFrame(summary_rows).to_csv(OUT / "summary_with_ci.csv", index=False)

    chemistry = data[data.model == "Chemistry"][["family_id", "pairwise_accuracy"]]
    chemistry_site = data[data.model == "Chemistry + Site"][["family_id", "pairwise_accuracy"]]
    paired = chemistry_site.merge(chemistry, on="family_id", suffixes=("_AB", "_A"))
    paired["difference"] = paired.pairwise_accuracy_AB - paired.pairwise_accuracy_A
    valid = paired.difference.dropna().to_numpy()
    bootstrap = np.array([rng.choice(valid, len(valid), replace=True).mean() for _ in range(5000)])
    positive = int((valid > 0).sum())
    negative = int((valid < 0).sum())
    paired.to_csv(OUT / "paired_comparison_AB_vs_A_family_values.csv", index=False)
    pd.DataFrame([{"comparison": "Chemistry + Site minus Chemistry", "mean_difference": valid.mean(), "ci95_low": np.quantile(bootstrap, 0.025), "ci95_high": np.quantile(bootstrap, 0.975), "proportion_improved": (valid > 0).mean(), "proportion_tied": (valid == 0).mean(), "proportion_worsened": (valid < 0).mean(), "exact_sign_test_p": stats.binomtest(positive, positive + negative, 0.5).pvalue, "n_families_total": len(paired), "n_families_valid": len(valid), "bootstrap_replicates": 5000, "interval": "percentile", "prediction_tie_score": 0.5}]).to_csv(OUT / "paired_comparison_AB_vs_A.csv", index=False)
    print("scaffold statistics recomputed from saved predictions")


if __name__ == "__main__":
    main()
