#!/usr/bin/env python3
"""Evaluate primary predictions on Mono-placeholder and unaffected test subsets.

This is deliberately a test-subset sensitivity analysis: models are not
retrained, so the output quantifies whether held-out performance estimates are
driven by records whose backbone parser skipped an undefined ``Mono*`` token.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.evaluation import compute_all_metrics
from scripts.compute_statistics import bootstrap_interval, monte_carlo_signflip

RESULTS = ROOT / "results"
OUT = RESULTS / "audits"
MODELS = {
    "Sequence": "seq_aa_xgb",
    "A": "chem_A_xgb",
    "B": "site_B_xgb",
    "C": "context_C_xgb",
    "A+B": "chem_site_AB_xgb",
    "A+C": "chem_context_AC_xgb",
    "B+C": "site_context_BC_xgb",
    "A+B+C": "full_ABC_xgb",
    "RDKit 2D": "rdkit_full_xgb",
    "ECFP": "ecfp_xgb",
}


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return mean, values.std(ddof=1), mean - half, mean + half


def main():
    audit = pd.read_csv(OUT / "mono_placeholder_records.csv")
    mono_ids = set(audit.loc[audit.retained_for_modeling, "sample_id"])
    rows = []
    for split in ("random", "sequence_cluster"):
        for representation, model_id in MODELS.items():
            for seed in range(30):
                path = RESULTS / "raw_runs" / split / f"seed_{seed}" / model_id / "predictions.csv"
                predictions = pd.read_csv(path)
                predictions["contains_mono_placeholder"] = predictions.sample_id.isin(mono_ids)
                for subset, frame in (
                    ("all_test_records", predictions),
                    ("no_mono_placeholder", predictions[~predictions.contains_mono_placeholder]),
                    ("mono_placeholder", predictions[predictions.contains_mono_placeholder]),
                ):
                    if len(frame) < 3:
                        continue
                    metrics = compute_all_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
                    rows.append(
                        {
                            "split": split,
                            "representation": representation,
                            "model_id": model_id,
                            "seed": seed,
                            "subset": subset,
                            "n_test": len(frame),
                            **{name: metrics[name] for name in ("r2", "rmse", "mae", "spearman")},
                        }
                    )
    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(OUT / "mono_test_subset_seed_level_metrics.csv", index=False)

    summaries = []
    for keys, frame in seed_level.groupby(["split", "representation", "subset"], sort=False):
        for metric in ("r2", "rmse", "mae", "spearman"):
            mean, sample_sd, low, high = interval(frame[metric])
            summaries.append(
                {
                    "split": keys[0],
                    "representation": keys[1],
                    "subset": keys[2],
                    "metric": metric,
                    "n_test": int(frame.n_test.iloc[0]),
                    "n_seeds": len(frame),
                    "mean": mean,
                    "sample_sd": sample_sd,
                    "t_ci95_low": low,
                    "t_ci95_high": high,
                }
            )
    pd.DataFrame(summaries).to_csv(OUT / "mono_test_subset_summary.csv", index=False)

    contrast_rows = []
    r2 = seed_level[seed_level.subset.isin(["all_test_records", "no_mono_placeholder"])]
    for split in ("random", "sequence_cluster"):
        for subset in ("all_test_records", "no_mono_placeholder"):
            frame = r2[(r2.split == split) & (r2.subset == subset)]
            for contrast_index, (left, right) in enumerate(
                (("A+B", "A"), ("A+B+C", "A+B"))
            ):
                left_values = frame[frame.representation == left].sort_values("seed").r2.to_numpy()
                right_values = frame[frame.representation == right].sort_values("seed").r2.to_numpy()
                differences = left_values - right_values
                low, high = bootstrap_interval(
                    differences,
                    20269000 + 100 * (split == "sequence_cluster")
                    + 10 * (subset == "no_mono_placeholder") + contrast_index,
                )
                contrast_rows.append(
                    {
                        "split": split,
                        "subset": subset,
                        "contrast": f"{left} minus {right}",
                        "mean_difference_r2": differences.mean(),
                        "bootstrap_bca_ci95_low": low,
                        "bootstrap_bca_ci95_high": high,
                        "p_value_signflip_mc": monte_carlo_signflip(
                            differences,
                            20269500 + 100 * (split == "sequence_cluster")
                            + 10 * (subset == "no_mono_placeholder") + contrast_index,
                        ),
                        "n_pairs": len(differences),
                    }
                )
    pd.DataFrame(contrast_rows).to_csv(OUT / "mono_test_subset_paired_contrasts.csv", index=False)

    metadata = {
        "analysis_type": "fixed-model held-out test-subset sensitivity; no retraining",
        "mono_placeholder_definition": "raw Sequence token starts with 'Mono'",
        "retained_records_with_mono_placeholder": len(mono_ids),
        "limitations": (
            "This analysis removes affected records only when calculating held-out metrics; "
            "it does not remove Mono-containing records from model fitting."
        ),
    }
    (OUT / "mono_test_subset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"seed_rows": len(seed_level), "summary_rows": len(summaries), "contrasts": len(contrast_rows)}))


if __name__ == "__main__":
    main()
