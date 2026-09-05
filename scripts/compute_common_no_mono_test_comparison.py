#!/usr/bin/env python3
"""Compare retained-data and Mono-excluded training on identical no-Mono tests."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.evaluation import compute_all_metrics
from scripts.compute_statistics import bootstrap_interval, monte_carlo_signflip

RESULTS = ROOT / "results"
OUT = RESULTS / "common_no_mono_test"
MODELS = {"A": "chem_A_xgb", "A+B": "chem_site_AB_xgb", "A+B+C": "full_ABC_xgb"}


def read_full_trained(split, seed, model_id, excluded_ids):
    path = RESULTS / "raw_runs" / split / f"seed_{seed}" / model_id / "predictions.csv"
    frame = pd.read_csv(path)
    return frame.loc[~frame.sample_id.isin(excluded_ids)].sort_values("sample_id").reset_index(drop=True)


def read_exclusion_trained(split, seed, model_id):
    path = (
        RESULTS / "mono_exclusion_sensitivity/raw_runs" / f"{split}_no_mono"
        / f"seed_{seed}" / model_id / "predictions.csv"
    )
    return pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(RESULTS / "audits/mono_placeholder_records.csv")
    excluded_ids = set(audit.loc[audit.retained_for_modeling, "sample_id"])
    rows = []
    alignment = []
    for split in ("random", "sequence_cluster"):
        for representation, model_id in MODELS.items():
            for seed in range(30):
                full = read_full_trained(split, seed, model_id, excluded_ids)
                excluded = read_exclusion_trained(split, seed, model_id)
                same_ids = full.sample_id.equals(excluded.sample_id)
                same_labels = np.allclose(full.y_true, excluded.y_true)
                if not same_ids or not same_labels:
                    raise ValueError(
                        f"Common-test mismatch for {split}, {representation}, seed {seed}"
                    )
                alignment.append(
                    {
                        "split": split,
                        "representation": representation,
                        "seed": seed,
                        "n_test": len(full),
                        "sample_ids_identical": same_ids,
                        "labels_identical": same_labels,
                    }
                )
                for scheme, frame in (
                    ("full retained training", full),
                    ("Mono-excluded training", excluded),
                ):
                    metrics = compute_all_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
                    rows.append(
                        {
                            "split": split,
                            "training_scheme": scheme,
                            "representation": representation,
                            "seed": seed,
                            "n_test": len(frame),
                            **{metric: metrics[metric] for metric in ("r2", "rmse", "mae")},
                        }
                    )
    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(OUT / "seed_level_metrics.csv", index=False)
    pd.DataFrame(alignment).to_csv(OUT / "alignment_checks.csv", index=False)

    summaries = []
    for keys, frame in seed_level.groupby(
        ["split", "training_scheme", "representation"], sort=False
    ):
        for metric_index, metric in enumerate(("r2", "rmse", "mae")):
            values = frame.sort_values("seed")[metric].to_numpy()
            low, high = bootstrap_interval(
                values,
                20261000
                + 100 * (keys[0] == "sequence_cluster")
                + 10 * (keys[1] == "Mono-excluded training")
                + metric_index,
            )
            summaries.append(
                {
                    "split": keys[0],
                    "training_scheme": keys[1],
                    "representation": keys[2],
                    "metric": metric,
                    "n_test": int(frame.n_test.iloc[0]),
                    "n_seeds": len(values),
                    "mean": values.mean(),
                    "bootstrap_bca_ci95_low": low,
                    "bootstrap_bca_ci95_high": high,
                }
            )
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)

    contrasts = []
    for split_index, split in enumerate(("random", "sequence_cluster")):
        for scheme_index, scheme in enumerate(("full retained training", "Mono-excluded training")):
            frame = seed_level[(seed_level.split == split) & (seed_level.training_scheme == scheme)]
            for metric_index, metric in enumerate(("r2", "rmse", "mae")):
                a = frame[frame.representation == "A"].sort_values("seed")[metric].to_numpy()
                ab = frame[frame.representation == "A+B"].sort_values("seed")[metric].to_numpy()
                differences = ab - a
                low, high = bootstrap_interval(
                    differences, 20262000 + 100 * split_index + 10 * scheme_index + metric_index
                )
                contrasts.append(
                    {
                        "split": split,
                        "training_scheme": scheme,
                        "comparison": "A+B minus A",
                        "metric": metric,
                        "mean_difference": differences.mean(),
                        "bootstrap_bca_ci95_low": low,
                        "bootstrap_bca_ci95_high": high,
                        "p_value_signflip_mc": monte_carlo_signflip(
                            differences,
                            20263000 + 100 * split_index + 10 * scheme_index + metric_index,
                        ),
                        "n_pairs": len(differences),
                    }
                )
    pd.DataFrame(contrasts).to_csv(OUT / "paired_contrasts.csv", index=False)
    metadata = {
        "analysis_type": "common held-out no-Mono test comparison",
        "test_definition": "records in the full fixed test partition that contain no raw token beginning with Mono",
        "training_schemes": [
            "models optimized on the full retained train/validation partitions",
            "models independently optimized after Mono-placeholder records were removed from train/validation partitions",
        ],
        "limitations": (
            "Using an identical test set removes test-composition differences, but the training and validation "
            "samples and the resulting hyperparameter optimization still differ between schemes."
        ),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()
