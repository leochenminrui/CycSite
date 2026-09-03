#!/usr/bin/env python3
"""Aggregate the 30-seed benchmark and paired inference."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RAW = OUT / "raw_runs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = range(30)
METRICS = ["r2", "rmse", "mae", "spearman"]
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
ESTIMATOR_FEATURES = {
    "A": "Chem",
    "B": "Site",
    "C": "Context",
    "A+B": "Chem_Site",
    "A+C": "Chem_Context",
    "B+C": "Site_Context",
    "A+B+C": "Chem_Site_Context",
}
ESTIMATORS = {
    "Ridge": "ridge",
    "ElasticNet": "elasticnet",
    "Random Forest": "random_forest",
    "RBF-SVR": "svr",
    "XGBoost": "xgboost",
}


def t_interval(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sem = stats.sem(values)
    half = stats.t.ppf(0.975, len(values) - 1) * sem
    return mean, values.std(ddof=1), mean - half, mean + half


def bootstrap_interval(values, seed):
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0], rtol=0.0, atol=1e-15):
        return float(values[0]), float(values[0])
    result = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=100_000,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    return result.confidence_interval.low, result.confidence_interval.high


def monte_carlo_signflip(values, seed, n_resamples=1_000_000):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    batch = 20_000
    while done < n_resamples:
        size = min(batch, n_resamples - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        done += size
    return (exceed + 1) / (n_resamples + 1)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def load_seed_results():
    rows = []
    for split in ("random", "sequence_cluster"):
        for representation, model_id in MODELS.items():
            for seed in SEEDS:
                path = RAW / split / f"seed_{seed}" / model_id / "metrics.json"
                payload = json.loads(path.read_text())
                if payload.get("status") != "completed":
                    raise RuntimeError(f"Incomplete run: {path}")
                if payload.get("n_optuna_trials") != 50:
                    raise RuntimeError(f"Wrong trial count: {path}")
                rows.append(
                    {
                        "split": split,
                        "representation": representation,
                        "model_id": model_id,
                        "seed": seed,
                        **{metric: payload["test_metrics"][metric] for metric in METRICS},
                        "runtime_seconds": payload.get("runtime_seconds"),
                        "prediction_file": str(
                            path.with_name("predictions.csv").relative_to(ROOT)
                        ),
                        "best_params_json": json.dumps(
                            payload.get("best_params", {}), sort_keys=True
                        ),
                    }
                )
    return pd.DataFrame(rows)


def load_estimator_results():
    rows = []
    base = OUT / "estimator_matrix/raw_runs"
    for representation, safe_feature in ESTIMATOR_FEATURES.items():
        for estimator, estimator_id in ESTIMATORS.items():
            for seed in SEEDS:
                path = base / f"{safe_feature}_{estimator_id}" / f"seed_{seed}" / "metrics.json"
                payload = json.loads(path.read_text())
                if payload.get("status") != "completed" or payload.get("n_trials") != 10:
                    raise RuntimeError(f"Incomplete estimator run: {path}")
                rows.append(
                    {
                        "representation": representation,
                        "estimator": estimator,
                        "estimator_id": estimator_id,
                        "seed": seed,
                        **{metric: payload["test_metrics"][metric] for metric in METRICS},
                        "runtime_seconds": payload.get("runtime_seconds"),
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def main():
    seed_results = load_seed_results()
    seed_results.to_csv(OUT / "seed_level_results.csv", index=False)

    summaries = []
    for (split, representation), group in seed_results.groupby(
        ["split", "representation"], sort=False
    ):
        for metric_index, metric in enumerate(METRICS):
            values = group.sort_values("seed")[metric].to_numpy()
            mean, sample_sd, t_low, t_high = t_interval(values)
            boot_low, boot_high = bootstrap_interval(
                values, 20260902 + metric_index
            )
            summaries.append(
                {
                    "split": split,
                    "representation": representation,
                    "metric": metric,
                    "mean": mean,
                    "sample_sd": sample_sd,
                    "t_ci95_low": t_low,
                    "t_ci95_high": t_high,
                    "bootstrap_bca_ci95_low": boot_low,
                    "bootstrap_bca_ci95_high": boot_high,
                    "n_seeds": len(values),
                    "bootstrap_resamples": 100_000,
                }
            )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUT / "summary_with_ci.csv", index=False)

    contrasts = [
        ("random", "A+B+C", "A"),
        ("random", "A+B", "A"),
        ("random", "A+B+C", "A+B"),
        ("sequence_cluster", "A+B", "A"),
        ("sequence_cluster", "A+B", "A+B+C"),
        ("sequence_cluster", "A+B", "ECFP"),
    ]
    rows = []
    for contrast_index, (split, left, right) in enumerate(contrasts):
        for metric_index, metric in enumerate(METRICS):
            left_values = (
                seed_results.query(
                    "split == @split and representation == @left"
                )
                .sort_values("seed")[metric]
                .to_numpy()
            )
            right_values = (
                seed_results.query(
                    "split == @split and representation == @right"
                )
                .sort_values("seed")[metric]
                .to_numpy()
            )
            differences = left_values - right_values
            mean, sample_sd, t_low, t_high = t_interval(differences)
            boot_low, boot_high = bootstrap_interval(
                differences, 20261000 + 10 * contrast_index + metric_index
            )
            pvalue = monte_carlo_signflip(
                differences, 20262000 + 10 * contrast_index + metric_index
            )
            rows.append(
                {
                    "split": split,
                    "contrast": f"{left} minus {right}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "mean_difference": mean,
                    "sample_sd": sample_sd,
                    "t_ci95_low": t_low,
                    "t_ci95_high": t_high,
                    "bootstrap_bca_ci95_low": boot_low,
                    "bootstrap_bca_ci95_high": boot_high,
                    "p_value_signflip_mc": pvalue,
                    "signflip_resamples": 1_000_000,
                    "n_pairs": len(differences),
                    "seed_differences_json": json.dumps(differences.tolist()),
                }
            )
    paired = pd.DataFrame(rows)
    for metric in METRICS:
        mask = paired.metric == metric
        paired.loc[mask, "p_value_holm"] = holm(
            paired.loc[mask, "p_value_signflip_mc"].to_numpy()
        )
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)

    estimator_results = load_estimator_results()
    estimator_results.to_csv(OUT / "estimator_matrix/seed_level_results.csv", index=False)
    estimator_summaries = []
    for (estimator, representation), group in estimator_results.groupby(
        ["estimator", "representation"], sort=False
    ):
        for metric_index, metric in enumerate(METRICS):
            values = group.sort_values("seed")[metric].to_numpy()
            mean, sample_sd, t_low, t_high = t_interval(values)
            boot_low, boot_high = bootstrap_interval(values, 20263000 + metric_index)
            estimator_summaries.append(
                {
                    "estimator": estimator,
                    "representation": representation,
                    "metric": metric,
                    "mean": mean,
                    "sample_sd": sample_sd,
                    "t_ci95_low": t_low,
                    "t_ci95_high": t_high,
                    "bootstrap_bca_ci95_low": boot_low,
                    "bootstrap_bca_ci95_high": boot_high,
                    "n_seeds": len(values),
                }
            )
    estimator_summary_frame = pd.DataFrame(estimator_summaries)
    estimator_summary_frame.to_csv(
        OUT / "estimator_matrix/summary_with_ci.csv", index=False
    )

    rf = estimator_results.query(
        "estimator == 'Random Forest' and representation == 'A+B+C'"
    ).sort_values("seed").r2.to_numpy()
    xgb = estimator_results.query(
        "estimator == 'XGBoost' and representation == 'A+B+C'"
    ).sort_values("seed").r2.to_numpy()
    difference = rf - xgb
    mean, sample_sd, t_low, t_high = t_interval(difference)
    boot_low, boot_high = bootstrap_interval(difference, 20264000)
    rf_vs_xgb = pd.DataFrame(
        [
            {
                "contrast": "Random Forest minus XGBoost for A+B+C",
                "metric": "r2",
                "mean_difference": mean,
                "sample_sd": sample_sd,
                "t_ci95_low": t_low,
                "t_ci95_high": t_high,
                "bootstrap_bca_ci95_low": boot_low,
                "bootstrap_bca_ci95_high": boot_high,
                "p_value_signflip_mc": monte_carlo_signflip(difference, 20264001),
                "signflip_resamples": 1_000_000,
                "n_pairs": len(difference),
            }
        ]
    )
    rf_vs_xgb.to_csv(OUT / "estimator_matrix/rf_vs_xgb_paired.csv", index=False)
    r2_summary = summary_frame[summary_frame.metric == "r2"]
    key_results = {
        "primary_leaders": {
            split: frame.loc[frame["mean"].idxmax()].to_dict()
            for split, frame in r2_summary.groupby("split")
        },
        "primary_r2": r2_summary.to_dict("records"),
        "paired_r2": paired[paired.metric == "r2"].drop(
            columns=["seed_differences_json"]
        ).to_dict("records"),
        "estimator_leaders": {
            estimator: frame.loc[frame["mean"].idxmax()].to_dict()
            for estimator, frame in estimator_summary_frame[
                estimator_summary_frame.metric == "r2"
            ].groupby("estimator")
        },
        "rf_vs_xgb_complete": rf_vs_xgb.iloc[0].to_dict(),
    }
    (OUT / "key_results.json").write_text(
        json.dumps(key_results, indent=2, default=float)
    )
    print(
        json.dumps(
            {
                "seed_rows": len(seed_results),
                "summary_rows": len(summaries),
                "paired_rows": len(paired),
                "estimator_seed_rows": len(estimator_results),
                "estimator_summary_rows": len(estimator_summaries),
            }
        )
    )


if __name__ == "__main__":
    main()
