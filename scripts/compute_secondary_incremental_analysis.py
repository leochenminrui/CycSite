#!/usr/bin/env python3
"""Secondary paired estimator analysis and exploratory SHAP sensitivity check.

This script consumes only the frozen 30-seed study outputs. It does not
fit or tune any model.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ESTIMATOR_INPUT = RESULTS / "estimator_matrix/seed_level_results.csv"
SHAP_INPUT = RESULTS / "shap/seed_level_group_attribution.csv"
FREEZE_INPUT = RESULTS / "RESULTS_MANIFEST.json"
OUT = RESULTS / "secondary_incremental_analysis"

ESTIMATORS = ["Ridge", "ElasticNet", "RBF-SVR", "Random Forest", "XGBoost"]
CONTRASTS = [
    ("A+B", "A", "A+B - A", "Site beyond Chemistry"),
    ("A+B+C", "A+B", "A+B+C - A+B", "Context beyond Chemistry+Site"),
]
EXPECTED_SEEDS = list(range(30))
BOOTSTRAP_RESAMPLES = 100_000
SIGNFLIP_RESAMPLES = 1_000_000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_input(path):
    manifest = json.loads(FREEZE_INPUT.read_text())
    key = str(path.relative_to(ROOT))
    expected = manifest["artifact_checksums"].get(key)
    if expected is None:
        raise RuntimeError(f"Input is not listed in frozen manifest: {key}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"Frozen checksum mismatch for {key}: expected {expected}, got {observed}"
        )
    return observed


def bootstrap_interval(values, seed):
    values = np.asarray(values, dtype=float)
    result = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=BOOTSTRAP_RESAMPLES,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def monte_carlo_signflip(values, seed):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    batch = 20_000
    while completed < SIGNFLIP_RESAMPLES:
        size = min(batch, SIGNFLIP_RESAMPLES - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += size
    return (exceed + 1) / (SIGNFLIP_RESAMPLES + 1)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def validate_estimator_matrix(data):
    required = {
        "representation", "estimator", "estimator_id", "seed", "r2", "source"
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"Missing estimator-matrix columns: {sorted(missing)}")
    if data.duplicated(["estimator", "representation", "seed"]).any():
        raise RuntimeError("Duplicate estimator/representation/seed rows detected")
    if len(data) != 5 * 7 * 30:
        raise RuntimeError(f"Expected 1050 estimator rows, found {len(data)}")

    for (estimator, representation), frame in data.groupby(
        ["estimator", "representation"]
    ):
        seeds = sorted(frame.seed.astype(int).tolist())
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"Seed mismatch for {estimator}/{representation}: {seeds}"
            )

    invalid_sources = []
    for row in data.itertuples():
        source = ROOT / row.source
        if not source.exists():
            invalid_sources.append(f"missing: {row.source}")
            continue
        payload = json.loads(source.read_text())
        if (
            payload.get("status") != "completed"
            or payload.get("n_trials") != 10
            or int(payload.get("seed")) != int(row.seed)
        ):
            invalid_sources.append(f"invalid: {row.source}")
    if invalid_sources:
        raise RuntimeError("Invalid estimator sources:\n" + "\n".join(invalid_sources))


def compute_estimator_effects(data):
    effect_rows = []
    paired_rows = []
    for estimator_index, estimator in enumerate(ESTIMATORS):
        frame = data[data.estimator == estimator]
        for contrast_index, (left, right, contrast, interpretation) in enumerate(
            CONTRASTS
        ):
            left_frame = (
                frame[frame.representation == left]
                .set_index("seed")
                .sort_index()
            )
            right_frame = (
                frame[frame.representation == right]
                .set_index("seed")
                .sort_index()
            )
            left_seeds = left_frame.index.astype(int).tolist()
            right_seeds = right_frame.index.astype(int).tolist()
            if left_seeds != EXPECTED_SEEDS or right_seeds != EXPECTED_SEEDS:
                raise RuntimeError(f"Unmatched seeds for {estimator}, {contrast}")

            differences = left_frame.r2.to_numpy() - right_frame.r2.to_numpy()
            bootstrap_seed = 20263000 + 10 * estimator_index + contrast_index
            signflip_seed = 20264000 + 10 * estimator_index + contrast_index
            ci_low, ci_high = bootstrap_interval(differences, bootstrap_seed)
            p_value = monte_carlo_signflip(differences, signflip_seed)
            effect_rows.append(
                {
                    "Estimator": estimator,
                    "Contrast": contrast,
                    "Interpretation": interpretation,
                    "Left_Representation": left,
                    "Right_Representation": right,
                    "Mean_Delta_R2": float(differences.mean()),
                    "BCa_CI_Lower": ci_low,
                    "BCa_CI_Upper": ci_high,
                    "MonteCarlo_SignFlip_P": p_value,
                    "N_Paired_Seeds": len(differences),
                    "N_Positive_Differences": int(np.count_nonzero(differences > 0)),
                    "N_Negative_Differences": int(np.count_nonzero(differences < 0)),
                    "N_Zero_Differences": int(np.count_nonzero(differences == 0)),
                    "Bootstrap_Resamples": BOOTSTRAP_RESAMPLES,
                    "SignFlip_Resamples": SIGNFLIP_RESAMPLES,
                    "Bootstrap_RNG_Seed": bootstrap_seed,
                    "SignFlip_RNG_Seed": signflip_seed,
                }
            )
            for seed, left_value, right_value, difference in zip(
                EXPECTED_SEEDS,
                left_frame.r2.to_numpy(),
                right_frame.r2.to_numpy(),
                differences,
            ):
                paired_rows.append(
                    {
                        "Estimator": estimator,
                        "Contrast": contrast,
                        "Seed": seed,
                        "Left_R2": left_value,
                        "Right_R2": right_value,
                        "Delta_R2": difference,
                    }
                )

    effects = pd.DataFrame(effect_rows)
    effects["Holm_Adjusted_P"] = holm(effects["MonteCarlo_SignFlip_P"])
    paired = pd.DataFrame(paired_rows)
    return effects, paired


def compute_shap_sensitivity():
    verify_frozen_input(SHAP_INPUT)
    shap = pd.read_csv(SHAP_INPUT)
    shap = shap[
        (shap.representation == "A+B+C") & shap.group.isin(["A", "B", "C"])
    ]
    if shap.duplicated(["seed", "group"]).any():
        raise RuntimeError("Duplicate SHAP seed/group rows detected")
    pivot = shap.pivot(index="seed", columns="group", values="attribution_proportion")
    if pivot.index.astype(int).tolist() != EXPECTED_SEEDS:
        raise RuntimeError("SHAP seed alignment failed")

    performance_rows = []
    for seed in EXPECTED_SEEDS:
        path = RESULTS / f"raw_runs/random/seed_{seed}/full_ABC_xgb/metrics.json"
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed" or payload.get("n_optuna_trials") != 50:
            raise RuntimeError(f"Invalid primary model metrics: {path}")
        performance_rows.append(
            {
                "seed": seed,
                "Validation_R2": payload["val_metrics"]["r2"],
                "Validation_RMSE": payload["val_metrics"]["rmse"],
                "Test_R2": payload["test_metrics"]["r2"],
                "Metrics_Source": str(path.relative_to(ROOT)),
            }
        )
    seed_level = pivot.join(pd.DataFrame(performance_rows).set_index("seed"))
    seed_level = seed_level.rename(
        columns={
            "A": "Chemistry_Attribution",
            "B": "Site_Attribution",
            "C": "Context_Attribution",
        }
    ).reset_index()

    rows = []
    for group, column in [
        ("Chemistry", "Chemistry_Attribution"),
        ("Site", "Site_Attribution"),
        ("Context", "Context_Attribution"),
    ]:
        association = stats.spearmanr(seed_level.Validation_R2, seed_level[column])
        rows.append(
            {
                "Performance_Metric": "validation R2",
                "Attribution_Group": group,
                "Spearman_Rho": float(association.statistic),
                "Two_Sided_Asymptotic_P": float(association.pvalue),
                "N_Seeds": len(seed_level),
                "Multiplicity_Adjusted": False,
                "Analysis_Label": "exploratory performance-conditioned SHAP sensitivity",
            }
        )
    return pd.DataFrame(rows), seed_level


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    estimator_checksum = verify_frozen_input(ESTIMATOR_INPUT)
    estimator_data = pd.read_csv(ESTIMATOR_INPUT)
    validate_estimator_matrix(estimator_data)
    effects, paired = compute_estimator_effects(estimator_data)
    shap_associations, shap_seed_level = compute_shap_sensitivity()

    effects.to_csv(OUT / "cross_estimator_paired_effects.csv", index=False)
    paired.to_csv(OUT / "cross_estimator_paired_seed_differences.csv", index=False)
    shap_associations.to_csv(OUT / "shap_performance_associations.csv", index=False)
    shap_seed_level.to_csv(OUT / "shap_performance_seed_level.csv", index=False)

    metadata = {
        "analysis_label": "secondary cross-estimator robustness analysis",
        "estimator_input": str(ESTIMATOR_INPUT.relative_to(ROOT)),
        "estimator_input_sha256": estimator_checksum,
        "shap_input": str(SHAP_INPUT.relative_to(ROOT)),
        "shap_input_sha256": sha256(SHAP_INPUT),
        "estimators": ESTIMATORS,
        "contrasts": [contrast for _, _, contrast, _ in CONTRASTS],
        "paired_seeds": EXPECTED_SEEDS,
        "multiplicity_family": "all 10 estimator-by-contrast tests",
        "bootstrap": {
            "method": "BCa over paired seed differences",
            "confidence_level": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "sign_flip": {
            "method": "two-sided Monte Carlo sign-flip on paired differences",
            "resamples": SIGNFLIP_RESAMPLES,
            "add_one_correction": True,
        },
        "shap_sensitivity": {
            "label": "exploratory",
            "performance_metric": "validation R2",
            "method": "Spearman association with normalized group attribution",
            "multiplicity_adjusted": False,
        },
    }
    (OUT / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    audit = f"""# Secondary-analysis audit

## Frozen inputs

- `{ESTIMATOR_INPUT.relative_to(ROOT)}` (SHA-256 `{estimator_checksum}`)
- `{SHAP_INPUT.relative_to(ROOT)}` (SHA-256 `{sha256(SHAP_INPUT)}`)
- Primary complete-model validation metrics from `results/raw_runs/random/seed_*/full_ABC_xgb/metrics.json`

## Analysis and alignment checks

- Script: `scripts/compute_secondary_incremental_analysis.py`
- Estimator matrix: 5 estimators x 7 representations x 30 seeds = {len(estimator_data)} completed rows
- New paired family: 5 estimators x 2 contrasts = {len(effects)} tests
- Contrasts: A+B - A; A+B+C - A+B
- Paired seed IDs: 0-29 for every estimator and contrast; no duplicates or missing seeds
- Source-run validation: every referenced estimator run had `status=completed` and 10 Optuna trials
- Confidence intervals: 95% BCa bootstrap over paired seed differences, {BOOTSTRAP_RESAMPLES:,} resamples
- Tests: two-sided Monte Carlo sign-flip, {SIGNFLIP_RESAMPLES:,} resamples, add-one correction
- Multiplicity: Holm correction across the full family of {len(effects)} new tests
- XAI sensitivity: exploratory Spearman associations between validation R2 and normalized A/B/C attribution over {len(shap_seed_level)} existing complete-representation models; no new model fitting

## Generated outputs

- `results/secondary_incremental_analysis/cross_estimator_paired_effects.csv`
- `results/secondary_incremental_analysis/cross_estimator_paired_seed_differences.csv`
- `results/secondary_incremental_analysis/shap_performance_associations.csv`
- `results/secondary_incremental_analysis/shap_performance_seed_level.csv`
- `results/secondary_incremental_analysis/analysis_metadata.json`
"""
    (RESULTS / "SECONDARY_ANALYSIS_AUDIT.md").write_text(audit)
    print(effects.to_string(index=False))
    print("\n" + shap_associations.to_string(index=False))


if __name__ == "__main__":
    main()
