# Secondary-analysis audit

## Frozen inputs

- `results/estimator_matrix/seed_level_results.csv` (SHA-256 `23ae92ff4ad99959592deef96016c938eebcebc37862a5591e4a794e137dfb12`)
- `results/shap/seed_level_group_attribution.csv` (SHA-256 `a3488568f83df882e60ebd3fdc0cf768b5be37818708c0772cccae6fa093aa3a`)
- Primary complete-model validation metrics from `results/raw_runs/random/seed_*/full_ABC_xgb/metrics.json`

## Analysis and alignment checks

- Script: `scripts/compute_secondary_incremental_analysis.py`
- Estimator matrix: 5 estimators x 7 representations x 30 seeds = 1050 completed rows
- New paired family: 5 estimators x 2 contrasts = 10 tests
- Contrasts: A+B - A; A+B+C - A+B
- Paired seed IDs: 0-29 for every estimator and contrast; no duplicates or missing seeds
- Source-run validation: every referenced estimator run had `status=completed` and 10 Optuna trials
- Confidence intervals: 95% BCa bootstrap over paired seed differences, 100,000 resamples
- Tests: two-sided Monte Carlo sign-flip, 1,000,000 resamples, add-one correction
- Multiplicity: Holm correction across the full family of 10 new tests
- XAI sensitivity: exploratory Spearman associations between validation R2 and normalized A/B/C attribution over 30 existing complete-representation models; no new model fitting

## Generated outputs

- `results/secondary_incremental_analysis/cross_estimator_paired_effects.csv`
- `results/secondary_incremental_analysis/cross_estimator_paired_seed_differences.csv`
- `results/secondary_incremental_analysis/shap_performance_associations.csv`
- `results/secondary_incremental_analysis/shap_performance_seed_level.csv`
- `results/secondary_incremental_analysis/analysis_metadata.json`
