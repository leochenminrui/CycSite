# Reproducing the CycSite analyses

Run commands from the repository root using the Python 3.12 environment in the
[README](../README.md#quick-start). The pinned packages in
[`requirements.txt`](../requirements.txt) are shared by descriptor construction,
training, evaluation, and plotting. CPU execution is supported; a GPU is not
required by the supplied configuration.

## Reference results

The archive inspected for this documentation is repository commit
`7661b9a6964f7fe8262442dc03963c20a5b70eed`. Its
[`RESULTS_MANIFEST.json`](../results/RESULTS_MANIFEST.json) records the earlier
computation revision, Python/platform information, completion counts, and
SHA-256 hashes of the data, fixed partitions, and required result tables.
The documentation update preserves those scientific inputs and results.

The current archive contains a 30-seed benchmark and subsequent sensitivity
analyses. The [Research Square preprint v1](https://doi.org/10.21203/rs.3.rs-9541490/v1)
describes an earlier analysis version; its headline values must not be
substituted for the archived multi-seed estimates. Journal acceptance is the
paper's publication status, not a claim that this archive exactly matches a
publisher-hosted version of record.

Useful reference files are:

| Output | Entry point | Reference |
|---|---|---|
| Primary and estimator summaries | `scripts/compute_statistics.py` | `results/summary_with_ci.csv`, `results/paired_comparisons.csv`, `results/estimator_matrix/summary_with_ci.csv` |
| Compact project summary | `scripts/summarize_results.py` | `results/summary.json` |
| Exact-sequence grouped bootstrap | `scripts/compute_grouped_paired_bootstrap.py` | `results/grouped_paired_bootstrap/summary.csv` |
| Common no-Mono test comparison | `scripts/compute_common_no_mono_test_comparison.py` | `results/common_no_mono_test/summary.csv` |
| Time-forward diagnostics | `scripts/finalize_time_forward_statistics.py` | `results/time_forward/cutoff_associations.csv` |
| Peptide-family ranking | `scripts/finalize_scaffold_statistics.py` | `results/scaffold_ranking/summary_with_ci.csv` |
| Position-summary sensitivity | `scripts/summarize_position_summary_ablation.py` | `results/position_summary_ablation/summary.csv` |
| Cross-estimator paired effects and SHAP sensitivity | `scripts/compute_secondary_incremental_analysis.py` | `results/secondary_incremental_analysis/` |
| Figures from saved estimates | `scripts/generate_figures.py` | Newly generated PNG/PDF files in `results/figures/` |

The random partition contains 5,056/1,084/1,084 train/validation/test records.
The strict sequence-component partition connects sequences at identity ≥0.70
and assigns connected components intact. Its dominant component prevents a
70/15/15 allocation: the actual sizes are 7,030/97/97, with maximum
cross-partition identity 0.6923. The two partitions answer different evaluation
questions; compare models within the same fixed partition.

### Compact summary consistency

At the archived revision above, the `scaffold_ranking` section of
`results/summary.json` differs from the source tables read by
`scripts/summarize_results.py`:

- `results/scaffold_ranking/summary_with_ci.csv`: all five models have different
  `overall_pairwise_accuracy`, `mean_family_pairwise_accuracy`,
  `family_bootstrap_ci95_low`, and `family_bootstrap_ci95_high` values. The CSV
  also contains `prediction_tie_score=0.5`, which is absent from the compact JSON.
- `results/scaffold_ranking/paired_comparison_AB_vs_A.csv`: `mean_difference`,
  `ci95_low`, `ci95_high`, `proportion_tied`, and `proportion_worsened` differ.
  The CSV's `prediction_tie_score=0.5` field is absent from the compact JSON.

For example, Chemistry's mean family pairwise accuracy is `0.6394603771` in the
compact JSON and `0.6349009823` in the current CSV. The paired Chemistry + Site
minus Chemistry mean difference is `0.0036099563` in the JSON and `0.0074549068`
in the CSV. Rebuilding the compact summary reflects the supplied source CSVs;
its scaffold section therefore differs from the frozen JSON. The other
top-level summary sections reproduce from their saved tables.

Both representations and their existing checksum-backed manifest are preserved
for traceability. Use the explicitly named source table when reporting a
scaffold-ranking estimate, and identify the revision and file used. This
documentation does not select a replacement scientific reference or change
either set of results. The README's primary R² table is independently checked
against the fixed test labels and per-seed predictions and is unaffected.

## Check the frozen inputs and summaries

This standard-library command verifies 58 recorded file hashes without changing
the archive:

```bash
python3 scripts/check_release.py
```

Expected status: `CHECKSUMS_MATCH`. This checks the files covered by the frozen
manifest; it does not retrain models or validate every individual run file.

To check completion counts and regenerate the manifest for a working copy:

```bash
.venv/bin/python scripts/verify_results.py
```

Expected counts are 600 primary runs, 1,050 estimator-matrix runs, 180
Mono-exclusion runs, 30 position-summary runs, 720 SHAP seed/group rows,
8 time cutoffs, and 49 peptide families. This command **writes** the manifest
with the current revision and hashes, so run the frozen checksum check first.

## Rebuild summaries from saved results

These commands use completed results and do not retrain the benchmark:

```bash
.venv/bin/python scripts/summarize_results.py
.venv/bin/python scripts/compute_grouped_paired_bootstrap.py
.venv/bin/python scripts/compute_common_no_mono_test_comparison.py
.venv/bin/python scripts/summarize_position_summary_ablation.py
.venv/bin/python scripts/finalize_time_forward_statistics.py
.venv/bin/python scripts/finalize_scaffold_statistics.py
.venv/bin/python scripts/compute_secondary_incremental_analysis.py
```

The scripts overwrite the corresponding summary tables. Use a separate checkout
for recomputation and compare its numeric columns with the untouched archive.
The primary statistical aggregation uses 100,000 BCa bootstrap resamples and
1,000,000 sign-flip resamples; these stages take more work than the checksum
check. Preserve seed pairing, held-out IDs, metric definitions, and multiplicity
families when comparing results. Numeric comparison is appropriate for rebuilt
CSV/JSON files because timestamps, revision fields, and serialization may change.

Generate figures using the bundled estimates and predictions:

```bash
.venv/bin/python scripts/generate_figures.py
```

Outputs are figure PNGs and PDFs under `results/figures/`, which is ignored by
Git. The plotting script uses Matplotlib's bundled DejaVu Sans font. These
figures rebuild saved estimates; generating them does not establish that a
training rerun reproduces the same predictions.

## Audit the dataset and partitions

Source data and derived records are already supplied; see
[data provenance and terms](data.md) before reusing them.

```bash
.venv/bin/python scripts/audit_dataset.py
.venv/bin/python scripts/audit_descriptor_finiteness.py
.venv/bin/python scripts/create_sequence_cluster_split.py
.venv/bin/python scripts/create_mono_exclusion_splits.py
```

The audit writes inventories under `results/audits/`. Partition regeneration
writes `data/splits/`; preserve the supplied fixed files before running it.
The sequence-component audit evaluates sequence pairs and can take longer than
loading the fixed partitions. The finiteness audit recalculates Chemistry and
RDKit 2D descriptors for all 7,224 retained records.

## Fit the archived benchmark configurations

The primary, estimator-matrix, Mono-exclusion, and position-summary training
scripts resume valid completed runs. On the bundled archive, those existing
runs are normally reused. The time-forward and scaffold-ranking scripts refit
their models on each invocation. To refit every benchmark run, make a separate
checkout and move the corresponding saved run directories out of that checkout
before execution, while keeping an untouched reference copy for comparison.
No fixed pretrained inference checkpoint is distributed; the saved predictions
are the evaluation evidence, and training code fits the models.

```bash
CASPER_WORKERS=8 .venv/bin/python scripts/run_primary_benchmark.py
CASPER_ESTIMATOR_WORKERS=8 CASPER_ESTIMATOR_THREADS=2 \
  .venv/bin/python scripts/run_estimator_matrix.py
CASPER_WORKERS=8 .venv/bin/python scripts/run_mono_exclusion_sensitivity.py
CASPER_WORKERS=8 .venv/bin/python scripts/run_position_summary_ablation.py
CASPER_MODEL_THREADS=2 .venv/bin/python scripts/run_time_forward.py
CASPER_MODEL_THREADS=2 .venv/bin/python scripts/run_scaffold_ranking.py
```

Primary and sensitivity XGBoost runs use 50 Optuna trials per seed. The estimator
matrix uses 10 trials per representation/estimator/seed. The default primary
configuration caps workers at eight and sets two model threads; reduce the
worker environment variables for machines with fewer CPU cores or less memory.
Runtime and peak RAM depend on the hardware and search outcome; the repository
does not provide a measured hardware-independent estimate. These commands
perform substantial model fitting and are separate from the quick verification.

The `CASPER_*` environment variables are retained configuration names from the
research implementation; use these exact names with this release.

After model fitting, regenerate the summaries and explanations in this order:

```bash
.venv/bin/python scripts/compute_statistics.py
.venv/bin/python scripts/compute_mono_test_sensitivity.py
.venv/bin/python scripts/summarize_position_summary_ablation.py
.venv/bin/python scripts/finalize_time_forward_statistics.py
.venv/bin/python scripts/finalize_scaffold_statistics.py
.venv/bin/python scripts/run_shap.py
.venv/bin/python scripts/compute_secondary_incremental_analysis.py
.venv/bin/python scripts/compute_grouped_paired_bootstrap.py
.venv/bin/python scripts/compute_common_no_mono_test_comparison.py
.venv/bin/python scripts/summarize_results.py
.venv/bin/python scripts/verify_results.py
```

`run_shap.py` fits the models needed for explanations using saved tuning records;
it is not a saved-table-only summary step. Prediction reruns may vary across
platforms, so assess metrics and paired intervals against the supplied numeric
references rather than changing the reference values to match a rerun.
