# CycSite

Code and data for **Site-Conditioned Edit Chemistry for Cyclic Peptide
Permeability Modeling**.

## Contents

| Path | Purpose |
|---|---|
| `data/raw/` | CycPeptMPDB source CSV files used by the study; reuse remains subject to the source database terms |
| `data/processed/` | Normalized dataset containing 7,224 records |
| `data/splits/` | Fixed random, strict sequence-component, and Mono-exclusion splits |
| `src/` | Dataset schema, descriptor construction, model tuning, and evaluation code |
| `scripts/` | Data audit, experiments, statistics, SHAP, temporal, scaffold, and figure workflows |
| `results/` | Run records, predictions, statistics, audits, and machine-readable summaries |

The compact result summary is
`results/summary.json`. The checksum-backed
result manifest is `results/RESULTS_MANIFEST.json`.

## Environment

The verified environment uses Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Run commands from the repository root.

## Reproduce the analyses

Rebuild and audit the fixed data partitions:

```bash
.venv/bin/python scripts/audit_dataset.py
.venv/bin/python scripts/create_sequence_cluster_split.py
.venv/bin/python scripts/create_mono_exclusion_splits.py
```

Run the 30-seed primary benchmark and estimator matrix:

```bash
CASPER_WORKERS=8 .venv/bin/python scripts/run_primary_benchmark.py
CASPER_ESTIMATOR_WORKERS=8 CASPER_ESTIMATOR_THREADS=2 \
  .venv/bin/python scripts/run_estimator_matrix.py
```

Run the sensitivity and transfer analyses:

```bash
CASPER_WORKERS=8 .venv/bin/python scripts/run_mono_exclusion_sensitivity.py
CASPER_MODEL_THREADS=2 .venv/bin/python scripts/run_time_forward.py
.venv/bin/python scripts/finalize_time_forward_statistics.py
CASPER_MODEL_THREADS=2 .venv/bin/python scripts/run_scaffold_ranking.py
.venv/bin/python scripts/finalize_scaffold_statistics.py
```

Run the limited position-summary ablation (full training run):

```bash
CASPER_WORKERS=8 .venv/bin/python scripts/run_position_summary_ablation.py
.venv/bin/python scripts/summarize_position_summary_ablation.py
```

Aggregate the statistics and explanations:

```bash
.venv/bin/python scripts/compute_statistics.py
.venv/bin/python scripts/compute_mono_test_sensitivity.py
.venv/bin/python scripts/run_shap.py
.venv/bin/python scripts/compute_secondary_incremental_analysis.py
.venv/bin/python scripts/summarize_results.py
.venv/bin/python scripts/verify_results.py
```

The following commands rebuild the new statistics from the published
predictions or completed run metrics; they do not retrain the benchmark:

```bash
.venv/bin/python scripts/compute_grouped_paired_bootstrap.py
.venv/bin/python scripts/compute_common_no_mono_test_comparison.py
.venv/bin/python scripts/summarize_position_summary_ablation.py
.venv/bin/python scripts/finalize_scaffold_statistics.py
.venv/bin/python scripts/verify_results.py
```

`scripts/audit_descriptor_finiteness.py` independently recalculates the
Chemistry and RDKit 2D descriptors for the 7,224 retained records and records
the before/after-imputation finiteness counts.

Generate the figures from the result tables and predictions with:

```bash
.venv/bin/python scripts/generate_figures.py
```

## Analysis coverage

- Primary descriptor and baseline benchmark: 600 runs
- Estimator-by-representation matrix: 1,050 runs
- Mono-placeholder exclusion sensitivity: 180 runs
- Position-summary ablation: 30 runs
- Exact-sequence grouped paired bootstrap: 5,000 replicates per split
- Common no-Mono held-out comparison: 30 matched seeds per training scheme
- TreeSHAP interpretation: 30 independently optimized seeds
- Time-forward evaluation: 8 cutoff years
- Scaffold-family ranking: 49 families
- Strict sequence-component split: maximum cross-partition identity 0.6923
