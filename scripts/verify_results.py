#!/usr/bin/env python3
"""Validate and checksum the study results."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRIMARY_RAW = RESULTS / "raw_runs"
ESTIMATOR_RAW = RESULTS / "estimator_matrix/raw_runs"
MONO_SENSITIVITY_RAW = RESULTS / "mono_exclusion_sensitivity/raw_runs"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def completed_metrics(base, expected_trials, field):
    completed = 0
    invalid = []
    for path in sorted(base.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text())
            if payload.get("status") == "completed" and payload.get(field) == expected_trials:
                completed += 1
            else:
                invalid.append(str(path.relative_to(ROOT)))
        except Exception:
            invalid.append(str(path.relative_to(ROOT)))
    return completed, invalid


def main():
    primary_count, primary_invalid = completed_metrics(
        PRIMARY_RAW, expected_trials=50, field="n_optuna_trials"
    )
    estimator_count, estimator_invalid = completed_metrics(
        ESTIMATOR_RAW, expected_trials=10, field="n_trials"
    )
    mono_sensitivity_count, mono_sensitivity_invalid = completed_metrics(
        MONO_SENSITIVITY_RAW, expected_trials=50, field="n_optuna_trials"
    )
    shap_seed_rows = len(pd.read_csv(RESULTS / "shap/seed_level_group_attribution.csv"))
    time_cutoffs = pd.read_csv(RESULTS / "time_forward/cutoff_statistics.csv").cutoff.nunique()
    scaffold_families = pd.read_csv(
        RESULTS / "scaffold_ranking/family_level_results.csv"
    ).family_id.nunique()
    secondary_tests = len(pd.read_csv(
        RESULTS / "secondary_incremental_analysis/cross_estimator_paired_effects.csv"
    ))
    shap_sensitivity_seeds = len(pd.read_csv(
        RESULTS / "secondary_incremental_analysis/shap_performance_seed_level.csv"
    ))

    required = [
        RESULTS / "seed_level_results.csv",
        RESULTS / "summary_with_ci.csv",
        RESULTS / "paired_comparisons.csv",
        RESULTS / "key_results.json",
        RESULTS / "estimator_matrix/seed_level_results.csv",
        RESULTS / "estimator_matrix/summary_with_ci.csv",
        RESULTS / "estimator_matrix/rf_vs_xgb_paired.csv",
        RESULTS / "shap/seed_level_group_attribution.csv",
        RESULTS / "shap/group_summary_with_ci.csv",
        RESULTS / "shap/pairwise_seed_stability.csv",
        RESULTS / "time_forward/cutoff_statistics.csv",
        RESULTS / "time_forward/model_performance_with_ci.csv",
        RESULTS / "scaffold_ranking/family_level_results.csv",
        RESULTS / "scaffold_ranking/summary_with_ci.csv",
        RESULTS / "secondary_incremental_analysis/cross_estimator_paired_effects.csv",
        RESULTS / "secondary_incremental_analysis/cross_estimator_paired_seed_differences.csv",
        RESULTS / "secondary_incremental_analysis/shap_performance_associations.csv",
        RESULTS / "secondary_incremental_analysis/shap_performance_seed_level.csv",
        RESULTS / "secondary_incremental_analysis/analysis_metadata.json",
        RESULTS / "feature_to_block_mapping.json",
        RESULTS / "SECONDARY_ANALYSIS_AUDIT.md",
        RESULTS / "summary.json",
        RESULTS / "audits/reviewer_audit_summary.csv",
        RESULTS / "audits/monomer_coverage.csv",
        RESULTS / "audits/mono_placeholder_records.csv",
        RESULTS / "audits/mono_test_subset_paired_contrasts.csv",
        RESULTS / "mono_exclusion_sensitivity/seed_level_results.csv",
        RESULTS / "mono_exclusion_sensitivity/summary_with_ci.csv",
        RESULTS / "mono_exclusion_sensitivity/paired_contrasts.csv",
        ROOT / "data/splits/CycPeptMPDB_PAMPA/sequence_cluster/metadata.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    complete = (
        primary_count == 600
        and estimator_count == 1050
        and mono_sensitivity_count == 180
        and shap_seed_rows == 720
        and time_cutoffs == 8
        and scaffold_families == 49
        and secondary_tests == 10
        and shap_sensitivity_seeds == 30
        and not primary_invalid
        and not estimator_invalid
        and not mono_sensitivity_invalid
        and not missing
    )
    manifest = {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "dataset_checksum": sha256(
            ROOT / "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl"
        ),
        "raw_data_checksums": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "data/raw").glob("*.csv"))
        },
        "split_checksums": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "data/splits/CycPeptMPDB_PAMPA").rglob("*.jsonl"))
        },
        "completion": {
            "primary_benchmark": {"completed": primary_count, "expected": 600},
            "estimator_matrix": {"completed": estimator_count, "expected": 1050},
            "mono_exclusion_sensitivity": {
                "completed": mono_sensitivity_count,
                "expected": 180,
            },
            "shap_seed_group_rows": shap_seed_rows,
            "time_forward_cutoffs": time_cutoffs,
            "scaffold_families": scaffold_families,
            "secondary_cross_estimator_tests": secondary_tests,
            "shap_sensitivity_seeds": shap_sensitivity_seeds,
        },
        "invalid_primary_metrics": primary_invalid,
        "invalid_estimator_metrics": estimator_invalid,
        "invalid_mono_sensitivity_metrics": mono_sensitivity_invalid,
        "missing_required_artifacts": missing,
        "artifact_checksums": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in required if path.exists()
        },
    }
    (RESULTS / "RESULTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    markdown = f"""# Results manifest

- Status: **{manifest['status']}**
- Primary benchmark: {primary_count}/600 runs
- Estimator matrix: {estimator_count}/1050 runs
- Mono-placeholder exclusion sensitivity: {mono_sensitivity_count}/180 runs
- SHAP: 30 optimization seeds
- Time-forward: {time_cutoffs}/8 cutoffs
- Scaffold ranking: {scaffold_families}/49 families
- Secondary cross-estimator paired tests: {secondary_tests}/10
- Exploratory performance-conditioned SHAP sensitivity: {shap_sensitivity_seeds}/30 seeds

Checksums and validation details are recorded in `RESULTS_MANIFEST.json`.
"""
    (RESULTS / "RESULTS_MANIFEST.md").write_text(markdown)
    print(json.dumps(manifest["completion"], indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
