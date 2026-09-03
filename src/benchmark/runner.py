"""
Benchmark runner — orchestrates model training, HPO, and evaluation.

Supports resume: if output files already exist for a (model, seed, split),
they are skipped unless --force is used.
"""

import json
import csv
import hashlib
import logging
import traceback
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure src/ is on path for sibling imports
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from .registry import MODEL_REGISTRY
from .featurizers import FEATURIZER_REGISTRY
from .evaluation import compute_all_metrics
from .optuna_tuner import tune_xgboost

logger = logging.getLogger(__name__)

# Fixed benchmark partitions and descriptor definitions are identical across
# optimization seeds.  Process-pool workers are long lived, so retaining the
# loaded samples and featurized matrices in each worker avoids recomputing
# expensive RDKit/ECFP representations for every seed without changing any
# split, feature value, hyperparameter search, or fitted estimator.
_DATA_CACHE: Dict[str, Dict[str, Any]] = {}
_FEATURE_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _split_fingerprint(split_dir: Path) -> str:
    """Checksum the exact serialized train/validation/test inputs."""
    digest = hashlib.sha256()
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.jsonl"
        digest.update(name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_samples(split_dir: Path, split_name: str) -> List:
    """Load PEM samples from a split directory."""
    from data.serialization import load_samples as pem_load
    jsonl_path = split_dir / f"{split_name}.jsonl"
    if jsonl_path.exists():
        return pem_load(jsonl_path)
    raise FileNotFoundError(f"No {split_name}.jsonl file in {split_dir}")


def run_single_model(
    model_id: str,
    split_type: str,
    seed: int,
    config: Dict[str, Any],
    output_dir: Path,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate a single model for one seed/split combination.

    Args:
        model_id: Model identifier from MODEL_REGISTRY
        split_type: 'random' or 'sequence_cluster'
        seed: Random seed
        config: Full benchmark config dict
        output_dir: Base output directory
        resume: Skip if output files already exist

    Returns:
        Dict with metrics, predictions, status
    """
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return {'status': 'failed', 'error': f'Unknown model_id: {model_id}'}

    model_dir = output_dir / split_type / f"seed_{seed}" / model_id
    metrics_file = model_dir / "metrics.json"
    preds_file = model_dir / "predictions.csv"

    # Resume check
    if resume and metrics_file.exists() and preds_file.exists():
        logger.info(f"  [{model_id}] seed={seed} — already completed, skipping")
        with open(metrics_file) as f:
            saved = json.load(f)
        return {'status': 'completed', **saved}

    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_started = time.perf_counter()
        # ── 1. Load data ─────────────────────────────────────────────────
        dataset = "CycPeptMPDB_PAMPA"
        project_root = Path(__file__).resolve().parents[2]
        split_dir = project_root / "data/splits" / dataset / split_type
        # Descriptor benchmarks only need serialized PEM samples.  Loading them
        # directly keeps the classical-ML pipeline independent of optional
        # PyTorch dataset wrappers in data.loader.
        if split_type not in _DATA_CACHE:
            data = {
                name: _load_samples(split_dir, name)
                for name in ("train", "val", "test")
            }
            _DATA_CACHE[split_type] = {
                "samples": data,
                "y_train": np.array([s.label for s in data["train"]]),
                "y_val": np.array([s.label for s in data["val"]]),
                "y_test": np.array([s.label for s in data["test"]]),
                "test_ids": [s.sample_id for s in data["test"]],
            }
        cached_data = _DATA_CACHE[split_type]
        data = cached_data["samples"]
        train_samples = data["train"]
        val_samples = data["val"]
        test_samples = data["test"]

        # ── 2. Featurize ─────────────────────────────────────────────────
        featurizer_cls = FEATURIZER_REGISTRY.get(spec.featurizer)
        if featurizer_cls is None:
            raise ValueError(f"Unknown featurizer: {spec.featurizer}")

        feature_key = (
            split_type,
            spec.featurizer,
            json.dumps(spec.featurizer_kwargs, sort_keys=True, default=str),
        )
        if feature_key not in _FEATURE_CACHE:
            featurizer = featurizer_cls(**spec.featurizer_kwargs)
            featurizer.fit(train_samples)
            _FEATURE_CACHE[feature_key] = {
                "X_train": np.nan_to_num(
                    featurizer.transform(train_samples),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "X_val": np.nan_to_num(
                    featurizer.transform(val_samples),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "X_test": np.nan_to_num(
                    featurizer.transform(test_samples),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "feature_names": featurizer.get_feature_names(),
            }
        cached_features = _FEATURE_CACHE[feature_key]
        X_train = cached_features["X_train"]
        X_val = cached_features["X_val"]
        X_test = cached_features["X_test"]
        feature_names = cached_features["feature_names"]

        y_train = cached_data["y_train"]
        y_val = cached_data["y_val"]
        y_test = cached_data["y_test"]
        test_ids = cached_data["test_ids"]

        # ── 3. HPO & Train ───────────────────────────────────────────────
        xgb_cfg = config.get('xgboost', {})
        n_trials = config.get('benchmark', {}).get('n_trials', 50)

        hpo_used = False
        best_params = {}

        if spec.hpo and spec.estimator_type == "xgboost":
            hpo_used = True
            tune_result = tune_xgboost(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials,
                random_seed=seed,
                output_dir=model_dir,
                search_space=xgb_cfg.get('search_space'),
                fixed_params=xgb_cfg.get('fixed_params', {}),
            )
            best_params = tune_result['best_params']
            model = tune_result.get('best_model')
            val_metrics = tune_result['best_val_metrics']

        else:
            # No HPO — use default XGBoost params
            from xgboost import XGBRegressor
            fixed = xgb_cfg.get('fixed_params', {})
            model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                verbosity=0,
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(X_train, y_train, verbose=False)
            y_val_pred = model.predict(X_val)
            val_metrics = compute_all_metrics(y_val, y_val_pred)

        # ── 4. Test Evaluation ───────────────────────────────────────────
        y_test_pred = model.predict(X_test)
        test_metrics = compute_all_metrics(y_test, y_test_pred)

        # ── 5. Feature Importance (XGBoost only) ─────────────────────────
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            try:
                importances = model.feature_importances_
                feature_importance = dict(
                    sorted(zip(feature_names, importances),
                           key=lambda x: x[1], reverse=True)[:30])
            except Exception:
                pass

        # ── 6. Save Results ──────────────────────────────────────────────
        result = {
            'model_id': model_id,
            'model_name': spec.model_name,
            'role': spec.role,
            'seed': seed,
            'split_type': split_type,
            'status': 'completed',
            'hpo': hpo_used,
            'n_features': int(X_train.shape[1]),
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'n_test': len(test_samples),
            'split_fingerprint_sha256': _split_fingerprint(split_dir),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params,
            'feature_importance_top30': feature_importance,
            'runtime_seconds': time.perf_counter() - run_started,
            'selection_metric': 'validation RMSE',
            'n_optuna_trials': n_trials if hpo_used else 0,
        }

        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Save predictions CSV
        with open(preds_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'y_true', 'y_pred', 'split', 'model_id',
                'seed', 'split_type'
            ])
            for sid, yt, yp in zip(test_ids, y_test, y_test_pred):
                writer.writerow([
                    sid, float(yt), float(yp), 'test', model_id,
                    seed, split_type
                ])

        logger.info(
            f"  [{model_id}] seed={seed} "
            f"R²={test_metrics['r2']:.4f} "
            f"RMSE={test_metrics['rmse']:.4f} "
            f"ρ={test_metrics['spearman']:.4f}")

        return result

    except Exception as e:
        logger.error(f"  [{model_id}] seed={seed} FAILED: {e}")
        traceback.print_exc()
        result = {
            'model_id': model_id,
            'seed': seed,
            'split_type': split_type,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return result
