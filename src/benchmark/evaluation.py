"""
Evaluation metrics for CASPer benchmark.

All models output identical metrics for fair comparison.
"""

import numpy as np
from typing import Dict, Optional
from scipy.stats import spearmanr, pearsonr, kendalltau


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compute_ranking: bool = False,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground-truth labels
        y_pred: Model predictions
        compute_ranking: Also compute top-k and pairwise ranking metrics

    Returns:
        Dictionary of metric_name -> value
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Remove NaN if any
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
                'spearman': float('nan'), 'pearson': float('nan'),
                'kendall_tau': float('nan'), 'n_samples': len(y_true)}

    results = {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
        'pearson': float(pearsonr(y_true, y_pred)[0]),
        'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
        'n_samples': len(y_true),
    }

    # Prediction variance ratio (model calibration check)
    pred_var = np.var(y_pred)
    true_var = np.var(y_true)
    results['variance_ratio'] = float(pred_var / true_var) if true_var > 0 else 0.0

    if compute_ranking:
        results.update(_compute_ranking_metrics(y_true, y_pred))

    return results


def _compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: float = 0.1,
) -> Dict[str, float]:
    """Compute ranking-specific metrics."""
    n = len(y_true)
    k = max(1, int(n * top_k))

    # Top-k enrichment: fraction of true top-k captured in predicted top-k
    true_top_k = set(np.argsort(-y_true)[:k])
    pred_top_k = set(np.argsort(-y_pred)[:k])
    top_k_enrichment = len(true_top_k & pred_top_k) / k if k > 0 else 0.0

    # Pairwise ranking accuracy
    n_pairs = min(10000, n * (n - 1) // 2)
    if n_pairs > 0:
        # Sample pairs
        rng = np.random.RandomState(42)
        indices = np.arange(n)
        correct = 0
        total = 0
        for _ in range(n_pairs):
            i, j = rng.choice(indices, 2, replace=False)
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                    correct += 1
        pairwise_acc = correct / total if total > 0 else 0.5
    else:
        pairwise_acc = 0.5

    return {
        'top_k_enrichment': float(top_k_enrichment),
        'pairwise_ranking_accuracy': float(pairwise_acc),
    }
