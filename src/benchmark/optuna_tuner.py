"""
Optuna hyperparameter optimization for benchmark models.

Supports XGBoost and RandomForest with consistent API.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

logger = logging.getLogger(__name__)


def _compute_val_metrics(y_true, y_pred):
    """Compute validation metrics without sklearn dependency for speed."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    sp, _ = spearmanr(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'spearman': float(sp)}


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_seed: int = 42,
    pruner_name: str = "median",
    n_startup_trials: int = 5,
    output_dir: Optional[Path] = None,
    search_space: Optional[Dict[str, list]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters with Optuna.

    Returns dict with best_params, best_val_metrics, trials_df, study.
    """
    import optuna
    from xgboost import XGBRegressor

    if search_space is None:
        search_space = {
            'n_estimators': [200, 2000],
            'max_depth': [2, 10],
            'learning_rate': [0.005, 0.2],
            'subsample': [0.5, 1.0],
            'colsample_bytree': [0.5, 1.0],
            'min_child_weight': [1, 20],
            'reg_alpha': [1e-8, 10.0],
            'reg_lambda': [1e-8, 100.0],
            'gamma': [0.0, 10.0],
        }
    # Ensure all search space bounds are numeric
    search_space = {k: [float(v[0]), float(v[1])] for k, v in search_space.items()}
    if fixed_params is None:
        fixed_params = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
            'subsample': trial.suggest_float('subsample', *search_space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
            'min_child_weight': trial.suggest_int('min_child_weight', *search_space['min_child_weight']),
            'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda'], log=True),
            'gamma': trial.suggest_float('gamma', *search_space['gamma']),
        }
        # Merge fixed params (overriding search space defaults)
        model_params = {**params}
        for k, v in fixed_params.items():
            if k != 'random_state' and k != 'n_jobs' and k != 'verbosity':
                model_params[k] = v

        model = XGBRegressor(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            learning_rate=model_params['learning_rate'],
            subsample=model_params['subsample'],
            colsample_bytree=model_params['colsample_bytree'],
            min_child_weight=model_params['min_child_weight'],
            reg_alpha=model_params['reg_alpha'],
            reg_lambda=model_params['reg_lambda'],
            gamma=model_params['gamma'],
            tree_method=fixed_params.get('tree_method', 'hist'),
            early_stopping_rounds=fixed_params.get('early_stopping_rounds', 50),
            verbosity=0,
            n_jobs=fixed_params.get('n_jobs', -1),
            random_state=random_seed,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    # Create pruner
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials)
    else:
        pruner = None

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name=f"xgb_seed{random_seed}",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params

    # Retrain with best params on train set, evaluate on val
    best_model = XGBRegressor(
        n_estimators=best_params.get('n_estimators', 500),
        max_depth=best_params.get('max_depth', 6),
        learning_rate=best_params.get('learning_rate', 0.1),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        min_child_weight=best_params.get('min_child_weight', 1),
        reg_alpha=best_params.get('reg_alpha', 0.0),
        reg_lambda=best_params.get('reg_lambda', 1.0),
        gamma=best_params.get('gamma', 0.0),
        tree_method=fixed_params.get('tree_method', 'hist'),
        verbosity=0,
        n_jobs=fixed_params.get('n_jobs', -1),
        random_state=random_seed,
    )
    best_model.fit(X_train, y_train, verbose=False)
    y_val_pred = best_model.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_val_pred)

    # Collect trials data
    trials_data = []
    for t in study.trials:
        trials_data.append({
            'number': t.number,
            'value': t.value if t.value is not None else float('nan'),
            'state': str(t.state),
            **{f'param_{k}': v for k, v in t.params.items()},
        })

    result = {
        'best_params': best_params,
        'best_val_metrics': val_metrics,
        'best_val_rmse': study.best_value,
        'n_trials': len(study.trials),
        'trials': trials_data,
        'study': study,
        'best_model': best_model,
    }

    # Save outputs if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save best_params.json
        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        # Save trials CSV
        import csv
        if trials_data:
            with open(output_dir / 'optuna_trials.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trials_data[0].keys())
                writer.writeheader()
                writer.writerows(trials_data)

    return result


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    search_space: Optional[Dict] = None,
    fixed_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Tune RandomForest hyperparameters with Optuna."""
    import optuna
    from sklearn.ensemble import RandomForestRegressor

    if search_space is None:
        search_space = {
            'n_estimators': [100, 1000],
            'max_depth': [5, 30],
            'min_samples_split': [2, 20],
            'min_samples_leaf': [1, 10],
        }
    if fixed_params is None:
        fixed_params = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
            'min_samples_split': trial.suggest_int('min_samples_split', *search_space['min_samples_split']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *search_space['min_samples_leaf']),
        }
        if 'max_features' in search_space:
            params['max_features'] = trial.suggest_categorical(
                'max_features', search_space['max_features'])

        model = RandomForestRegressor(
            random_state=random_seed,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name=f"rf_seed{random_seed}",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = RandomForestRegressor(
        random_state=random_seed, n_jobs=-1, **best_params)
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_val_pred)

    trials_data = []
    for t in study.trials:
        trials_data.append({
            'number': t.number,
            'value': t.value if t.value is not None else float('nan'),
            'state': str(t.state),
            **{f'param_{k}': v for k, v in t.params.items()},
        })

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        if trials_data:
            import csv
            with open(output_dir / 'optuna_trials.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trials_data[0].keys())
                writer.writeheader()
                writer.writerows(trials_data)

    return {
        'best_params': best_params,
        'best_val_metrics': val_metrics,
        'best_val_rmse': study.best_value,
        'n_trials': len(study.trials),
        'trials': trials_data,
        'best_model': best_model,
    }


def _scale_features(X_train, X_val, X_test=None):
    """StandardScaler for linear models and SVR. Fit on train only."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    if X_test is not None:
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s
    return X_train_s, X_val_s


def tune_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Tune Ridge regression with Optuna. Assumes pre-scaled features."""
    import optuna
    from sklearn.linear_model import Ridge

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 1e4, log=True)
        model = Ridge(alpha=alpha, random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name=f"ridge_seed{random_seed}",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = Ridge(alpha=best_params['alpha'], random_state=random_seed)
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_val_pred)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

    return {
        'best_params': best_params,
        'best_val_metrics': val_metrics,
        'best_val_rmse': study.best_value,
        'n_trials': len(study.trials),
        'best_model': best_model,
    }


def tune_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Tune ElasticNet with Optuna. Assumes pre-scaled features."""
    import optuna
    from sklearn.linear_model import ElasticNet

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 100.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                          random_state=random_seed, max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name=f"en_seed{random_seed}",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'],
                           random_state=random_seed, max_iter=5000)
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_val_pred)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

    return {
        'best_params': best_params,
        'best_val_metrics': val_metrics,
        'best_val_rmse': study.best_value,
        'n_trials': len(study.trials),
        'best_model': best_model,
    }


def tune_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Tune SVR (RBF kernel) with Optuna. Assumes pre-scaled features."""
    import optuna
    from sklearn.svm import SVR

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        gamma = trial.suggest_float('gamma', 1e-5, 1.0, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 0.5, log=True)
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon,
                   max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name=f"svr_seed{random_seed}",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'],
                    epsilon=best_params['epsilon'], max_iter=5000)
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_val_pred)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

    return {
        'best_params': best_params,
        'best_val_metrics': val_metrics,
        'best_val_rmse': study.best_value,
        'n_trials': len(study.trials),
        'best_model': best_model,
    }
