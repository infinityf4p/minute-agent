"""Model wrappers with a common fit/predict interface. M1: lgbm and linear."""

from __future__ import annotations

import numpy as np

LGBM_DEFAULTS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
ALIASES = {"lr": "learning_rate"}


class LGBMModel:
    def __init__(self, params: dict | None = None, seed: int = 0, num_threads: int = 0):
        params = dict(params or {})
        self.num_boost_round = int(params.pop("n_estimators", 500))
        p = dict(LGBM_DEFAULTS)
        p.update({ALIASES.get(k, k): v for k, v in params.items()})
        p.update({"seed": seed, "deterministic": True, "num_threads": num_threads})
        self.params = p
        self.booster = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        import lightgbm as lgb

        ds = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, num_threads=self.params["num_threads"])

    def importance(self) -> dict[str, float]:
        gains = self.booster.feature_importance("gain")
        return dict(zip(self.booster.feature_name(), map(float, gains)))


class LinearModel:
    """Ridge regression on features with NaN -> 0 (features are cross-sectionally normalized)."""

    def __init__(self, params: dict | None = None, seed: int = 0, **_):
        self.alpha = float((params or {}).get("alpha", 1.0))
        self.coef = None

    def fit(self, X, y, feature_names):
        X = np.nan_to_num(X)
        Xc = np.column_stack([X, np.ones(len(X))])
        A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        self.coef = np.linalg.solve(A, Xc.T @ y)
        self.names = feature_names
        return self

    def predict(self, X):
        X = np.nan_to_num(X)
        return np.column_stack([X, np.ones(len(X))]) @ self.coef

    def importance(self):
        return dict(zip(self.names, map(float, np.abs(self.coef[:-1]))))


MODELS = {"lgbm": LGBMModel, "linear": LinearModel}


def make_model(spec: dict, seed: int = 0):
    t = spec.get("type")
    if t not in MODELS:
        raise ValueError(f"model type {t!r} not available in M1; choose from {sorted(MODELS)}")
    return MODELS[t](spec.get("params", {}), seed=seed)
