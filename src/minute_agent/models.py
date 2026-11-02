"""Model wrappers with a common fit/predict interface.

    lgbm    LightGBM, CPU (this build has no CUDA learner); windows train in parallel threads
    xgb     XGBoost hist, CUDA when available
    mlp     PyTorch MLP, CUDA when available
    linear  ridge regression, CPU

Every model has `device` ("cpu" or "cuda") and `set_threads(n)`; the runner
uses them to train CPU models in parallel and keep GPU models sequential.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _device(requested: str | None) -> str:
    if requested in (None, "auto"):
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        raise RuntimeError("model asked for device=cuda but CUDA is not available")
    return requested


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


class LGBMModel:
    device = "cpu"

    def __init__(self, params: dict | None = None, seed: int = 0):
        params = dict(params or {})
        params.pop("device", None)
        self.num_boost_round = int(params.pop("n_estimators", 500))
        p = dict(LGBM_DEFAULTS)
        p.update({{"lr": "learning_rate"}.get(k, k): v for k, v in params.items()})
        p.update({"seed": seed, "deterministic": True, "num_threads": 0})
        self.params = p

    def set_threads(self, n: int):
        self.params["num_threads"] = int(n)

    def fit(self, X, y, feature_names):
        import lightgbm as lgb

        ds = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        return self.booster.predict(X, num_threads=self.params["num_threads"])

    def importance(self):
        return dict(zip(self.booster.feature_name(), map(float, self.booster.feature_importance("gain"))))


class XGBModel:
    def __init__(self, params: dict | None = None, seed: int = 0):
        params = dict(params or {})
        self.device = _device(params.pop("device", "auto"))
        self.num_boost_round = int(params.pop("n_estimators", 500))
        alias = {"lr": "eta", "learning_rate": "eta", "num_leaves": "max_leaves"}
        p = {"objective": "reg:squarederror", "tree_method": "hist", "eta": 0.05, "max_depth": 6,
             "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
             "max_bin": 256, "verbosity": 0}
        p.update({alias.get(k, k): v for k, v in params.items()})
        if "max_leaves" in p and "max_depth" not in params:
            p["max_depth"] = 0
            p["grow_policy"] = "lossguide"
        p.update({"seed": seed, "device": self.device})
        self.params = p

    def set_threads(self, n: int):
        self.params["nthread"] = int(n)

    def fit(self, X, y, feature_names):
        import xgboost as xgb

        d = xgb.QuantileDMatrix(X.astype(np.float32), y.astype(np.float32),
                                feature_names=feature_names, max_bin=self.params["max_bin"])
        self.booster = xgb.train(self.params, d, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        import xgboost as xgb

        return self.booster.inplace_predict(X.astype(np.float32))

    def importance(self):
        return {k: float(v) for k, v in self.booster.get_score(importance_type="total_gain").items()}


class MLPModel:
    """Feed-forward net on cross-sectionally normalized features (NaN -> 0)."""

    def __init__(self, params: dict | None = None, seed: int = 0):
        params = dict(params or {})
        self.device = _device(params.pop("device", "auto"))
        self.hidden = list(params.pop("hidden", [256, 128]))
        self.dropout = float(params.pop("dropout", 0.1))
        self.lr = float(params.pop("lr", 1e-3))
        self.epochs = int(params.pop("epochs", 8))
        self.batch = int(params.pop("batch_size", 4096))
        self.weight_decay = float(params.pop("weight_decay", 1e-4))
        if params:
            raise ValueError(f"unknown mlp params {sorted(params)}")
        self.seed = seed

    def set_threads(self, n: int):
        self.threads = int(n)

    def fit(self, X, y, feature_names):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        Xt = torch.as_tensor(np.nan_to_num(X).astype(np.float32), device=dev)
        yt = torch.as_tensor(y.astype(np.float32), device=dev)
        layers, d_in = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        g = torch.Generator(device=dev).manual_seed(self.seed)
        n = len(Xt)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                loss = nn.functional.mse_loss(self.net(Xt[idx]).squeeze(-1), yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.names = feature_names
        return self

    def predict(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(np.nan_to_num(X).astype(np.float32), device=torch.device(self.device))
            out = torch.cat([self.net(Xt[i:i + 65536]).squeeze(-1) for i in range(0, len(Xt), 65536)])
        return out.cpu().numpy().astype(np.float64)

    def importance(self):
        w = self.net[0].weight.detach().abs().sum(0).cpu().numpy()
        return dict(zip(self.names, map(float, w)))


class LinearModel:
    """Ridge regression on features with NaN -> 0 (features are cross-sectionally normalized)."""

    device = "cpu"

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.alpha = float((params or {}).get("alpha", 1.0))

    def set_threads(self, n: int):
        pass

    def fit(self, X, y, feature_names):
        Xc = np.column_stack([np.nan_to_num(X), np.ones(len(X))])
        self.coef = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1]), Xc.T @ y)
        self.names = feature_names
        return self

    def predict(self, X):
        return np.column_stack([np.nan_to_num(X), np.ones(len(X))]) @ self.coef

    def importance(self):
        return dict(zip(self.names, map(float, np.abs(self.coef[:-1]))))


class _SeqBase:
    """Sequence model on the bucketed intraday sequence (columns sq_<step>_<channel>),
    optionally concatenated with the tabular features before the output head."""

    needs_seq = True

    def __init__(self, params: dict | None = None, seed: int = 0):
        params = dict(params or {})
        self.device = _device(params.pop("device", "auto"))
        self.lr = float(params.pop("lr", 1e-3))
        self.epochs = int(params.pop("epochs", 4))
        self.batch = int(params.pop("batch_size", 2048))
        self.dropout = float(params.pop("dropout", 0.1))
        self.with_features = bool(params.pop("with_features", True))
        self.arch = params
        self.seed = seed

    def set_threads(self, n: int):
        pass

    def _split(self, X: np.ndarray, names: list[str]):
        seq_idx = [i for i, n in enumerate(names) if n.startswith("sq_")]
        tab_idx = [i for i, n in enumerate(names) if not n.startswith("sq_")] if self.with_features else []
        steps = sorted({names[i][3:5] for i in seq_idx})
        T, C = len(steps), len(seq_idx) // max(len(steps), 1)
        return seq_idx, tab_idx, T, C

    def _tensors(self, X, names):
        import torch

        seq_idx, tab_idx, T, C = self._split(X, names)
        S = torch.from_numpy(np.ascontiguousarray(X[:, seq_idx], dtype=np.float32).reshape(len(X), T, C))
        F = torch.from_numpy(np.nan_to_num(X[:, tab_idx]).astype(np.float32)) if tab_idx else None
        return S, F, T, C, len(tab_idx)

    def _encoder(self, C):
        raise NotImplementedError

    def fit(self, X, y, feature_names):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        S, F, T, C, nf = self._tensors(X, feature_names)
        if T == 0:
            raise ValueError("sequence model needs seq_bars features")
        self.names = feature_names
        enc, d_out = self._encoder(C)
        head_in = d_out + nf
        self.net = nn.ModuleDict({
            "enc": enc,
            "head": nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(64, 1)),
        }).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        yt = torch.from_numpy(y.astype(np.float32))
        n = len(S)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                s = S[idx].to(dev, non_blocking=True)
                f = F[idx].to(dev, non_blocking=True) if F is not None else None
                loss = nn.functional.mse_loss(self._forward(s, f), yt[idx].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def _forward(self, s, f):
        import torch

        h = self._encode(s)
        if f is not None:
            h = torch.cat([h, f], dim=1)
        return self.net["head"](h).squeeze(-1)

    def predict(self, X):
        import torch

        S, F, *_ = self._tensors(X, self.names)
        dev = torch.device(self.device)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(S), 16384):
                s = S[i:i + 16384].to(dev)
                f = F[i:i + 16384].to(dev) if F is not None else None
                out.append(self._forward(s, f).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

    def importance(self):
        return {}


class GRUSeqModel(_SeqBase):
    def _encoder(self, C):
        from torch import nn

        hidden, layers = int(self.arch.get("hidden", 64)), int(self.arch.get("layers", 1))
        self._gru = nn.GRU(C, hidden, num_layers=layers, batch_first=True,
                           dropout=self.dropout if layers > 1 else 0.0)
        return self._gru, hidden

    def _encode(self, s):
        _, h = self.net["enc"](s)
        return h[-1]


class TransformerSeqModel(_SeqBase):
    def _encoder(self, C):
        import torch
        from torch import nn

        dm, heads, layers = int(self.arch.get("d_model", 64)), int(self.arch.get("heads", 4)), int(self.arch.get("layers", 1))

        class Enc(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(C, dm)
                s.pos = nn.Parameter(torch.zeros(1, 64, dm))
                layer = nn.TransformerEncoderLayer(dm, heads, dim_feedforward=2 * dm, dropout=self.dropout,
                                                   batch_first=True)
                s.tf = nn.TransformerEncoder(layer, layers)

            def forward(s, x):
                z = s.inp(x) + s.pos[:, : x.shape[1]]
                return s.tf(z)[:, -1]
        return Enc(), dm

    def _encode(self, s):
        return self.net["enc"](s)


MODELS = {"lgbm": LGBMModel, "xgb": XGBModel, "mlp": MLPModel, "linear": LinearModel,
          "gru_seq": GRUSeqModel, "transformer_seq": TransformerSeqModel}
SEQ_MODEL_TYPES = {"gru_seq", "transformer_seq"}


def make_model(spec: dict, seed: int = 0):
    t = spec.get("type")
    if t not in MODELS:
        raise ValueError(f"model type {t!r} not available; choose from {sorted(MODELS)}")
    return MODELS[t](spec.get("params", {}), seed=seed)


def cpu_count() -> int:
    return os.cpu_count() or 4
