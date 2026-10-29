"""Cross-validation and metric helpers shared by notebooks 02 and 03."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from .features import RANDOM_STATE


def make_cv(n_splits: int = 5) -> StratifiedGroupKFold:
    """5-fold CV that keeps class balance AND keeps duplicate rows in the same fold."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series, groups, cv=None, name: str = ""):
    """Run N-fold CV once and return (per-fold metrics DataFrame, out-of-fold probabilities).

    The out-of-fold (OOF) probabilities let us analyse errors and compare models
    without ever touching the test set.

    `cv` is a splitter (default: grouped 5-fold) or a ready list of (train_idx, val_idx)
    pairs, e.g. from `time_cv`. Rows never used for validation get NaN in the OOF array.
    """
    cv = cv or make_cv()
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    rows = []
    splits = cv if isinstance(cv, list) else cv.split(X, y, groups)
    for fold, (tr, va) in enumerate(splits):
        t0 = time.time()
        m = clone(model).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        pred = (p >= 0.5).astype(int)
        rows.append(dict(model=name, fold=fold,
                         roc_auc=roc_auc_score(y[va], p),
                         pr_auc=average_precision_score(y[va], p),
                         f1=f1_score(y[va], pred),
                         precision=precision_score(y[va], pred, zero_division=0),
                         recall=recall_score(y[va], pred),
                         fit_seconds=time.time() - t0))
    return pd.DataFrame(rows), oof


def summarise(fold_df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std per model, sorted by ROC-AUC."""
    agg = fold_df.groupby("model").agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
        out[f"{m}_mean"] = agg[(m, "mean")]
        out[f"{m}_std"] = agg[(m, "std")]
    out["fit_s_per_fold"] = agg[("fit_seconds", "mean")]
    return out.sort_values("roc_auc_mean", ascending=False)


def best_f1_threshold(y_true, proba):
    """Threshold that maximises F1 on (out-of-fold) predictions."""
    thr = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in thr]
    i = int(np.argmax(f1s))
    return float(thr[i]), float(f1s[i])


# ---------------------------------------------------------------------------
# Forward-in-time validation (used for tuning and model selection)
# ---------------------------------------------------------------------------
TIME_FOLDS = [  # expanding window: train on everything before `start`, validate on [start, end)
    ("2016-05-01", "2016-09-01"),
    ("2016-09-01", "2017-01-01"),
    ("2017-01-01", "2017-05-01"),
]


def time_cv(arrival_dates: pd.Series, folds=TIME_FOLDS):
    """Expanding-window folds on arrival date: the validation period is always in the
    future of the training period, as in deployment. Identical group bookings share an
    arrival date, so they can never straddle a fold boundary."""
    d = pd.to_datetime(pd.Series(arrival_dates)).reset_index(drop=True)
    out = []
    for start, end in folds:
        tr = np.where(d < start)[0]
        va = np.where((d >= start) & (d < end))[0]
        out.append((tr, va))
    return out


# ---------------------------------------------------------------------------
# Business cost of a decision threshold
# ---------------------------------------------------------------------------
def expected_cost(y_true, flagged, c):
    """Cost in 'empty-room units': a missed cancellation costs 1, a false alarm costs c."""
    y_true = np.asarray(y_true); flagged = np.asarray(flagged).astype(bool)
    fn = ((y_true == 1) & ~flagged).sum()
    fp = ((y_true == 0) & flagged).sum()
    return fn + c * fp


def best_cost_threshold(y_true, proba, c, grid=None):
    """Threshold that minimises expected_cost on (out-of-fold) predictions."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    costs = np.array([expected_cost(y_true, proba >= t, c) for t in grid])
    return float(grid[int(np.argmin(costs))]), costs


# ---------------------------------------------------------------------------
# Uncertainty: bootstrap over duplicate groups
# ---------------------------------------------------------------------------
def group_bootstrap(y_true, scores: dict, groups, metric, n_boot=300, seed=RANDOM_STATE):
    """95% interval of `metric` for each score vector, resampling whole duplicate groups
    (identical rows are resampled together). Also returns the interval of the
    difference between the first two score vectors (paired)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    idx_by_group = pd.Series(np.arange(len(y_true))).groupby(np.asarray(groups)).apply(np.array).values
    names = list(scores)
    draws = {n: [] for n in names}
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
        idx = np.concatenate(idx_by_group[pick])
        vals = {n: metric(y_true[idx], np.asarray(scores[n])[idx]) for n in names}
        for n in names:
            draws[n].append(vals[n])
        diffs.append(vals[names[0]] - vals[names[1]])
    out = {n: np.percentile(draws[n], [2.5, 97.5]) for n in names}
    out[f"{names[0]} − {names[1]}"] = np.percentile(diffs, [2.5, 97.5])
    return out
