"""
Shared data-loading, cleaning, splitting and feature-engineering code.

Everything that turns the raw CSV into model-ready features lives here so the
three notebooks use exactly the same logic (one source of truth, no copy-paste
drift). All *learned* transformations (imputation values, scaling, encoding,
rare-category grouping) are done inside scikit-learn Pipelines in the notebooks,
so they are fitted on training folds only -> no data leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hotel_bookings.csv"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"

TARGET = "is_canceled"
RANDOM_STATE = 42

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# --------------------------------------------------------------------------
# Columns that must never be used as features (see README "Data leakage")
# --------------------------------------------------------------------------
LEAKY_COLUMNS = [
    "reservation_status",        # 'Canceled' / 'No-Show' / 'Check-Out' == the target itself
    "reservation_status_date",   # date of the final status -> only known after the outcome
    "assigned_room_type",        # assigned at check-in; changed for 21% of stays vs 2% of cancellations
]
# --------------------------------------------------------------------------
# Columns that are updated AFTER the booking is created. The model is used at
# booking time (most cancellations happen weeks before arrival: median 48 days
# in the training data),
# so the main model excludes them. `booking_time=False` adds them back as an
# "upper bound" variant, i.e. what the model could do if they were known early.
# --------------------------------------------------------------------------
POST_BOOKING_COLUMNS = [
    "required_car_parking_spaces",   # often confirmed later / at arrival (0% of requesters cancel: too perfect)
    "total_of_special_requests",     # guests add requests after booking
    "booking_changes",               # accumulates over the life of the booking
]
POST_BOOKING_DERIVED = ["needs_parking", "has_special_requests"]
# Identifiers / columns replaced by engineered versions
DROP_COLUMNS = [
    "arrival_date_year",         # calendar year does not generalise to future years
    "arrival_date_week_number",  # redundant with month + day
    "arrival_date_day_of_month",
    "company",                   # 94% missing -> replaced by has_company flag
]


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV and add a proper arrival date column."""
    df = pd.read_csv(path)
    month_num = df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    df["arrival_date"] = pd.to_datetime(
        dict(year=df["arrival_date_year"], month=month_num, day=df["arrival_date_day_of_month"])
    )
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that does not learn anything from the data
    (safe to apply before the split)."""
    df = df.copy()
    # 1) bookings with zero guests are data-entry errors (180 rows)
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df[guests > 0]
    # 2) negative or absurd average daily rate (1 row at 5,400 EUR, 1 negative)
    df = df[(df["adr"] >= 0) & (df["adr"] < 1000)]
    # 3) 'Undefined' is a missing-value code in these columns
    for col in ["meal", "market_segment", "distribution_channel"]:
        df[col] = df[col].replace("Undefined", np.nan)
    # note: in `meal`, 'Undefined' and 'SC' both mean "no meal package" per the data dictionary
    df["meal"] = df["meal"].fillna("SC")
    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_start: str = "2017-05-01"):
    """Out-of-time split: train on arrivals before `test_start`, test on later ones.
    Mirrors deployment (model trained on the past, used on future bookings)."""
    train = df[df["arrival_date"] < test_start].reset_index(drop=True)
    test = df[df["arrival_date"] >= test_start].reset_index(drop=True)
    return train, test


def duplicate_groups(df: pd.DataFrame) -> np.ndarray:
    """Group id shared by rows that are identical *as the model sees them*
    (mostly multi-room group bookings). Used with StratifiedGroupKFold so such
    rows never sit in both the training and validation fold of CV.

    The hash is taken over the model's input features (the output of
    FeatureEngineer), NOT over all raw columns: two rows that differ only in a
    column the model never sees (e.g. reservation_status or booking_changes)
    are still twins for the model and must stay in the same fold."""
    feats = FeatureEngineer().transform(df.drop(columns=[TARGET], errors="ignore"))
    return pd.util.hash_pandas_object(feats, index=False).values


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-wise feature engineering (no fitting needed).

    Parameters
    ----------
    use_country : bool
        `country` is suspected of partial leakage (nationality is often only
        corrected at check-in, so cancelled bookings keep the default 'PRT').
        Treated as a hyper-parameter / ablation in the notebooks.
    booking_time : bool
        True (default, main model): only information available when the booking
        is created -> drop POST_BOOKING_COLUMNS and the flags derived from them.
        False: keep them (upper-bound sensitivity check).
    """

    def __init__(self, use_country: bool = True, booking_time: bool = True):
        self.use_country = use_country
        self.booking_time = booking_time

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["children"] = X["children"].fillna(0)
        # --- stay / party composition
        X["total_nights"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
        X["total_guests"] = X["adults"] + X["children"] + X["babies"]
        X["has_kids"] = ((X["children"] + X["babies"]) > 0).astype(int)
        X["weekend_share"] = np.where(X["total_nights"] > 0,
                                      X["stays_in_weekend_nights"] / X["total_nights"].clip(lower=1), 0)
        # --- price
        X["adr_per_guest"] = X["adr"] / X["total_guests"].clip(lower=1)
        X["total_revenue"] = X["adr"] * X["total_nights"]
        # --- time features (decompose dates)
        X["arrival_month"] = X["arrival_date"].dt.month
        X["arrival_dow"] = X["arrival_date"].dt.dayofweek
        booking_date = X["arrival_date"] - pd.to_timedelta(X["lead_time"], unit="D")
        X["booking_month"] = booking_date.dt.month
        X["booking_dow"] = booking_date.dt.dayofweek
        X["log_lead_time"] = np.log1p(X["lead_time"])
        # --- customer history
        hist = X["previous_cancellations"] + X["previous_bookings_not_canceled"]
        X["prev_cancel_ratio"] = np.where(hist > 0, X["previous_cancellations"] / hist.clip(lower=1), 0)
        X["has_history"] = (hist > 0).astype(int)
        # --- agent / company
        X["has_agent"] = X["agent"].notna().astype(int)
        X["agent"] = X["agent"].fillna(0).astype(int).astype(str)  # 0 = booked without agent
        X["has_company"] = X["company"].notna().astype(int) if "company" in X else 0
        X["has_special_requests"] = (X["total_of_special_requests"] > 0).astype(int)
        X["needs_parking"] = (X["required_car_parking_spaces"] > 0).astype(int)

        drop = [c for c in LEAKY_COLUMNS + DROP_COLUMNS + ["arrival_date", "arrival_date_month", TARGET]
                if c in X.columns]
        if not self.use_country:
            drop.append("country")
        if self.booking_time:
            drop += POST_BOOKING_COLUMNS + POST_BOOKING_DERIVED
        return X.drop(columns=drop)


NUMERIC_FEATURES = [
    "lead_time", "log_lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr",
    "required_car_parking_spaces", "total_of_special_requests", "total_nights",
    "total_guests", "has_kids", "weekend_share", "adr_per_guest", "total_revenue",
    "arrival_month", "arrival_dow", "booking_month", "booking_dow", "prev_cancel_ratio",
    "has_history", "has_agent", "has_company", "has_special_requests", "needs_parking",
]
# Original numeric columns only (used for the "no feature engineering" ablation)
RAW_NUMERIC_FEATURES = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "arrival_month",
]
LOW_CARD_CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel",
                        "reserved_room_type", "deposit_type", "customer_type"]
HIGH_CARD_CATEGORICAL = ["country", "agent"]   # ~170 countries, ~330 agents


def make_preprocessor(kind: str = "onehot", use_country: bool = True,
                      min_frequency: int = 50, engineered: bool = True,
                      booking_time: bool = True) -> ColumnTransformer:
    """Build the learned part of preprocessing.

    kind='onehot'  -> impute + scale numerics, one-hot categoricals with rare
                      levels (< min_frequency rows) grouped together.
                      Good for linear models, SVM, NB, k-NN, MLP.
    kind='ordinal' -> impute numerics (no scaling), ordinal-encode categoricals.
                      Good for tree ensembles (scale-invariant, compact).
    """
    high_card = [c for c in HIGH_CARD_CATEGORICAL if use_country or c != "country"]
    cats = LOW_CARD_CATEGORICAL + high_card
    if kind == "onehot":
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                             ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                      min_frequency=min_frequency,
                                                      sparse_output=False))])
    elif kind == "ordinal":
        num_pipe = SimpleImputer(strategy="median")
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                             ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                        unknown_value=-1,
                                                        min_frequency=min_frequency))])
    else:
        raise ValueError(kind)
    numeric = NUMERIC_FEATURES if engineered else RAW_NUMERIC_FEATURES
    if booking_time:
        numeric = [c for c in numeric if c not in POST_BOOKING_COLUMNS + POST_BOOKING_DERIVED]
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, cats)],
                             verbose_feature_names_out=False)


def make_pipeline(model, kind: str = "onehot", use_country: bool = True,
                  engineered: bool = True, min_frequency: int = 50,
                  booking_time: bool = True) -> Pipeline:
    """Full leak-free pipeline: raw (cleaned) rows -> features -> encoding -> model.
    By default only booking-time information is used (booking_time=True)."""
    return Pipeline([
        ("features", FeatureEngineer(use_country=use_country, booking_time=booking_time)),
        ("prep", make_preprocessor(kind=kind, use_country=use_country, engineered=engineered,
                                   min_frequency=min_frequency, booking_time=booking_time)),
        ("model", model),
    ])


class SegmentRateBaseline(ClassifierMixin, BaseEstimator):
    """Business-rule baseline: predict the historical cancellation rate of the
    booking's (market_segment, deposit_type, hotel) cell, learned on the training fold.
    This is what a revenue manager would do with a pivot table today."""

    def __init__(self, keys=("hotel", "market_segment", "deposit_type")):
        self.keys = keys

    def fit(self, X, y):
        k = list(self.keys)
        d = X[k].astype(str).assign(_y=np.asarray(y))
        self.rates_ = d.groupby(k)["_y"].mean()
        self.prior_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        k = list(self.keys)
        idx = pd.MultiIndex.from_frame(X[k].astype(str))
        p = self.rates_.reindex(idx).fillna(self.prior_).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
