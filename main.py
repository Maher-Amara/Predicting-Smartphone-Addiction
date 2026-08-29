"""
Predicting Smartphone Addiction -- Kaggle Playground Series S6E8
================================================================
Multi-model stacking pipeline: self-trained base models + 150+ public OOF
library members -> Rank-Gauss -> Nested L2 Logistic Regression meta-learner.

Key design principles (from community analysis):
  1. Diversity beats strength -- different model families matter more than tuning
  2. "No target encoding" views are the most valuable for diversity
  3. Trust CV, never the Public LB -- avoid overfitting the 20% public split
  4. Rank-Gauss normalisation before the meta-model
  5. Frozen 5-fold StratifiedKFold with seed=42, original row order preserved
  6. Public OOF libraries provide ~150+ additional diverse members
  7. Quarantine: hash-dedup + KS-drift filter + AUC floor before stacking
"""

import os
import copy
import glob
import hashlib
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ================================================================================
# CONFIGURATION
# ================================================================================
SEED = 42
N_FOLDS = 5
INPUT_DIR = Path("input")
OOF_DIR = Path("input/oof_libraries")
OUTPUT_DIR = Path("submissions")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET = "addicted_label"
ID_COL = "id"

RAW_NUM_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
RAW_CAT_COLS = ["gender", "stress_level", "academic_work_impact"]

EPS = 1e-4  # small constant for safe division


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================================
# PHASE 1: DATA LOADING
# ================================================================================
def load_data():
    """Load train and test CSVs without sorting or reindexing."""
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    print(f"Train: {train.shape}  Test: {test.shape}  Positive rate: {train[TARGET].mean():.4f}")
    return train, test


# ================================================================================
# PHASE 2: FEATURE ENGINEERING
# ================================================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals for tree models (both numeric and string representations)."""
    df = df.copy()
    # Numeric encoding for LightGBM / XGBoost
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    df["gender_enc"] = df["gender"].map(gender_map)

    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    df["stress_level_enc"] = df["stress_level"].map(stress_map)

    impact_map = {"No": 0, "Yes": 1}
    df["academic_work_impact_enc"] = df["academic_work_impact"].map(impact_map)

    # String encoding for CatBoost native categorical handling
    df["gender_cat"] = df["gender"].fillna("Missing").astype(str)
    df["stress_level_cat"] = df["stress_level"].fillna("Missing").astype(str)
    df["academic_work_impact_cat"] = df["academic_work_impact"].fillna("Missing").astype(str)

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and interaction features."""
    df = df.copy()

    # Ratios & proportions
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"] / (df["sleep_hours"] + EPS)
    df["social_share_of_screen"] = df["social_media_hours"] / (df["daily_screen_time_hours"] + EPS)
    df["gaming_share_of_screen"] = df["gaming_hours"] / (df["daily_screen_time_hours"] + EPS)
    df["productive_ratio"] = df["work_study_hours"] / (df["daily_screen_time_hours"] + EPS)
    df["weekend_to_weekday_ratio"] = df["weekend_screen_time"] / (df["daily_screen_time_hours"] + EPS)

    # Interaction & frequency metrics
    df["notifications_per_screen_hour"] = df["notifications_per_day"] / (df["daily_screen_time_hours"] + EPS)
    df["app_opens_per_screen_hour"] = df["app_opens_per_day"] / (df["daily_screen_time_hours"] + EPS)
    df["notifications_per_open"] = df["notifications_per_day"] / (df["app_opens_per_day"] + EPS)
    df["total_active_hours"] = df["daily_screen_time_hours"] + df["work_study_hours"] + df["sleep_hours"]

    # Missingness count
    feature_cols = RAW_NUM_COLS + RAW_CAT_COLS
    df["missing_count"] = df[feature_cols].isna().sum(axis=1)

    # Individual missingness flags
    for col in RAW_NUM_COLS:
        df[f"is_missing_{col}"] = df[col].isna().astype(np.float32)

    return df


def add_decimal_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract first decimal digit (generator fingerprint with float tolerance)."""
    df = df.copy()
    decimal_cols = [
        "daily_screen_time_hours", "social_media_hours", "gaming_hours",
        "work_study_hours", "sleep_hours", "weekend_screen_time",
    ]
    for col in decimal_cols:
        # First decimal digit: floor(value * 10 + 1e-6) % 10
        df[f"{col}_d1"] = np.where(
            df[col].isna(),
            np.nan,
            np.floor(df[col] * 10.0 + 1e-6) % 10
        )
    return df


def add_frequency_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Add frequency encoding for each column (count/proportion of each value).
    Frequency is computed on train only, then applied to both. This is a cheap
    orthogonal signal that complements target encoding.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    freq_cols = (
        ["gender_enc", "stress_level_enc", "academic_work_impact_enc"]
        + [f"{c}_d1" for c in [
            "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "weekend_screen_time",
        ]]
    )

    for col in freq_cols:
        freq = train_df[col].value_counts(normalize=True, dropna=False)
        fe_col = f"freq_{col}"
        train_df[fe_col] = train_df[col].map(freq).fillna(0).astype(np.float32)
        test_df[fe_col] = test_df[col].map(freq).fillna(0).astype(np.float32)

    return train_df, test_df


def oof_target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      col: str, target: str, folds, smoothing: float = 20.0):
    """Out-of-fold target encoding for a single column with smoothing."""
    global_mean = train_df[target].mean()
    train_enc = np.full(len(train_df), np.nan)
    test_enc = np.zeros(len(test_df))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        fold_train = train_df.iloc[tr_idx]
        agg = fold_train.groupby(col)[target].agg(["mean", "count"])
        smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        train_enc[va_idx] = train_df.iloc[va_idx][col].map(smooth).values
        test_enc += test_df[col].map(smooth).fillna(global_mean).values / N_FOLDS

    train_enc = np.where(np.isnan(train_enc), global_mean, train_enc)
    return train_enc, test_enc


def add_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame, folds):
    """
    Add OOF target encoding for:
    - 3 base categoricals
    - All numeric columns (binned into 20 buckets) -- the "lookup" channel
    - Interaction pairs
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Single-column target encoding (categoricals)
    for col in ["gender_enc", "stress_level_enc", "academic_work_impact_enc"]:
        te_col = f"te_{col}"
        train_df[te_col], test_df[te_col] = oof_target_encode(
            train_df, test_df, col, TARGET, folds
        )

    # Interaction target encoding
    interactions = [
        ("gender_enc", "stress_level_enc"),
        ("gender_enc", "academic_work_impact_enc"),
        ("stress_level_enc", "academic_work_impact_enc"),
    ]
    for col_a, col_b in interactions:
        interaction_col = f"{col_a}_x_{col_b}"
        train_df[interaction_col] = (
            train_df[col_a].astype(str) + "_" + train_df[col_b].astype(str)
        )
        test_df[interaction_col] = (
            test_df[col_a].astype(str) + "_" + test_df[col_b].astype(str)
        )
        te_col = f"te_{interaction_col}"
        train_df[te_col], test_df[te_col] = oof_target_encode(
            train_df, test_df, interaction_col, TARGET, folds
        )
        train_df.drop(columns=[interaction_col], inplace=True)
        test_df.drop(columns=[interaction_col], inplace=True)

    # Per-numeric-column binned target encoding ("lookup" channel)
    # Bin each numeric column into 20 equal-width buckets then TE-encode
    # This is the same signal the strong public `lookup_*` members use.
    for col in RAW_NUM_COLS:
        # Create binned version
        bin_col = f"{col}_bin20"
        all_vals = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        bin_edges = np.nanpercentile(all_vals.dropna(), np.linspace(0, 100, 21))
        bin_edges = np.unique(bin_edges)
        train_df[bin_col] = pd.cut(
            train_df[col], bins=bin_edges, labels=False, include_lowest=True
        ).astype("float32")
        test_df[bin_col] = pd.cut(
            test_df[col], bins=bin_edges, labels=False, include_lowest=True
        ).astype("float32")
        te_col = f"te_{bin_col}"
        train_df[te_col], test_df[te_col] = oof_target_encode(
            train_df, test_df, bin_col, TARGET, folds
        )
        train_df.drop(columns=[bin_col], inplace=True)
        test_df.drop(columns=[bin_col], inplace=True)

    return train_df, test_df


def get_tree_features(with_te: bool = True):
    """Get feature column names for tree models."""
    base = (
        RAW_NUM_COLS
        + ["gender_enc", "stress_level_enc", "academic_work_impact_enc"]
        + [
            "screen_to_sleep_ratio", "social_share_of_screen",
            "gaming_share_of_screen", "productive_ratio",
            "weekend_to_weekday_ratio", "notifications_per_screen_hour",
            "app_opens_per_screen_hour", "notifications_per_open",
            "total_active_hours", "missing_count",
        ]
        + [f"is_missing_{c}" for c in RAW_NUM_COLS]
        + [f"{c}_d1" for c in [
            "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "weekend_screen_time",
        ]]
        # Frequency encoding
        + [f"freq_gender_enc", "freq_stress_level_enc", "freq_academic_work_impact_enc"]
        + [f"freq_{c}_d1" for c in [
            "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "weekend_screen_time",
        ]]
    )
    if with_te:
        te_cols = (
            ["te_gender_enc", "te_stress_level_enc", "te_academic_work_impact_enc"]
            + ["te_gender_enc_x_stress_level_enc",
               "te_gender_enc_x_academic_work_impact_enc",
               "te_stress_level_enc_x_academic_work_impact_enc"]
            + [f"te_{c}_bin20" for c in RAW_NUM_COLS]
        )
        base = base + te_cols
    return base


# ================================================================================
# PHASE 3: BASE MODELS
# ================================================================================
def train_lgbm(train_X, train_y, test_X, folds, with_te: bool = True, seed: int = SEED):
    """Train LightGBM with 5-fold CV, return OOF and test predictions."""
    label = "LightGBM" + ("" if with_te else " (no TE)")
    print(f"\n{'='*60}")
    print(f"Training {label}...")
    print(f"{'='*60}")

    oof_preds = np.zeros(len(train_X))
    test_preds = np.zeros(len(test_X))

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(test_X)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
        print(f"  Fold {fold_idx + 1} AUC: {fold_auc:.6f} (best_iter={model.best_iteration_})")

    oof_auc = roc_auc_score(train_y, oof_preds)
    print(f"  {label} OOF AUC: {oof_auc:.6f}")
    return oof_preds, test_preds


def train_xgb(train_X, train_y, test_X, folds, with_te: bool = True, seed: int = SEED):
    """Train XGBoost with 5-fold CV, return OOF and test predictions."""
    label = "XGBoost" + ("" if with_te else " (no TE)")
    print(f"\n{'='*60}")
    print(f"Training {label}...")
    print(f"{'='*60}")

    oof_preds = np.zeros(len(train_X))
    test_preds = np.zeros(len(test_X))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 1.5,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(test_X)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
        best_iter = getattr(model, "best_iteration", model.n_estimators)
        print(f"  Fold {fold_idx + 1} AUC: {fold_auc:.6f} (best_iter={best_iter})")

    oof_auc = roc_auc_score(train_y, oof_preds)
    print(f"  {label} OOF AUC: {oof_auc:.6f}")
    return oof_preds, test_preds


def train_catboost(train_df, train_y, test_df, folds, with_te: bool = True, seed: int = SEED):
    """Train CatBoost with 5-fold CV, return OOF and test predictions."""
    label = "CatBoost" + ("" if with_te else " (no TE)")
    print(f"\n{'='*60}")
    print(f"Training {label}...")
    print(f"{'='*60}")

    cat_features = ["gender_cat", "stress_level_cat", "academic_work_impact_cat"]
    feature_cols = (
        RAW_NUM_COLS
        + cat_features
        + [
            "screen_to_sleep_ratio", "social_share_of_screen",
            "gaming_share_of_screen", "productive_ratio",
            "weekend_to_weekday_ratio", "notifications_per_screen_hour",
            "app_opens_per_screen_hour", "notifications_per_open",
            "total_active_hours", "missing_count",
        ]
        + [f"is_missing_{c}" for c in RAW_NUM_COLS]
        + [f"{c}_d1" for c in [
            "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "weekend_screen_time",
        ]]
        # Frequency encoding
        + ["freq_gender_enc", "freq_stress_level_enc", "freq_academic_work_impact_enc"]
        + [f"freq_{c}_d1" for c in [
            "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "weekend_screen_time",
        ]]
    )
    if with_te:
        feature_cols += (
            ["te_gender_enc", "te_stress_level_enc", "te_academic_work_impact_enc"]
            + ["te_gender_enc_x_stress_level_enc",
               "te_gender_enc_x_academic_work_impact_enc",
               "te_stress_level_enc_x_academic_work_impact_enc"]
            + [f"te_{c}_bin20" for c in RAW_NUM_COLS]
        )

    train_X = train_df[feature_cols]
    test_X = test_df[feature_cols]

    oof_preds = np.zeros(len(train_X))
    test_preds = np.zeros(len(test_X))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]

        model = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=3.0,
            random_strength=0.5,
            # Higher border_count + no one_hot forces CatBoost to use its own
            # ordered target statistics -- this is the "catnative" model that
            # got the highest coefficient (+0.749) in the diversity notebook.
            border_count=128,
            one_hot_max_size=2,
            eval_metric="AUC",
            random_seed=seed,
            verbose=0,
            early_stopping_rounds=100,
            task_type="CPU",
            thread_count=-1,
        )

        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features)

        oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(test_X)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
        best_iter = model.get_best_iteration()
        print(f"  Fold {fold_idx + 1} AUC: {fold_auc:.6f} (best_iter={best_iter})")

    oof_auc = roc_auc_score(train_y, oof_preds)
    print(f"  {label} OOF AUC: {oof_auc:.6f}")
    return oof_preds, test_preds


# ── Neural Network ──────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and ReLU."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(x + self.net(x)))


class TabularResNet(nn.Module):
    """Tabular ResNet for binary classification."""
    def __init__(self, input_dim, hidden_dim=256, n_blocks=3, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


def prepare_nn_data(train_df, test_df, feature_cols=None):
    """Prepare data for neural network and logistic regression models."""
    if feature_cols is None:
        feature_cols = RAW_NUM_COLS + RAW_CAT_COLS

    num_cols = [c for c in feature_cols if c not in RAW_CAT_COLS]
    cat_cols = [c for c in feature_cols if c in RAW_CAT_COLS]

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), num_cols),
    ]
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit on train, transform both
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
    )


def train_resnet(X_train, y_train, X_test, folds, seed=SEED,
                 epochs=80, hidden_dim=256, n_blocks=3):
    """Train TabularResNet with 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Training TabularResNet (epochs={epochs}, dim={hidden_dim}, blocks={n_blocks})...")
    print(f"{'='*60}")
    print(f"  Using device: {DEVICE}")
    print(f"  Input features: {X_train.shape[1]}")

    set_seed(seed)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        Xtr = torch.from_numpy(X_train[tr_idx]).to(DEVICE)
        ytr = torch.from_numpy(y_train[tr_idx].astype(np.float32)).to(DEVICE)
        Xva = torch.from_numpy(X_train[va_idx]).to(DEVICE)
        Xte = torch.from_numpy(X_test).to(DEVICE)

        train_ds = TensorDataset(Xtr, ytr)
        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)

        model = TabularResNet(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=0.25,
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_auc = -np.inf
        best_state = None

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                va_logits = model(Xva).cpu().numpy()
                va_probs = 1 / (1 + np.exp(-va_logits))
                va_auc = roc_auc_score(y_train[va_idx], va_probs)

            if va_auc > best_auc:
                best_auc = va_auc
                best_state = copy.deepcopy(model.state_dict())

        # Restore best and predict
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_preds[va_idx] = torch.sigmoid(model(Xva)).cpu().numpy()
            te_preds = []
            for i in range(0, len(X_test), 8192):
                chunk = Xte[i:i + 8192]
                te_preds.append(torch.sigmoid(model(chunk)).cpu().numpy())
            test_preds += np.concatenate(te_preds) / N_FOLDS

        print(f"  Fold {fold_idx + 1} AUC: {best_auc:.6f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"  TabularResNet OOF AUC: {oof_auc:.6f}")
    return oof_preds, test_preds


def train_logistic(X_train, y_train, X_test, folds, seed=SEED):
    """Train Ridge Logistic Regression with 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Training Logistic Ridge...")
    print(f"{'='*60}")

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr = y_train[tr_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(X_test)

        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000, random_state=seed
        )
        model.fit(Xtr_s, ytr)

        oof_preds[va_idx] = model.predict_proba(Xva_s)[:, 1]
        test_preds += model.predict_proba(Xte_s)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_train[va_idx], oof_preds[va_idx])
        print(f"  Fold {fold_idx + 1} AUC: {fold_auc:.6f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"  Logistic Ridge OOF AUC: {oof_auc:.6f}")
    return oof_preds, test_preds


# ================================================================================
# PHASE 4: PUBLIC OOF LIBRARY LOADING
# ================================================================================
def load_oof_pool(train_df, test_df, oof_dir: Path = OOF_DIR):
    """
    Load public OOF prediction libraries.

    Handles three file formats used by community members:
      1. .npy pairs: oof_<name>.npy + test_<name>.npy  (or reversed: <name>_oof.npy)
      2. .parquet pairs with an 'id' column for safe alignment
      3. CSV pairs: <k>_blend_oof_predictions.csv + <k>_blend_submission.csv

    Returns: dict of {name: (oof_array, test_array)}
    All arrays are float64, shape (n_train,) and (n_test,) respectively.
    """
    if not oof_dir.exists():
        print(f"  [WARN] OOF directory not found: {oof_dir}")
        print(f"         Run: python download_oof_libraries.py")
        return {}

    n_train = len(train_df)
    n_test = len(test_df)
    tr_id = train_df[ID_COL].values
    te_id = test_df[ID_COL].values

    members = {}

    def add(name, o, t):
        """Validate shape and add to members dict."""
        o = np.asarray(o, np.float64).ravel()
        t = np.asarray(t, np.float64).ravel()
        if o.shape != (n_train,) or t.shape != (n_test,):
            return
        if not (np.isfinite(o).all() and np.isfinite(t).all()):
            return
        members[name] = (o, t)

    # ── Format 1: .npy pairs ───────────────────────────────────────────────────
    for p in sorted(oof_dir.rglob("*.npy")):
        b = p.stem
        d = p.parent

        if b.startswith("oof_"):
            key = b[4:]
            test_p = d / f"test_{key}.npy"
        elif b.endswith("_oof"):
            key = b[:-4]
            test_p = d / f"{key}_test.npy"
        else:
            continue

        # Skip fold_id files
        if "fold_id" in b:
            continue

        if test_p.exists():
            try:
                add(key, np.load(p), np.load(test_p))
            except Exception:
                pass

    # ── Format 2: .parquet pairs (align by id) ─────────────────────────────────
    for p in sorted(oof_dir.rglob("*oof*.parquet")):
        # Replace only in FILENAME, not directory path
        test_p = p.parent / p.name.replace("oof", "test")
        if not test_p.exists() or test_p == p:
            continue
        try:
            do = pd.read_parquet(p)
            dt = pd.read_parquet(test_p)
            if ID_COL not in do.columns:
                continue
            do = do.set_index(ID_COL).reindex(tr_id)
            dt = dt.set_index(ID_COL).reindex(te_id)
            for col in do.columns:
                if col in dt.columns:
                    add(col, do[col].to_numpy(np.float64), dt[col].to_numpy(np.float64))
        except Exception:
            pass

    # ── Format 3: @najiama CSV blend pairs ────────────────────────────────────
    for p in sorted(oof_dir.rglob("*_blend_oof_predictions.csv")):
        k = p.name.split("_")[0]
        sub_p = p.parent / f"{k}_blend_submission.csv"
        if not sub_p.exists():
            continue
        try:
            do = pd.read_csv(p)
            dt = pd.read_csv(sub_p)
            oc = [c for c in do.columns if c.lower() != ID_COL]
            tc = [c for c in dt.columns if c.lower() != ID_COL]
            if not oc or not tc:
                continue
            if ID_COL in do.columns:
                do = do.set_index(ID_COL).reindex(tr_id)
            if ID_COL in dt.columns:
                dt = dt.set_index(ID_COL).reindex(te_id)
            add(f"naji_blend{k}", do[oc[0]].to_numpy(np.float64), dt[tc[0]].to_numpy(np.float64))
        except Exception:
            pass

    return members


def quarantine_pool(pool: dict, y: np.ndarray, n_train: int, n_test: int):
    """
    Apply three quarantine checks to the pool:

    1. Hash deduplication — exact byte-identical OOF arrays double-count one model.
       (The diversity notebook found 2 such duplicates in the public libraries.)
    2. KS-drift filter — members with KS statistic > 0.05 between rank-transformed
       OOF and test distributions signal train/test leakage or miscalibration.
       (knn, rf, extratrees failed this in the original analysis.)
    3. AUC floor — members with AUC < 0.90 are dropped, EXCEPT deliberate
       correctors (those with "perp" in their name).

    Returns: (keep_names, drop_summary)
    """
    rng = np.random.default_rng(0)
    ia = rng.choice(n_train, min(40000, n_train), replace=False)
    ib = rng.choice(n_test, min(40000, n_test), replace=False)

    R = lambda v: (rankdata(v, method="average") - 0.5) / len(v)

    # Step 1: hash deduplication
    seen_hash, dups = {}, []
    pool_deduped = {}
    for name in sorted(pool.keys()):
        o, t = pool[name]
        h = hashlib.md5(np.ascontiguousarray(o).tobytes()).hexdigest()
        if h in seen_hash:
            dups.append((name, seen_hash[h]))
        else:
            seen_hash[h] = name
            pool_deduped[name] = (o, t)

    if dups:
        print(f"\n  Exact duplicates removed: {len(dups)}")
        for a, b in dups:
            print(f"    {a}  ==  {b}")

    # Steps 2 & 3: KS filter + AUC floor
    keep, dropped = [], []
    for name, (o, t) in pool_deduped.items():
        corrector = "perp" in name
        au = roc_auc_score(y, o)

        if au < 0.90 and not corrector:
            dropped.append((name, f"auc={au:.4f}<0.90", au))
            continue

        ks = ks_2samp(R(o)[ia], R(t)[ib]).statistic
        if ks > 0.05 and not corrector:
            dropped.append((name, f"ks={ks:.3f}>0.05", au))
            continue

        keep.append(name)

    print(f"\n  Pool after quarantine: {len(keep)} kept, {len(dropped)} dropped")
    if dropped:
        print("  Dropped members:")
        for name, reason, au in dropped[:20]:
            print(f"    - {name:<40s} {reason} (auc={au:.4f})")
        if len(dropped) > 20:
            print(f"    ... and {len(dropped)-20} more")

    return keep, dropped


# ================================================================================
# PHASE 5: META-MODEL STACKING
# ================================================================================
def rank_gauss(v):
    """Rank-Gauss (normal quantile) transform."""
    r = (rankdata(v, method="average") - 0.5) / len(v)
    return norm.ppf(np.clip(r, 1e-7, 1 - 1e-7))


def nested_meta_model(oof_pool: dict, test_pool: dict, y: np.ndarray,
                      folds, C: float = 0.03):
    """
    Fit a nested L2 Logistic Regression meta-model over rank-Gauss features.

    oof_pool:  dict of {name: oof_predictions}
    test_pool: dict of {name: test_predictions}

    Returns: (nested_oof_auc, test_submission_probabilities)

    Design notes (from diversity notebook):
    - Rank-Gauss beats logit space by +0.00008 on large pools
    - StandardScaler is MANDATORY -- without it lbfgs diverges, and a
      non-converged fit reads HIGHER than truth (silent failure)
    - C=0.03 is fine; swept 0.003–1.0 moves the score by only 4e-6
    - Assert convergence rather than hoping
    """
    print(f"\n{'='*60}")
    print(f"Training Nested Meta-Model ({len(oof_pool)} base models)...")
    print(f"{'='*60}")

    names = sorted(oof_pool.keys())
    print(f"  Members: {len(names)}")

    # Build Rank-Gauss matrices
    G_oof = np.column_stack([rank_gauss(oof_pool[n]) for n in names])
    G_test = np.column_stack([rank_gauss(test_pool[n]) for n in names])

    # Nested CV for honest OOF score
    meta_oof = np.zeros(len(y))
    n_iters = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        sc = StandardScaler().fit(G_oof[tr_idx])
        m = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", tol=1e-5)
        m.fit(sc.transform(G_oof[tr_idx]), y[tr_idx])
        n_iters.append(int(np.max(m.n_iter_)))
        meta_oof[va_idx] = m.decision_function(sc.transform(G_oof[va_idx]))

    assert max(n_iters) < 5000, (
        f"Meta-model NOT converged (max_iter reached)! n_iters={n_iters}\n"
        f"A non-converged LR reads HIGHER than truth. Increase max_iter or reduce C."
    )
    nested_auc = roc_auc_score(y, meta_oof)
    print(f"  Nested OOF AUC: {nested_auc:.6f}")
    print(f"  Meta-model converged in {n_iters} iterations per fold")

    # Final meta-model on all data
    sc_final = StandardScaler().fit(G_oof)
    meta_final = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", tol=1e-5)
    meta_final.fit(sc_final.transform(G_oof), y)
    assert int(np.max(meta_final.n_iter_)) < 5000, "Final meta-model NOT converged!"

    # Test predictions -- rank transform for submission (AUC only cares about ordering)
    test_raw = meta_final.decision_function(sc_final.transform(G_test))
    test_ranked = (rankdata(test_raw, method="average") - 0.5) / len(test_raw)

    # Print top/bottom coefficients
    coef = pd.Series(meta_final.coef_[0], index=names).sort_values(ascending=False)
    print(f"\n  Top 10 positive coefficients:")
    for name, c in coef.head(10).items():
        print(f"    {name:<45s} {c:+.4f}")
    print(f"\n  Top 5 negative coefficients (correctors):")
    for name, c in coef.tail(5).items():
        print(f"    {name:<45s} {c:+.4f}")

    return nested_auc, test_ranked


# ================================================================================
# PHASE 6: MAIN PIPELINE
# ================================================================================
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    train_df, test_df = load_data()
    y = train_df[TARGET].to_numpy(np.int8)
    n_train, n_test = len(train_df), len(test_df)

    # ── Frozen fold split (CRITICAL: same split all public libraries use) ──────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(n_train), y))

    # ── Feature engineering ────────────────────────────────────────────────────
    print("\nEngineering features...")
    train_df = encode_categoricals(train_df)
    test_df = encode_categoricals(test_df)
    train_df = add_ratio_features(train_df)
    test_df = add_ratio_features(test_df)
    train_df = add_decimal_digit_features(train_df)
    test_df = add_decimal_digit_features(test_df)

    # Frequency encoding
    train_df, test_df = add_frequency_encoding(train_df, test_df)

    # Target encoding (OOF-safe) -- includes per-column binned "lookup" TE
    train_df, test_df = add_target_encoding(train_df, test_df, folds)

    # ── Prepare feature sets ───────────────────────────────────────────────────
    tree_feats_te = get_tree_features(with_te=True)
    tree_feats_no_te = get_tree_features(with_te=False)

    train_X_te = train_df[tree_feats_te]
    test_X_te = test_df[tree_feats_te]
    train_X_no_te = train_df[tree_feats_no_te]
    test_X_no_te = test_df[tree_feats_no_te]

    print(f"  Features with TE: {len(tree_feats_te)}")
    print(f"  Features without TE: {len(tree_feats_no_te)}")

    # ── Prepare NN/LR data (full engineered features for better diversity) ─────
    # Use all numeric engineered features (ratio + d1 + freq + te) for NN
    nn_feature_cols = tree_feats_te  # use full feature set (no cat strings)
    X_nn_train, X_nn_test = prepare_nn_data(train_df, test_df, feature_cols=nn_feature_cols)
    print(f"  NN/LR features: {X_nn_train.shape[1]}")

    # ── Train self-trained base models ─────────────────────────────────────────
    own_oof = {}
    own_test = {}

    # Model 1: LightGBM + TE
    own_oof["lgbm_te"], own_test["lgbm_te"] = train_lgbm(
        train_X_te, y, test_X_te, folds, with_te=True, seed=SEED
    )

    # Model 2: LightGBM - no TE
    own_oof["lgbm_no_te"], own_test["lgbm_no_te"] = train_lgbm(
        train_X_no_te, y, test_X_no_te, folds, with_te=False, seed=SEED
    )

    # Model 3: XGBoost + TE
    own_oof["xgb_te"], own_test["xgb_te"] = train_xgb(
        train_X_te, y, test_X_te, folds, with_te=True, seed=SEED
    )

    # Model 4: XGBoost - no TE (highest-coefficient model in diversity notebook)
    own_oof["xgb_no_te"], own_test["xgb_no_te"] = train_xgb(
        train_X_no_te, y, test_X_no_te, folds, with_te=False, seed=SEED
    )

    # Model 5: CatBoost + TE (with higher border_count for better native TS)
    own_oof["cat_te"], own_test["cat_te"] = train_catboost(
        train_df, y, test_df, folds, with_te=True, seed=SEED
    )

    # Model 6: CatBoost - no TE ("catnative" -- top +0.749 coefficient)
    own_oof["cat_no_te"], own_test["cat_no_te"] = train_catboost(
        train_df, y, test_df, folds, with_te=False, seed=SEED
    )

    # Model 7: TabularResNet (improved: more epochs, larger hidden, cosine LR)
    own_oof["resnet"], own_test["resnet"] = train_resnet(
        X_nn_train, y, X_nn_test, folds, seed=SEED,
        epochs=80, hidden_dim=256, n_blocks=3
    )

    # Model 8: Logistic Ridge (weak but high diversity, +0.137 coefficient in DBS)
    own_oof["logistic"], own_test["logistic"] = train_logistic(
        X_nn_train, y, X_nn_test, folds, seed=SEED
    )

    # ── Summary of self-trained base models ────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF-TRAINED BASE MODEL SUMMARY")
    print(f"{'='*60}")
    for name in sorted(own_oof.keys()):
        auc = roc_auc_score(y, own_oof[name])
        print(f"  {name:<35s} OOF AUC: {auc:.6f}")

    # ── Load public OOF libraries ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading public OOF libraries from {OOF_DIR}...")
    print(f"{'='*60}")

    public_pool = load_oof_pool(train_df, test_df, OOF_DIR)
    print(f"  Raw public members loaded: {len(public_pool)}")

    # ── Merge self-trained into public pool ────────────────────────────────────
    # Prefix our own models to avoid name collisions
    combined_pool = {}
    for name, (o, t) in public_pool.items():
        combined_pool[name] = (o, t)
    for name in own_oof:
        combined_pool[f"own_{name}"] = (own_oof[name], own_test[name])

    print(f"  Combined pool before quarantine: {len(combined_pool)} members")

    # ── Quarantine ─────────────────────────────────────────────────────────────
    keep_names, _ = quarantine_pool(combined_pool, y, n_train, n_test)

    # Build final OOF/test pools from kept names
    oof_pool = {n: combined_pool[n][0] for n in keep_names}
    test_pool = {n: combined_pool[n][1] for n in keep_names}

    print(f"\n  Final pool size: {len(oof_pool)} members")

    # ── Equal rank average baseline ────────────────────────────────────────────
    if len(oof_pool) > 0:
        rank_avg = np.column_stack([
            (rankdata(oof_pool[n]) - 0.5) / n_train for n in keep_names
        ]).mean(axis=1)
        rank_avg_auc = roc_auc_score(y, rank_avg)
        print(f"  Equal rank average OOF AUC: {rank_avg_auc:.6f}")

    # ── Meta-model ─────────────────────────────────────────────────────────────
    nested_auc, test_submission = nested_meta_model(oof_pool, test_pool, y, folds)

    # ── Generate submission ────────────────────────────────────────────────────
    sub = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET: test_submission,
    })
    assert len(sub) == n_test, f"Submission length mismatch: {len(sub)} vs {n_test}"
    assert np.isfinite(sub[TARGET]).all(), "Submission contains non-finite values!"

    sub_path = OUTPUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUBMISSION SAVED: {sub_path}")
    print(f"{'='*60}")
    print(f"  Rows: {len(sub):,}")
    print(f"  Distinct values: {sub[TARGET].nunique():,}")
    print(f"  Pool members used: {len(oof_pool)}")
    print(f"  Nested meta-model OOF AUC: {nested_auc:.6f}")
    print(f"  Expected LB (OOF + ~0.001 offset): ~{nested_auc + 0.001:.6f}")
    print(sub.head())
    print("\nDone!")


if __name__ == "__main__":
    main()
