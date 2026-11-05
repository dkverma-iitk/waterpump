
# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import time
import warnings
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numerical / ML
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-safe; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

import shap
import joblib
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from scipy.special import expit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# ─── CONFIGURATION ──────────────────────────────────────────────────────────
# =============================================================================
CSV_PATH        = "rul_hrs.csv"   # <-- set your dataset path here
TARGET_COL      = "rul"
FEATURE_PREFIX  = "sensor_"
UNIT_COL        = "unit_id"       # created automatically if absent

HORIZONS        = [24.0, 72.0, 168.0]   # prediction horizons in hours
H_PLOT          = 72.0                  # primary diagnostic horizon
TMAX            = 168.0                 # planning horizon / health score base

RUL_CAP         = 5000.0
N_ENSEMBLE      = 10          # ensemble size
N_REGIMES       = 3           # KMeans operating regimes
REGIME_SAMPLE   = 50_000      # max rows used to fit KMeans

# Calibration
CALIB_FRAC      = 0.25        # fraction of train data held for calibration
USE_REGIME_CALIB = True       # per-regime isotonic calibration

# Alert logic
ALERT_THRESHOLD         = 0.5   # default probability threshold
MIN_ALERT_DURATION_H    = 12.0  # minimum alert persistence in hours
PERSISTENCE_LOOKBACK    = 3     # rolling window for persistence smoothing (samples)

# Feature engineering toggles
USE_ROLLING_FEATURES    = True
ROLLING_WINDOWS         = [5, 20, 50]   # sample windows for rolling stats
USE_TREND_FEATURES      = True
TREND_WINDOW            = 30

# Cost model for threshold optimisation (ratio: cost_false_negative / cost_false_positive)
COST_RATIO              = 5.0

# SHAP
N_SHAP_SAMPLES          = 500   # cap for SHAP background (speed)
COMPUTE_SHAP_INTERACTION = True  # SHAP interaction values (slow for large datasets)
N_INTERACTION_SAMPLES   = 200

# Output
FIG_DIR     = "figures"
RESULTS_DIR = "results"
MODEL_DIR   = "saved_models"
FIG_DPI     = 180
FIG_FORMAT  = "png"

for d in [FIG_DIR, RESULTS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# ─── PUBLICATION STYLE ───────────────────────────────────────────────────────
# =============================================================================
plt.rcParams.update({
    "figure.dpi"       : FIG_DPI,
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 9,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "lines.linewidth"  : 1.6,
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# =============================================================================
# ─── HELPER: version log ─────────────────────────────────────────────────────
# =============================================================================
def log_versions():
    import sklearn, scipy
    vers = {
        "python"  : sys.version.split()[0],
        "numpy"   : np.__version__,
        "pandas"  : pd.__version__,
        "sklearn" : sklearn.__version__,
        "scipy"   : scipy.__version__,
        "shap"    : shap.__version__,
    }
    log.info("Package versions: %s", vers)
    pd.DataFrame([vers]).to_csv(os.path.join(RESULTS_DIR, "package_versions.csv"), index=False)

log_versions()

# =============================================================================
# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────
# =============================================================================
@dataclass
class CalibratedEnsemble:
    """Container for an ensemble of calibrated GB classifiers."""
    models      : List[Pipeline]
    calibrators : List              # IsotonicRegression per member
    feature_cols: List[str]
    regime_calibrators: Dict[int, List] = field(default_factory=dict)

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw (uncalibrated) probabilities [n_members, n_samples]."""
        return np.stack(
            [m.predict_proba(X[self.feature_cols])[:, 1] for m in self.models],
            axis=0
        )

    def predict_proba_calibrated(
        self,
        X: pd.DataFrame,
        regimes: Optional[np.ndarray] = None
    ) -> np.ndarray:

        probs = []
        for k, (mdl, cal) in enumerate(zip(self.models, self.calibrators)):
            p_raw = mdl.predict_proba(X[self.feature_cols])[:, 1]

            if USE_REGIME_CALIB and self.regime_calibrators and regimes is not None:
                p_cal = np.empty_like(p_raw)
                for rg, reg_cals in self.regime_calibrators.items():
                    mask = regimes == rg
                    if mask.sum() == 0:
                        continue
                    if k < len(reg_cals) and reg_cals[k] is not None:
                        p_cal[mask] = np.clip(reg_cals[k].predict(p_raw[mask]), 0.0, 1.0)
                    else:
                        p_cal[mask] = np.clip(cal.predict(p_raw[mask]), 0.0, 1.0)
                # fill any unmapped samples
                mapped = np.isin(regimes, list(self.regime_calibrators.keys()))
                if (~mapped).any():
                    p_cal[~mapped] = np.clip(cal.predict(p_raw[~mapped]), 0.0, 1.0)
            else:
                p_cal = np.clip(cal.predict(p_raw), 0.0, 1.0)

            probs.append(p_cal)
        return np.asarray(probs)

# =============================================================================
# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
# =============================================================================
def add_rolling_features(df: pd.DataFrame, feature_cols: List[str], unit_col: str) -> Tuple[pd.DataFrame, List[str]]:

    log.info("Engineering rolling & trend features ...")

    # ── Build the exact list of new column names (once, no duplicates) ───────
    new_col_names: List[str] = []
    for col in feature_cols:
        for w in ROLLING_WINDOWS:
            new_col_names.append(f"{col}_rmean{w}")
            new_col_names.append(f"{col}_rstd{w}")
        if USE_TREND_FEATURES:
            new_col_names.append(f"{col}_trend{TREND_WINDOW}")
    # deduplicate while preserving order (safety guard)
    seen: dict = {}
    new_col_names = [seen.setdefault(c, c) for c in new_col_names if c not in seen]

    parts = []
    for uid, grp in df.groupby(unit_col, sort=False):
        grp = grp.copy()
        engineered: Dict[str, np.ndarray] = {}

        for col in feature_cols:
            s = grp[col]
            for w in ROLLING_WINDOWS:
                engineered[f"{col}_rmean{w}"] = s.rolling(w, min_periods=1).mean().values
                engineered[f"{col}_rstd{w}"]  = s.rolling(w, min_periods=1).std().fillna(0.0).values
            if USE_TREND_FEATURES:
                trend_col = f"{col}_trend{TREND_WINDOW}"
                x = np.arange(len(grp), dtype=float)
                roll_mean_x = pd.Series(x).rolling(TREND_WINDOW, min_periods=2).mean().values
                roll_mean_y = s.rolling(TREND_WINDOW, min_periods=2).mean().values
                roll_cov = (
                    pd.Series(x * s.values).rolling(TREND_WINDOW, min_periods=2).mean().values
                    - roll_mean_x * roll_mean_y
                )
                roll_var = pd.Series(x).rolling(TREND_WINDOW, min_periods=2).var(ddof=0).values
                with np.errstate(divide="ignore", invalid="ignore"):
                    slope = np.where(roll_var > 0, roll_cov / roll_var, 0.0)
                slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
                engineered[trend_col] = slope

        # Assign all engineered columns to the group copy at once
        for cname, arr in engineered.items():
            grp[cname] = arr

        parts.append(grp)

    df_out = pd.concat(parts).sort_index()

    # Safe per-column fillna (avoids the multi-column assignment length mismatch)
    for c in new_col_names:
        if c in df_out.columns:
            df_out[c] = df_out[c].fillna(0.0)

    all_feature_cols = feature_cols + new_col_names
    log.info("Feature count: %d raw + %d engineered = %d total",
             len(feature_cols), len(new_col_names), len(all_feature_cols))
    return df_out, all_feature_cols

# =============================================================================
# ─── LABEL CREATION ──────────────────────────────────────────────────────────
# =============================================================================
def failure_label(rul: pd.Series, H: float) -> np.ndarray:
    return (rul <= H).astype(int).values

# =============================================================================
# ─── CALIBRATION SPLIT (STRATIFIED TEMPORAL) ─────────────────────────────────
# =============================================================================
def temporal_stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    calib_frac: float = CALIB_FRAC,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos_cal = max(1, int(len(pos_idx) * calib_frac))
    n_neg_cal = max(1, int(len(neg_idx) * calib_frac))

    cal_idx = np.concatenate([pos_idx[-n_pos_cal:], neg_idx[-n_neg_cal:]])
    fit_mask = np.ones(len(y), dtype=bool)
    fit_mask[cal_idx] = False
    fit_idx = np.where(fit_mask)[0]

    return (
        X.iloc[fit_idx], X.iloc[cal_idx],
        y[fit_idx],      y[cal_idx]
    )

# =============================================================================
# ─── BASE MODEL ──────────────────────────────────────────────────────────────
# =============================================================================
def make_base_model(
    random_state: int,
    feature_cols: List[str],
    class_weight: Optional[float] = None
) -> Pipeline:

    clf = GradientBoostingClassifier(
        n_estimators   = 300,
        learning_rate  = 0.04,
        max_depth      = 4,
        subsample      = 0.75,
        min_samples_leaf = 20,
        max_features   = "sqrt",
        random_state   = random_state,
    )
    pipe = Pipeline([
        ("scaler", RobustScaler()),   # RobustScaler: better for sensor outliers
        ("gb", clf)
    ])
    return pipe

# =============================================================================
# ─── SAMPLE WEIGHTS (class imbalance) ────────────────────────────────────────
# =============================================================================
def compute_sample_weights(y: np.ndarray, cost_ratio: float = COST_RATIO) -> np.ndarray:

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return np.ones(len(y))
    w_pos = cost_ratio * (n_neg / n_pos)
    w = np.where(y == 1, w_pos, 1.0)
    return w / w.mean()   # normalise to mean=1 to keep scale

# =============================================================================
# ─── ENSEMBLE TRAINING ───────────────────────────────────────────────────────
# =============================================================================
def train_calibrated_ensemble(
    X_train      : pd.DataFrame,
    y_train      : np.ndarray,
    feature_cols : List[str],
    regimes_train: Optional[np.ndarray] = None,
    seed0        : int = RANDOM_SEED,
) -> "CalibratedEnsemble":

    models, calibrators = [], []
    regime_calibrators: Dict[int, List] = {}

    sample_weights = compute_sample_weights(y_train)

    for k in range(N_ENSEMBLE):
        mdl = make_base_model(seed0 + k, feature_cols)
        X_fit, X_cal, y_fit, y_cal = temporal_stratified_split(X_train, y_train)
        sw_fit, _ = (
            compute_sample_weights(y_fit)[np.newaxis].flatten(),
            compute_sample_weights(y_cal)[np.newaxis].flatten()
        )
        sw_fit = compute_sample_weights(y_fit)

        mdl.fit(X_fit[feature_cols], y_fit, gb__sample_weight=sw_fit)

        p_cal_raw = mdl.predict_proba(X_cal[feature_cols])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(p_cal_raw, y_cal)

        models.append(mdl)
        calibrators.append(iso)

    ens = CalibratedEnsemble(
        models=models,
        calibrators=calibrators,
        feature_cols=feature_cols,
    )

    # ── Regime-aware calibration ───────────────────────────────────────────
    if USE_REGIME_CALIB and regimes_train is not None:
        unique_regimes = np.unique(regimes_train)
        for rg in unique_regimes:
            mask_rg = (regimes_train == rg)
            if mask_rg.sum() < 100:
                continue
            X_rg = X_train[mask_rg]
            y_rg = y_train[mask_rg]
            reg_cals_for_regime = []
            for k, mdl in enumerate(models):
                p_rg = mdl.predict_proba(X_rg[feature_cols])[:, 1]
                try:
                    iso_rg = IsotonicRegression(out_of_bounds="clip", increasing=True)
                    iso_rg.fit(p_rg, y_rg)
                    reg_cals_for_regime.append(iso_rg)
                except Exception:
                    reg_cals_for_regime.append(None)
            regime_calibrators[rg] = reg_cals_for_regime
        ens.regime_calibrators = regime_calibrators

    return ens

# =============================================================================
# ─── METRICS ─────────────────────────────────────────────────────────────────
# =============================================================================
def summarize_metrics(
    y_true : np.ndarray,
    p_mean : np.ndarray,
    prefix : str = ""
) -> Dict[str, float]:
    auc   = roc_auc_score(y_true, p_mean) if len(np.unique(y_true)) > 1 else np.nan
    ap    = average_precision_score(y_true, p_mean)
    brier = brier_score_loss(y_true, p_mean)

    # ── Brier Skill Score (vs. climatological baseline) ───────────────────
    clim  = y_true.mean()
    brier_ref = brier_score_loss(y_true, np.full_like(p_mean, clim))
    bss = 1.0 - brier / brier_ref if brier_ref > 0 else np.nan

    # ── MCB decomposition: reliability + resolution - uncertainty ─────────
    frac_pos, mean_pred = calibration_curve(y_true, p_mean, n_bins=10, strategy="uniform")
    weights = np.array([
        ((mean_pred[i] <= p_mean) & (p_mean < (mean_pred[i] + 0.1 if i < len(mean_pred)-1 else 1.001))).sum()
        for i in range(len(mean_pred))
    ])
    n = len(y_true)
    reliability = np.sum(weights / n * (mean_pred - frac_pos) ** 2) if len(mean_pred) > 0 else np.nan
    resolution  = np.sum(weights / n * (frac_pos - clim) ** 2)      if len(mean_pred) > 0 else np.nan
    uncertainty = clim * (1 - clim)

    return {
        f"{prefix}AUC"          : auc,
        f"{prefix}AP"           : ap,
        f"{prefix}Brier"        : brier,
        f"{prefix}BSS"          : bss,
        f"{prefix}Reliability"  : reliability,
        f"{prefix}Resolution"   : resolution,
        f"{prefix}Uncertainty"  : uncertainty,
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    N    = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / N) * abs(y[mask].mean() - p[mask].mean())
    return ece


def optimal_threshold(
    y_true   : np.ndarray,
    p_mean   : np.ndarray,
    cost_ratio: float = COST_RATIO,
    beta     : float = 2.0
) -> Tuple[float, float, float]:

    prec, rec, thresholds = precision_recall_curve(y_true, p_mean)
    f_beta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)
    best_f_idx = np.argmax(f_beta[:-1])
    f_thresh = float(thresholds[best_f_idx])
    best_fb  = float(f_beta[best_f_idx])

    # cost-sensitive: misclassification cost = FN * cost_ratio + FP * 1.0
    fpr, tpr, thresholds_roc = roc_curve(y_true, p_mean)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    fn_rate = 1.0 - tpr
    fp_rate = fpr
    cost = cost_ratio * fn_rate * n_pos + fp_rate * n_neg
    best_cost_idx = np.argmin(cost)
    cost_thresh   = float(thresholds_roc[best_cost_idx])

    return f_thresh, cost_thresh, best_fb


def brier_decomposition(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict:
    """Returns MCB decomposition: reliability, resolution, uncertainty."""
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    bins = np.linspace(0, 1, n_bins + 1)
    clim = y_true.mean()
    weights = []
    for i in range(len(mean_pred)):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi)
        weights.append(mask.sum())
    weights = np.array(weights)
    n = len(y_true)
    rel = np.sum(weights / n * (mean_pred - frac_pos) ** 2)
    res = np.sum(weights / n * (frac_pos - clim) ** 2)
    unc = clim * (1 - clim)
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier_decomp_check": rel - res + unc}

# =============================================================================
# ─── CALIBRATION BIN TABLE ───────────────────────────────────────────────────
# =============================================================================
def calib_bins_table(y_true: np.ndarray, p_med: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    rows    = []
    N       = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (p_med >= lo) & (p_med <= hi) if i == n_bins - 1 else (p_med >= lo) & (p_med < hi)
        count  = int(mask.sum())
        obs    = float(y_true[mask].mean()) if count > 0 else np.nan
        rows.append({
            "bin"           : i + 1,
            "pred_center"   : round(float(centers[i]), 2),
            "observed_rate" : round(obs, 4) if not np.isnan(obs) else np.nan,
            "count"         : count,
            "ci_low"        : round(max(0, obs - 1.96 * np.sqrt(obs*(1-obs)/count)), 4) if count > 0 and not np.isnan(obs) else np.nan,
            "ci_high"       : round(min(1, obs + 1.96 * np.sqrt(obs*(1-obs)/count)), 4) if count > 0 and not np.isnan(obs) else np.nan,
        })
    return pd.DataFrame(rows)

# =============================================================================
# ─── SURVIVAL & WEIBULL ──────────────────────────────────────────────────────
# =============================================================================
def probs_to_survival(horizons: List[float], p_fail_leq: Dict[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    hs = np.array(sorted(horizons))
    S  = np.asarray([1.0 - np.clip(p_fail_leq[h], 0.0, 1.0) for h in hs])
    return hs, S


def fit_weibull_from_survival(hs: np.ndarray, S_median: np.ndarray) -> Tuple[float, float]:

    valid = (S_median > 0) & (S_median < 1)
    if valid.sum() < 2:
        return np.nan, np.nan
    log_h  = np.log(hs[valid])
    log_nS = np.log(-np.log(S_median[valid]))
    try:
        slope, intercept, *_ = stats.linregress(log_h, log_nS)
        beta = float(slope)
        eta  = float(np.exp(-intercept / beta)) if beta != 0 else np.nan
        return beta, eta
    except Exception:
        return np.nan, np.nan

# =============================================================================
# ─── ALERT DETECTION ─────────────────────────────────────────────────────────
# =============================================================================
def detect_alert_windows(p: np.ndarray, threshold: float, min_len: int) -> List[Tuple[int, int]]:
    alert  = p >= threshold
    # smooth with rolling persistence (avoids flicker at boundary)
    roll   = pd.Series(alert.astype(float)).rolling(PERSISTENCE_LOOKBACK, min_periods=1).mean().values
    smooth = roll >= 0.5
    windows, start = [], None
    for i, flag in enumerate(smooth):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                windows.append((start, i - 1))
            start = None
    if start is not None and len(smooth) - start >= min_len:
        windows.append((start, len(smooth) - 1))
    return windows


def probs_to_expected_rul(p: np.ndarray, cap: float) -> np.ndarray:
    return np.clip((1.0 - p) * cap, 0.0, cap)

# =============================================================================
# ─── LOAD DATA ───────────────────────────────────────────────────────────────
# =============================================================================
log.info("Loading data from: %s", CSV_PATH)
df = pd.read_csv(CSV_PATH)
log.info("Raw dataset shape: %s", df.shape)

raw_feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
assert TARGET_COL in df.columns, f"Missing target column '{TARGET_COL}'"

# ── Auto-detect unit IDs via RUL resets ──────────────────────────────────────
if UNIT_COL not in df.columns:
    log.info("No '%s' column — auto-creating by detecting RUL resets.", UNIT_COL)
    unit_id, units, prev_rul = 0, [], df[TARGET_COL].iloc[0]
    for r in df[TARGET_COL].values:
        if r > prev_rul + 5:   # hysteresis to avoid noise triggering
            unit_id += 1
        units.append(unit_id)
        prev_rul = r
    df[UNIT_COL] = units

unit_ids = sorted(df[UNIT_COL].unique())
log.info("Units detected: %d", len(unit_ids))

# ── Z-score normalisation (raw features only, before engineering) ─────────────
scaler_global = StandardScaler()
df[raw_feature_cols] = scaler_global.fit_transform(df[raw_feature_cols])

# ── Feature engineering ───────────────────────────────────────────────────────
if USE_ROLLING_FEATURES or USE_TREND_FEATURES:
    df, feature_cols = add_rolling_features(df, raw_feature_cols, UNIT_COL)
else:
    feature_cols = raw_feature_cols

X_all  = df[feature_cols]
rul_all = df[TARGET_COL]

log.info("Total samples: %d | Features: %d", len(df), len(feature_cols))

# =============================================================================
# ─── TRAIN / TEST SPLIT ──────────────────────────────────────────────────────
# =============================================================================
train_units = unit_ids[:-1]
test_units  = unit_ids[-1:]

train_mask  = df[UNIT_COL].isin(train_units)
test_mask   = df[UNIT_COL].isin(test_units)

X_train     = X_all[train_mask].reset_index(drop=True)
X_test      = X_all[test_mask].reset_index(drop=True)
rul_train   = rul_all[train_mask].reset_index(drop=True)
rul_test    = rul_all[test_mask].reset_index(drop=True)

log.info("Train rows: %d (%d units) | Test rows: %d (%d units)",
         len(X_train), len(train_units), len(X_test), len(test_units))

# =============================================================================
# ─── OPERATING REGIME DETECTION ──────────────────────────────────────────────
# =============================================================================
log.info("Fitting KMeans regime detector (k=%d) ...", N_REGIMES)
scaler_reg  = StandardScaler()
Xtr_scaled  = scaler_reg.fit_transform(X_train[raw_feature_cols])

n_sub       = min(REGIME_SAMPLE, len(X_train))
rng         = np.random.default_rng(RANDOM_SEED)
sub_idx     = rng.choice(len(X_train), size=n_sub, replace=False)

kmeans = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_SEED, n_init=20, max_iter=500)
kmeans.fit(Xtr_scaled[sub_idx])

Xte_scaled      = scaler_reg.transform(X_test[raw_feature_cols])
Xtr_all_scaled  = scaler_reg.transform(X_train[raw_feature_cols])
train_regimes   = kmeans.predict(Xtr_all_scaled)
test_regimes    = kmeans.predict(Xte_scaled)

log.info("Regime distribution (test): %s",
         dict(zip(*np.unique(test_regimes, return_counts=True))))

# =============================================================================
# ─── TRAIN CALIBRATED ENSEMBLES ──────────────────────────────────────────────
# =============================================================================
ensembles              : Dict[float, CalibratedEnsemble]       = {}
metrics_dict           : Dict[float, Dict]                     = {}
p_calibrated           : Dict[float, np.ndarray]               = {}
p_calibrated_members   : Dict[float, np.ndarray]               = {}

for H in HORIZONS:
    y_train_H = failure_label(rul_train, H)
    y_test_H  = failure_label(rul_test,  H)
    pos_rate  = y_train_H.mean()
    log.info("Horizon H=%dh | train positive rate=%.4f (n_pos=%d)", int(H), pos_rate, y_train_H.sum())

    ens = train_calibrated_ensemble(
        X_train, y_train_H, feature_cols,
        regimes_train=train_regimes,
        seed0=RANDOM_SEED + int(H)
    )
    ensembles[H] = ens

    members   = ens.predict_proba_calibrated(X_test, regimes=test_regimes)   # [m, n]
    p_mean    = members.mean(axis=0)

    p_calibrated[H]         = p_mean
    p_calibrated_members[H] = members

    m = summarize_metrics(y_test_H, p_mean)
    metrics_dict[H] = m
    log.info("H=%dh -> AUC=%.4f | AP=%.4f | Brier=%.4f | BSS=%.4f",
             int(H), m["AUC"], m["AP"], m["Brier"], m["BSS"])

# =============================================================================
# ─── LEAVE-ONE-UNIT-OUT CROSS-VALIDATION ─────────────────────────────────────
# =============================================================================
log.info("Starting leave-one-unit-out cross-validation ...")
cv_rows = []

for H in HORIZONS:
    fold_results = []
    for test_unit in unit_ids:
        train_u   = [u for u in unit_ids if u != test_unit]
        tr_mask   = df[UNIT_COL].isin(train_u)
        te_mask   = df[UNIT_COL] == test_unit

        X_tr_cv   = X_all[tr_mask].reset_index(drop=True)
        X_te_cv   = X_all[te_mask].reset_index(drop=True)
        rul_tr_cv = rul_all[tr_mask].reset_index(drop=True)
        rul_te_cv = rul_all[te_mask].reset_index(drop=True)

        y_tr = failure_label(rul_tr_cv, H)
        y_te = failure_label(rul_te_cv, H)

        if len(np.unique(y_te)) < 2:
            log.info("  Unit %s H=%dh: skipped (no positives in test)", test_unit, int(H))
            continue

        Xtr_reg_cv = scaler_reg.transform(X_tr_cv[raw_feature_cols])
        Xte_reg_cv = scaler_reg.transform(X_te_cv[raw_feature_cols])
        km_cv = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_SEED, n_init=10)
        km_cv.fit(Xtr_reg_cv[:min(REGIME_SAMPLE, len(X_tr_cv))])
        reg_tr_cv = km_cv.predict(Xtr_reg_cv)
        reg_te_cv = km_cv.predict(Xte_reg_cv)

        ens_cv = train_calibrated_ensemble(
            X_tr_cv, y_tr, feature_cols,
            regimes_train=reg_tr_cv,
            seed0=RANDOM_SEED + int(H)
        )
        members_cv = ens_cv.predict_proba_calibrated(X_te_cv, regimes=reg_te_cv)
        p_cv       = members_cv.mean(axis=0)

        m_cv       = summarize_metrics(y_te, p_cv)
        ece_cv     = expected_calibration_error(y_te, p_cv)
        m_cv["ECE"] = ece_cv
        m_cv["unit"] = test_unit
        m_cv["H"]    = int(H)
        fold_results.append(m_cv)
        log.info("  Unit %s H=%dh | AUC=%.4f AP=%.4f ECE=%.4f BSS=%.4f",
                 test_unit, int(H), m_cv["AUC"], m_cv["AP"], ece_cv, m_cv["BSS"])

    if fold_results:
        fold_df = pd.DataFrame(fold_results)
        summary_row = {"H": int(H)}
        for metric in ["AUC", "AP", "Brier", "BSS", "ECE"]:
            vals = fold_df[metric].dropna()
            summary_row[f"{metric}_mean"] = vals.mean()
            summary_row[f"{metric}_std"]  = vals.std()
            # 95% CI via t-distribution
            n = len(vals)
            if n > 1:
                se = vals.std() / np.sqrt(n)
                summary_row[f"{metric}_ci95_low"]  = vals.mean() - 1.96 * se
                summary_row[f"{metric}_ci95_high"] = vals.mean() + 1.96 * se
        cv_rows.append(summary_row)
        log.info("H=%dh LOUO | AUC=%.4f±%.4f | AP=%.4f±%.4f | BSS=%.4f±%.4f",
                 int(H),
                 summary_row["AUC_mean"], summary_row["AUC_std"],
                 summary_row["AP_mean"],  summary_row["AP_std"],
                 summary_row["BSS_mean"], summary_row["BSS_std"])

cv_df = pd.DataFrame(cv_rows).set_index("H")
cv_df.to_csv(os.path.join(RESULTS_DIR, "louo_cv_metrics.csv"))
with open(os.path.join(RESULTS_DIR, "louo_cv_metrics.tex"), "w") as f:
    f.write(cv_df.to_latex(float_format="%.4f",
                            caption="Leave-one-unit-out cross-validation results (mean $\\pm$ std across folds).",
                            label="tab:louo_cv"))
log.info("LOUO CV metrics saved.")

# =============================================================================
# ─── THRESHOLD OPTIMISATION ──────────────────────────────────────────────────
# =============================================================================
y_test_H_plot = failure_label(rul_test, H_PLOT)
p_H_plot      = p_calibrated[H_PLOT]
members_H_plot= p_calibrated_members[H_PLOT]
p10           = np.percentile(members_H_plot, 10, axis=0)
p50           = np.percentile(members_H_plot, 50, axis=0)
p90           = np.percentile(members_H_plot, 90, axis=0)

f_thresh, cost_thresh, best_fb = optimal_threshold(y_test_H_plot, p_H_plot)
log.info("Optimal F2 threshold: %.3f | Cost-sensitive threshold: %.3f | F2=%.4f",
         f_thresh, cost_thresh, best_fb)

# Infer sampling interval
sampling_interval_hours = 1.0
try:
    dr = -np.diff(rul_test.values)
    dr_pos = dr[dr > 0]
    if len(dr_pos) > 0:
        sampling_interval_hours = float(np.median(dr_pos))
except Exception:
    pass
log.info("Inferred sampling interval: %.4f h/sample", sampling_interval_hours)

min_len_samples = max(1, int(np.ceil(MIN_ALERT_DURATION_H / sampling_interval_hours)))

# Alert windows at default and cost-optimised thresholds
alert_windows       = detect_alert_windows(p50, threshold=ALERT_THRESHOLD, min_len=min_len_samples)
alert_windows_cost  = detect_alert_windows(p50, threshold=cost_thresh,     min_len=min_len_samples)

def summarise_alert_windows(windows, dt):
    if not windows:
        return {}
    durs = [(e - s + 1) * dt for s, e in windows]
    return {
        "n_windows"    : len(durs),
        "mean_h"       : float(np.mean(durs)),
        "median_h"     : float(np.median(durs)),
        "min_h"        : float(np.min(durs)),
        "max_h"        : float(np.max(durs)),
        "total_h"      : float(np.sum(durs)),
    }

alert_stats      = summarise_alert_windows(alert_windows,      sampling_interval_hours)
alert_stats_cost = summarise_alert_windows(alert_windows_cost, sampling_interval_hours)

# =============================================================================
# ─── PERMUTATION IMPORTANCE ──────────────────────────────────────────────────
# =============================================================================
log.info("Computing permutation importance ...")
ens0     = ensembles[H_PLOT]
base_mdl = ens0.models[0]
try:
    perm_result = permutation_importance(
        base_mdl, X_test[feature_cols], y_test_H_plot,
        n_repeats=10, random_state=RANDOM_SEED, scoring="average_precision"
    )
    perm_importance_mean = perm_result.importances_mean
    perm_importance_std  = perm_result.importances_std
except Exception as e:
    log.warning("Permutation importance failed: %s", e)
    perm_importance_mean = np.zeros(len(feature_cols))
    perm_importance_std  = np.zeros(len(feature_cols))

# =============================================================================
# ─── SHAP ────────────────────────────────────────────────────────────────────
# =============================================================================
log.info("Computing SHAP values ...")
est_name   = "gb" if "gb" in base_mdl.named_steps else "hgb"
base_est   = base_mdl.named_steps[est_name]
X_test_sc  = base_mdl.named_steps["scaler"].transform(X_test[feature_cols])
X_test_sc_df = pd.DataFrame(X_test_sc, columns=feature_cols)

# ── Sample a subset for SHAP to keep runtime manageable ─────────────────────
shap_sample_size = min(N_SHAP_SAMPLES, len(X_test_sc_df))
shap_idx         = rng.choice(len(X_test_sc_df), size=shap_sample_size, replace=False)
X_shap           = X_test_sc_df.iloc[shap_idx].reset_index(drop=True)

explainer = shap.TreeExplainer(
    base_est,
    feature_perturbation="tree_path_dependent",  # correct for sklearn GBM
    model_output="raw",                           # log-odds: additive by construction
)

try:
    shap_values = explainer.shap_values(X_shap, check_additivity=True)
except Exception:
    log.warning("SHAP additivity check failed — retrying with check_additivity=False")
    shap_values = explainer.shap_values(X_shap, check_additivity=False)

# Handle multi-class / list output
if isinstance(shap_values, list):
    shap_arr = np.array(shap_values[1]) if len(shap_values) > 1 else np.array(shap_values[0])
else:
    shap_arr = np.array(shap_values)
if shap_arr.ndim == 3:
    shap_arr = shap_arr[:, :, 1] if shap_arr.shape[2] == 2 else shap_arr.reshape(shap_arr.shape[0], -1)
if shap_arr.shape[0] != X_shap.shape[0]:
    shap_arr = shap_arr.T

mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
mean_shap     = np.mean(shap_arr, axis=0)
total_shap    = mean_abs_shap.sum()
top_idx_shap  = np.argsort(-mean_abs_shap)[:10]

shap_top10 = pd.DataFrame([{
    "rank"          : r + 1,
    "feature"       : feature_cols[i],
    "mean_abs_shap" : float(mean_abs_shap[i]),
    "contribution_pct": float(100 * mean_abs_shap[i] / total_shap),
    "direction"     : "increase" if mean_shap[i] > 0 else "decrease",
    "perm_importance": float(perm_importance_mean[i]),
    "perm_std"      : float(perm_importance_std[i]),
} for r, i in enumerate(top_idx_shap)])

shap_top10.to_csv(os.path.join(RESULTS_DIR, "shap_top10.csv"), index=False)
log.info("Top SHAP feature: %s (%.1f%%)", shap_top10.iloc[0].feature, shap_top10.iloc[0].contribution_pct)

# ── SHAP interactions for top 5 features ──────────────────────────────────
if COMPUTE_SHAP_INTERACTION:
    try:
        log.info("Computing SHAP interaction values (n=%d) ...", N_INTERACTION_SAMPLES)
        interact_size = min(N_INTERACTION_SAMPLES, len(X_shap))
        interact_idx  = rng.choice(len(X_shap), size=interact_size, replace=False)
        shap_interact = explainer.shap_interaction_values(X_shap.iloc[interact_idx])
        if isinstance(shap_interact, list):
            shap_interact = shap_interact[1] if len(shap_interact) > 1 else shap_interact[0]
        # Extract top-5 x top-5 sub-matrix (mean |interaction|)
        top5      = top_idx_shap[:5]
        interact_mat = np.mean(np.abs(shap_interact[:, top5, :][:, :, top5]), axis=0)
        interact_df  = pd.DataFrame(interact_mat, index=[feature_cols[i] for i in top5],
                                    columns=[feature_cols[i] for i in top5])
        interact_df.to_csv(os.path.join(RESULTS_DIR, "shap_interaction_top5.csv"))
    except Exception as e:
        log.warning("SHAP interaction failed: %s", e)
        interact_df = None
else:
    interact_df = None

# ── Regime-specific SHAP ──────────────────────────────────────────────────
# shap_arr rows correspond to X_shap (the sampled subset), so align regimes
regime_shap_summaries = {}
test_regimes_shap = test_regimes[shap_idx]   # regimes for the SHAP sample rows
for rg in np.unique(test_regimes_shap):
    mask_rg = test_regimes_shap == rg
    if mask_rg.sum() < 30:
        continue
    shap_rg = shap_arr[mask_rg]
    regime_shap_summaries[rg] = {
        "mean_abs_shap" : np.mean(np.abs(shap_rg), axis=0),
        "n"             : mask_rg.sum()
    }

# =============================================================================
# ─── SURVIVAL & WEIBULL ──────────────────────────────────────────────────────
# =============================================================================
hs, S_mat = probs_to_survival(HORIZONS, p_calibrated)
S_median  = np.median(S_mat, axis=1)
weibull_beta, weibull_eta = fit_weibull_from_survival(hs, S_median)
log.info("Weibull fit: shape=%.3f, scale=%.3f", weibull_beta, weibull_eta)

# =============================================================================
# ─── BRIER DECOMPOSITION ─────────────────────────────────────────────────────
# =============================================================================
brier_decomp = brier_decomposition(y_test_H_plot, p_H_plot)

# =============================================================================
# ─── ECE AND SUMMARY TABLE ───────────────────────────────────────────────────
# =============================================================================
ece_H_plot = expected_calibration_error(y_test_H_plot, p_H_plot)
unc_width  = p90 - p10

summary_stats = {
    "H_plot"        : int(H_PLOT),
    "AUC"           : metrics_dict[H_PLOT]["AUC"],
    "AP"            : metrics_dict[H_PLOT]["AP"],
    "Brier"         : metrics_dict[H_PLOT]["Brier"],
    "BSS"           : metrics_dict[H_PLOT]["BSS"],
    "ECE"           : ece_H_plot,
    "Reliability"   : brier_decomp["reliability"],
    "Resolution"    : brier_decomp["resolution"],
    "Uncertainty"   : brier_decomp["uncertainty"],
    "MeanUncWidth"  : float(unc_width.mean()),
    "F2_threshold"  : f_thresh,
    "CostOpt_threshold": cost_thresh,
    "Weibull_beta"  : weibull_beta,
    "Weibull_eta"   : weibull_eta,
}
summary_stats.update({f"alert_{k}": v for k, v in alert_stats.items()})

pd.DataFrame([summary_stats]).to_csv(
    os.path.join(RESULTS_DIR, "summary_H_plot.csv"), index=False
)

# Horizon metrics table
metrics_rows = []
for H in HORIZONS:
    row = {"H": int(H)}
    row.update(metrics_dict[H])
    row["ECE"] = expected_calibration_error(failure_label(rul_test, H), p_calibrated[H])
    metrics_rows.append(row)
metrics_df = pd.DataFrame(metrics_rows).set_index("H")
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_per_horizon.csv"))
with open(os.path.join(RESULTS_DIR, "metrics_per_horizon.tex"), "w") as f:
    f.write(metrics_df.to_latex(float_format="%.4f",
                                 caption="Failure prediction metrics by horizon (held-out test unit).",
                                 label="tab:metrics_horizon"))

# Alert windows detail table
if alert_windows:
    aw_rows = []
    for i, (s, e) in enumerate(alert_windows, 1):
        dur = (e - s + 1) * sampling_interval_hours
        aw_rows.append({"alert": i, "start_t": s, "end_t": e,
                        "duration_h": round(dur, 2), "duration_days": round(dur/24, 2)})
    pd.DataFrame(aw_rows).to_csv(os.path.join(RESULTS_DIR, "alert_windows.csv"), index=False)

# SHAP top-10 LaTeX
with open(os.path.join(RESULTS_DIR, "shap_top10.tex"), "w") as f:
    f.write(shap_top10.to_latex(index=False, float_format="%.4f",
                                 caption="Top 10 sensors by mean $|$SHAP$|$ contribution (H=72h model).",
                                 label="tab:shap_top10"))

# Calibration bins
calib_df = calib_bins_table(y_test_H_plot, p_H_plot)
calib_df.to_csv(os.path.join(RESULTS_DIR, f"calibration_bins_H{int(H_PLOT)}.csv"), index=False)

# =============================================================================
# ─── FIGURES ─────────────────────────────────────────────────────────────────
# =============================================================================
def save_fig(name: str):
    path = os.path.join(FIG_DIR, f"{name}.{FIG_FORMAT}")
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved figure: %s", path)

# ── Figure 1: Calibration curves (multi-horizon) ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for i, H in enumerate(HORIZONS):
    y_H   = failure_label(rul_test, H)
    fp, mp = calibration_curve(y_H, p_calibrated[H], n_bins=10, strategy="uniform")
    ax.plot(mp, fp, marker="o", color=PALETTE[i], label=f"H={int(H)}h")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves – Multi-Horizon")
ax.legend()
save_fig("fig1_calibration_multi_horizon")

# ── Figure 2: Regime-wise reliability ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
y_plot = failure_label(rul_test, H_PLOT)
for rg in range(N_REGIMES):
    mask = test_regimes == rg
    if mask.sum() < 50:
        continue
    fp, mp = calibration_curve(y_plot[mask], p_H_plot[mask], n_bins=8, strategy="uniform")
    ax.plot(mp, fp, marker="o", color=PALETTE[rg], label=f"Regime {rg} (n={mask.sum()})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title(f"Regime-Wise Reliability (H={int(H_PLOT)}h)")
ax.legend()
save_fig("fig2_regime_reliability")

# ── Figure 3: Brier decomposition bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
components = ["Reliability\n(↓ better)", "Resolution\n(↑ better)", "Uncertainty\n(baseline)"]
vals = [brier_decomp["reliability"], brier_decomp["resolution"], brier_decomp["uncertainty"]]
colors = ["#d62728", "#2ca02c", "#7f7f7f"]
bars = ax.bar(components, vals, color=colors, edgecolor="black", width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score")
ax.set_title(f"Brier Score Decomposition (H={int(H_PLOT)}h)")
save_fig("fig3_brier_decomposition")

# ── Figure 4: Probability trajectory + uncertainty + alert threshold ──────────
t_idx = np.arange(len(p50))
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(t_idx, p50, color=PALETTE[0], label=f"P(Failure ≤{int(H_PLOT)}h) – median")
ax.fill_between(t_idx, p10, p90, alpha=0.2, color=PALETTE[0], label="P10–P90 band")
ax.axhline(ALERT_THRESHOLD, color="red", ls="--", lw=1.2, label=f"Default threshold ({ALERT_THRESHOLD})")
ax.axhline(cost_thresh,     color="orange", ls=":", lw=1.2,
           label=f"Cost-opt. threshold ({cost_thresh:.2f})")
ax.set_ylabel("Failure Probability")
ax.set_title(f"Failure-Within-{int(H_PLOT)}h Probability – Unit {test_units[0]}")
ax.legend(ncol=2)
# Uncertainty width panel
ax2 = axes[1]
ax2.plot(t_idx, p90 - p10, color="#7f7f7f", lw=1)
ax2.set_xlabel("Time Index")
ax2.set_ylabel("Uncertainty\n(P90–P10)")
plt.tight_layout()
save_fig("fig4_prob_trajectory_uncertainty")

# ── Figure 5: Alert timeline ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t_idx, p50, color=PALETTE[0], lw=1, label=f"P(Failure ≤{int(H_PLOT)}h)")
ax.fill_between(t_idx, p10, p90, alpha=0.2, color=PALETTE[0])
ax.axhline(ALERT_THRESHOLD, color="red", ls="--", lw=1.2, label=f"Threshold ({ALERT_THRESHOLD})")
for s, e in alert_windows:
    ax.axvspan(s, e, color="red", alpha=0.20)
ax.set_xlabel("Time Index")
ax.set_ylabel("Probability")
ax.set_title(f"Alert Timeline – H={int(H_PLOT)}h (min duration={MIN_ALERT_DURATION_H}h)")
ax.legend()
save_fig("fig5_alert_timeline")

# ── Figure 6: Expected RUL vs true RUL ───────────────────────────────────────
true_rul  = rul_test.values
exp_rul   = probs_to_expected_rul(p50, RUL_CAP)
exp_rul_lo= probs_to_expected_rul(p90, RUL_CAP)
exp_rul_hi= probs_to_expected_rul(p10, RUL_CAP)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_idx, true_rul,  color="steelblue", lw=1.5, label="True RUL (h)")
ax.plot(t_idx, exp_rul,   color=PALETTE[1],  lw=1.2, label=f"Expected RUL proxy (H={int(H_PLOT)}h, cap={int(RUL_CAP)}h)")
ax.fill_between(t_idx, exp_rul_lo, exp_rul_hi, alpha=0.2, color=PALETTE[1], label="P10–P90 band")
ax.set_xlabel("Time Index")
ax.set_ylabel("Hours")
ax.set_title(f"Expected RUL (capped) vs True RUL – Unit {test_units[0]}")
ax.legend()
save_fig("fig6_expected_rul_vs_true")

# ── Figure 7: Health score ────────────────────────────────────────────────────
p_tmax_members = p_calibrated_members[TMAX]
p_tmax_50 = np.percentile(p_tmax_members, 50, axis=0)
p_tmax_10 = np.percentile(p_tmax_members, 10, axis=0)
p_tmax_90 = np.percentile(p_tmax_members, 90, axis=0)
health_50 = 100.0 * (1.0 - p_tmax_50)
health_lo = 100.0 * (1.0 - p_tmax_90)
health_hi = 100.0 * (1.0 - p_tmax_10)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_idx, health_50, color=PALETTE[2], label="Health Score (0–100)")
ax.fill_between(t_idx, health_lo, health_hi, alpha=0.2, color=PALETTE[2], label="P10–P90 band")
ax.axhline(50, color="orange", ls="--", lw=1, label="Warning level (50)")
ax.axhline(20, color="red",    ls="--", lw=1, label="Critical level (20)")
ax.set_xlabel("Time Index")
ax.set_ylabel("Health Score")
ax.set_ylim(-5, 105)
ax.set_title(f"Deployable Health Score – Unit {test_units[0]} (Tmax={int(TMAX)}h)")
ax.legend(ncol=2)
save_fig("fig7_health_score")

# ── Figure 8: Survival curve + Weibull fit ───────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(hs, S_median, "o-", color=PALETTE[0], label="Empirical survival")
if not np.isnan(weibull_beta):
    h_range = np.linspace(hs.min(), hs.max(), 200)
    S_weib  = np.exp(-(h_range / weibull_eta) ** weibull_beta)
    ax.plot(h_range, S_weib, "--", color=PALETTE[3],
            label=f"Weibull fit (β={weibull_beta:.2f}, η={weibull_eta:.1f}h)")
ax.set_xlabel("Horizon h (hours)")
ax.set_ylabel("Survival S(h) = P(T > h)")
ax.set_ylim(0, 1.05)
ax.set_title("Discrete Survival Curve + Weibull Fit")
ax.legend()
save_fig("fig8_survival_weibull")

# ── Figure 9: SHAP global importance ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
top20_idx  = np.argsort(-mean_abs_shap)[:20]
top20_vals = mean_abs_shap[top20_idx]
top20_feat = [feature_cols[i] for i in top20_idx]
bars = ax.barh(top20_feat[::-1], top20_vals[::-1], color=PALETTE[0], edgecolor="white")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"SHAP Global Feature Importance – H={int(H_PLOT)}h")
ax.xaxis.set_major_locator(MaxNLocator(5))
save_fig("fig9_shap_global")

# ── Figure 10: SHAP beeswarm ──────────────────────────────────────────────────
# shap 0.49.x summary_plot (legacy API) does not accept an 'ax' argument.
# We let SHAP create its own figure internally (show=False), then grab it.
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_arr, X_shap, feature_names=feature_cols,
                  max_display=20, show=False, plot_type="dot")
plt.title(f"SHAP Beeswarm – H={int(H_PLOT)}h")
plt.tight_layout()
save_fig("fig10_shap_beeswarm")

# ── Figure 11: SHAP local waterfall ──────────────────────────────────────────
try:
    local_idx = -1
    exp = shap.Explanation(
        values       = shap_arr[local_idx],
        base_values  = explainer.expected_value,
        data         = X_shap.iloc[local_idx].values,
        feature_names= feature_cols
    )
    # shap.plots.waterfall owns its figure — do not pre-create one
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.tight_layout()
    save_fig("fig11_shap_local_waterfall")
except Exception as e:
    log.warning("Waterfall plot failed: %s", e)

# ── Figure 12: SHAP interaction heatmap ──────────────────────────────────────
if interact_df is not None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(interact_df.values, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(interact_df.columns)))
    ax.set_yticks(range(len(interact_df.index)))
    ax.set_xticklabels(interact_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(interact_df.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean |SHAP interaction|")
    ax.set_title(f"SHAP Interaction Values – Top 5 Features (H={int(H_PLOT)}h)")
    save_fig("fig12_shap_interaction_heatmap")

# ── Figure 13: ROC and PR curves ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax1, ax2 = axes
for i, H in enumerate(HORIZONS):
    y_H = failure_label(rul_test, H)
    if len(np.unique(y_H)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_H, p_calibrated[H])
    auc_val     = roc_auc_score(y_H, p_calibrated[H])
    ax1.plot(fpr, tpr, color=PALETTE[i], label=f"H={int(H)}h (AUC={auc_val:.3f})")
    prec, rec, _ = precision_recall_curve(y_H, p_calibrated[H])
    ap_val       = average_precision_score(y_H, p_calibrated[H])
    ax2.plot(rec, prec, color=PALETTE[i], label=f"H={int(H)}h (AP={ap_val:.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curves"); ax1.legend()
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR Curves"); ax2.legend()
plt.tight_layout()
save_fig("fig13_roc_pr_curves")

# ── Figure 14: Permutation importance vs SHAP ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
top10_names = [feature_cols[i] for i in top_idx_shap]
perm_top10  = [perm_importance_mean[i] for i in top_idx_shap]
perm_std_top10 = [perm_importance_std[i] for i in top_idx_shap]
shap_top10_vals= [mean_abs_shap[i] for i in top_idx_shap]

x = np.arange(len(top10_names))
w = 0.35
ax.bar(x - w/2, shap_top10_vals,  w, label="Mean |SHAP|",              color=PALETTE[0], edgecolor="white")
ax.bar(x + w/2, perm_top10,       w, label="Permutation Importance",   color=PALETTE[1], edgecolor="white",
       yerr=perm_std_top10, capsize=3, error_kw={"elinewidth":1})
ax.set_xticks(x)
ax.set_xticklabels(top10_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Importance Score")
ax.set_title(f"SHAP vs Permutation Importance – H={int(H_PLOT)}h (Top 10)")
ax.legend()
plt.tight_layout()
save_fig("fig14_shap_vs_perm_importance")

# ── Figure 15: Multi-horizon planning RUL ────────────────────────────────────
planning_50 = np.clip((1 - p_tmax_50) * TMAX, 0, TMAX)
planning_lo = np.clip((1 - p_tmax_90) * TMAX, 0, TMAX)
planning_hi = np.clip((1 - p_tmax_10) * TMAX, 0, TMAX)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_idx, true_rul,    color="steelblue", lw=1.5, label="True RUL (h)")
ax.plot(t_idx, planning_50, color=PALETTE[1],  lw=1.2, label=f"Planning RUL – E[min(T,Tmax)] (Tmax={int(TMAX)}h)")
ax.fill_between(t_idx, planning_lo, planning_hi, alpha=0.2, color=PALETTE[1], label="P10–P90 band")
ax.set_xlabel("Time Index")
ax.set_ylabel("Hours")
ax.set_title(f"Multi-Horizon Planning RUL vs True RUL – Unit {test_units[0]}")
ax.legend()
save_fig("fig15_planning_rul")

# ── Figure 16: Calibration bin heatmap (publication table alternative) ───────
fig, ax = plt.subplots(figsize=(9, 3))
cmap  = plt.cm.RdYlGn
norms = matplotlib.colors.Normalize(vmin=0, vmax=1)
for _, row in calib_df.iterrows():
    obs = row["observed_rate"] if not np.isnan(row["observed_rate"]) else 0.0
    pred= row["pred_center"]
    color = cmap(norms(obs))
    ax.bar(pred, obs, width=0.08, color=color, edgecolor="black", lw=0.5)
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, x_line, "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Failure Rate")
ax.set_title(f"Bin-Wise Calibration – H={int(H_PLOT)}h (with 95% CI)")
# Add CI error bars
valid = calib_df.dropna(subset=["ci_low", "ci_high"])
ax.errorbar(valid["pred_center"], valid["observed_rate"],
            yerr=[valid["observed_rate"] - valid["ci_low"],
                  valid["ci_high"] - valid["observed_rate"]],
            fmt="none", color="black", capsize=4, lw=1)
ax.legend()
save_fig("fig16_calibration_bins")

# ── Figure 17: LOUO CV summary (mean ± std bar chart) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics_to_plot = ["AUC", "AP", "BSS"]
for ax, metric in zip(axes, metrics_to_plot):
    means = cv_df[f"{metric}_mean"].values
    stds  = cv_df[f"{metric}_std"].values
    labels= [str(int(h)) for h in cv_df.index]
    ax.bar(labels, means, yerr=stds, capsize=5, color=PALETTE[:len(labels)],
           edgecolor="black", error_kw={"elinewidth":1.2})
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel(metric)
    ax.set_title(f"LOUO CV – {metric}")
plt.tight_layout()
save_fig("fig17_louo_cv_bar")

# ── Figure 18: Fleet-level risk surface (illustrative multi-unit risk map) ───
# Uses LOUO predictions already computed — aggregate p_mean across all units
log.info("Generating fleet-level risk surface ...")
fleet_risk = {}   # unit -> p_mean array

for test_unit in unit_ids:
    tr_units  = [u for u in unit_ids if u != test_unit]
    tr_mask   = df[UNIT_COL].isin(tr_units)
    te_mask   = df[UNIT_COL] == test_unit
    X_tr_fl   = X_all[tr_mask].reset_index(drop=True)
    X_te_fl   = X_all[te_mask].reset_index(drop=True)
    rul_tr_fl = rul_all[tr_mask].reset_index(drop=True)
    y_tr_fl   = failure_label(rul_tr_fl, H_PLOT)
    Xtr_rf = scaler_reg.transform(X_tr_fl[raw_feature_cols])
    Xte_rf = scaler_reg.transform(X_te_fl[raw_feature_cols])
    km_fl  = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_SEED, n_init=10)
    km_fl.fit(Xtr_rf[:min(REGIME_SAMPLE, len(X_tr_fl))])
    reg_te_fl = km_fl.predict(Xte_rf)
    ens_fl = train_calibrated_ensemble(
        X_tr_fl, y_tr_fl, feature_cols,
        regimes_train=km_fl.predict(Xtr_rf),
        seed0=RANDOM_SEED + int(H_PLOT)
    )
    members_fl = ens_fl.predict_proba_calibrated(X_te_fl, regimes=reg_te_fl)
    fleet_risk[test_unit] = members_fl.mean(axis=0)

# Heatmap: unit x time (normalise each unit's timeline to [0,1] for alignment)
N_TIME_BINS = 100
heat_matrix = np.full((len(unit_ids), N_TIME_BINS), np.nan)
for j, uid in enumerate(unit_ids):
    p_unit = fleet_risk[uid]
    x      = np.linspace(0, 1, len(p_unit))
    x_new  = np.linspace(0, 1, N_TIME_BINS)
    heat_matrix[j] = np.interp(x_new, x, p_unit)

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(heat_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
               extent=[0, N_TIME_BINS, -0.5, len(unit_ids) - 0.5])
ax.set_yticks(range(len(unit_ids)))
ax.set_yticklabels([f"Unit {u}" for u in unit_ids], fontsize=9)
ax.set_xlabel("Normalised Lifecycle [%]")
ax.set_title(f"Fleet-Level Risk Surface (H={int(H_PLOT)}h) – All Units")
plt.colorbar(im, ax=ax, label="P(Failure ≤ 72h)")
save_fig("fig18_fleet_risk_surface")

# =============================================================================
# ─── CONSOLE SUMMARY ─────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*70)
print("METRICS SUMMARY (held-out test unit)")
print("="*70)
for H in HORIZONS:
    m = metrics_dict[H]
    print(f"H={int(H):>4}h | AUC={m['AUC']:.4f} | AP={m['AP']:.4f} | "
          f"Brier={m['Brier']:.4f} | BSS={m['BSS']:.4f}")

print("\n" + "-"*70)
print(f"H={int(H_PLOT)}h Extended Stats:")
print(f"  ECE                   = {ece_H_plot:.4f}")
print(f"  Brier Reliability     = {brier_decomp['reliability']:.4f}")
print(f"  Brier Resolution      = {brier_decomp['resolution']:.4f}")
print(f"  Brier Uncertainty     = {brier_decomp['uncertainty']:.4f}")
print(f"  Mean uncertainty width= {unc_width.mean():.4f}")
print(f"  F2-optimal threshold  = {f_thresh:.3f}")
print(f"  Cost-opt. threshold   = {cost_thresh:.3f}")
if not np.isnan(weibull_beta):
    print(f"  Weibull β (shape)     = {weibull_beta:.3f}")
    print(f"  Weibull η (scale)     = {weibull_eta:.1f} h")
print(f"\n  Alert windows (default thresh): {alert_stats.get('n_windows',0)}")
for k, v in alert_stats.items():
    print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")

print("\n" + "-"*70)
print("SHAP Top-10 Features:")
print(shap_top10[["rank","feature","mean_abs_shap","contribution_pct","direction"]].to_string(index=False))

print("\n" + "-"*70)
print("LOUO Cross-Validation Summary:")
print(cv_df[["AUC_mean","AUC_std","AP_mean","AP_std","BSS_mean","BSS_std","ECE_mean"]].to_string())

print("\n" + "="*70)
print(f"All figures saved to: {FIG_DIR}/")
print(f"All tables saved to : {RESULTS_DIR}/")
print("="*70)

# =============================================================================
# ─── SAVE MODELS ─────────────────────────────────────────────────────────────
# =============================================================================
joblib.dump(ensembles, os.path.join(MODEL_DIR, "ensembles_all_horizons.pkl"))
joblib.dump(kmeans,    os.path.join(MODEL_DIR, "kmeans_regime_detector.pkl"))
joblib.dump(scaler_reg, os.path.join(MODEL_DIR, "scaler_regime.pkl"))
log.info("Models saved to: %s/", MODEL_DIR)
log.info("Pipeline complete.")
