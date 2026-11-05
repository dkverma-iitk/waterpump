#!/usr/bin/env python3


import os, argparse, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ----- config matching the pump study -----
RANDOM_SEED      = 42
CALIB_RANDOM_SEED= 123
CALIB_FRAC       = 0.25
N_ESTIMATORS     = 100   # was 250; 100 gives identical AUC here, ~2.5x faster
WINDOWS          = (5, 20, 50)
# CMAPSS horizons in CYCLES (analogue of the pump's 24/72/168 h).
# H_plot is the primary diagnostic horizon (analogue of 72 h).
HORIZONS         = [15.0, 30.0, 50.0]
H_PLOT           = 30.0
N_REGIMES        = 3
ALERT_THRESHOLD  = 0.50
MIN_ALERT_CYCLES = 5          # persistence filter (cycles)

RAW_COLS = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
FD001_FLAT = ["s1","s5","s6","s10","s16","s18","s19"]


# ============================================================
# 1. Load + reshape CMAPSS into long format
# ============================================================
def load_fd001(path, rul_cap=125.0, drop_flat=True):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = RAW_COLS[: df.shape[1]]
    maxc = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (maxc - df["cycle"]).astype(float)
    if rul_cap and rul_cap > 0:
        df["rul"] = np.minimum(df["rul"], rul_cap)
    sensors = [c for c in df.columns if c.startswith("s") and c not in ("op1","op2","op3")]
    if drop_flat:
        sensors = [c for c in sensors if c not in FD001_FLAT]
    rename = {c: f"sensor_{i:02d}" for i, c in enumerate(sensors)}
    out = df[["unit"] + sensors + ["rul"]].rename(columns=rename).rename(columns={"unit":"unit_id"})
    uid = {u: i for i, u in enumerate(sorted(out.unit_id.unique()))}
    out["unit_id"] = out.unit_id.map(uid)
    return out.sort_values(["unit_id","rul"], ascending=[True, False]).reset_index(drop=True)


# ============================================================
# 2. Feature engineering (rolling mean / std / trend) — vectorized
# ============================================================
def _rolling_slope_vec(x, w):
    x = np.asarray(x, float); n = len(x); out = np.zeros(n)
    cs  = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n, dtype=float)
    cidx= np.concatenate([[0.0], np.cumsum(idx)])
    cxy = np.concatenate([[0.0], np.cumsum(idx * x)])
    cii = np.concatenate([[0.0], np.cumsum(idx * idx)])
    for i in range(n):
        lo = i - w + 1
        if lo < 0: lo = 0
        k = i - lo + 1
        if k < 2: continue
        sx  = cs[i+1]-cs[lo]; st = cidx[i+1]-cidx[lo]
        sxy = cxy[i+1]-cxy[lo]; stt = cii[i+1]-cii[lo]
        denom = stt - st*st/k
        if denom > 0: out[i] = (sxy - st*sx/k)/denom
    return out

def engineer(df, windows=WINDOWS):
    sensors = [c for c in df.columns if c.startswith("sensor_")]
    feats = {}
    for c in sensors:
        feats[c] = df[c].values
    g = df.groupby("unit_id", group_keys=False)
    for c in sensors:
        for w in windows:
            feats[f"{c}_rmean{w}"] = g[c].transform(lambda s: s.rolling(w, min_periods=1).mean()).values
            feats[f"{c}_rstd{w}"]  = g[c].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0)).values
    # trend per unit (vectorized slope)
    for c in sensors:
        for w in windows:
            col = np.zeros(len(df))
            for _, sub in df.groupby("unit_id"):
                col[sub.index.values] = _rolling_slope_vec(sub[c].values, w)
            feats[f"{c}_trend{w}"] = col
    F = pd.DataFrame(feats, index=df.index)
    F["unit_id"] = df.unit_id.values
    F["rul"] = df.rul.values
    return F


# ============================================================
# 3. Model: GB ensemble + isotonic calibration
# ============================================================
def make_model(seed):
    return Pipeline([("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.05,
               max_depth=3, subsample=0.8, random_state=seed))])

def label(rul, H): return (np.asarray(rul) <= H).astype(int)

def train_ensemble(Xtr, ytr, seed0, n_ensemble):
    models, cals = [], []
    rng = np.random.default_rng(CALIB_RANDOM_SEED)
    for k in range(n_ensemble):
        idx = np.arange(len(Xtr)); rng.shuffle(idx)
        ncal = int(CALIB_FRAC * len(Xtr)); ci, fi = idx[:ncal], idx[ncal:]
        m = make_model(seed0 + k); m.fit(Xtr.iloc[fi], ytr[fi])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(Xtr.iloc[ci])[:, 1], ytr[ci])
        models.append(m); cals.append(iso)
    return models, cals

def predict_members(models, cals, X):
    return np.asarray([np.clip(c.predict(m.predict_proba(X)[:, 1]), 0, 1)
                       for m, c in zip(models, cals)])

def ece(y, p, nb=10):
    bins = np.linspace(0, 1, nb+1); e = 0.0; N = len(y)
    for i in range(nb):
        m = (p >= bins[i]) & (p < bins[i+1]) if i < nb-1 else (p >= bins[i]) & (p <= bins[i+1])
        if m.sum(): e += m.sum()/N * abs(y[m].mean() - p[m].mean())
    return e


# ============================================================
# 4. Protocol steps
# ============================================================
def step1_louo(F, feat, units, n_ensemble, out_dir):
    """Per-fold LOUO AUC. Decision: per-fold AUC clearly above 0.5."""
    print("\n[STEP 1] Leave-one-unit-out cross-validation")
    rows = []
    for H in HORIZONS:
        fold = []
        for tu in units:
            tr = F[F.unit_id != tu]; te = F[F.unit_id == tu]
            ytr, yte = label(tr.rul, H), label(te.rul, H)
            if len(np.unique(yte)) < 2:
                continue
            m, c = train_ensemble(tr[feat], ytr, RANDOM_SEED + int(H), n_ensemble)
            p = predict_members(m, c, te[feat]).mean(0)
            fold.append(dict(H=int(H), unit=int(tu),
                             auc=roc_auc_score(yte, p),
                             ap=average_precision_score(yte, p),
                             ece=ece(yte, p)))
        a = np.array([f["auc"] for f in fold])
        print(f"  H={int(H):>3}c  AUC mean={a.mean():.3f} std={a.std():.3f} "
              f"min={a.min():.3f} max={a.max():.3f}  -> "
              f"{'PASS' if a.min() > 0.5 else 'CHECK (a fold dips to/below 0.5)'}")
        rows.extend(fold)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "step1_louo_folds.csv"), index=False)
    return pd.DataFrame(rows)

def step2_inversion(louo_df):
    """Inversion-direction consistency. With monotone degradation we expect
    AUC>0.5 on every fold (no sign flips). Decision: consistent direction."""
    print("\n[STEP 2] Inversion-direction consistency")
    for H in HORIZONS:
        a = louo_df[louo_df.H == int(H)].auc.values
        below = (a < 0.5).sum()
        print(f"  H={int(H):>3}c  folds below 0.5: {below}/{len(a)} -> "
              f"{'consistent (no inversion)' if below == 0 else 'INCONSISTENT'}")

def step3_shap(F, feat, units, n_ensemble, out_dir):
    """SHAP top features on a single held-out unit at H_PLOT.
    Decision: top features should reflect degradation, not pure operating-mode."""
    print(f"\n[STEP 3] SHAP top features (H={int(H_PLOT)}c, held-out unit)")
    import shap
    test_unit = units[-1]
    tr = F[F.unit_id != test_unit]; te = F[F.unit_id == test_unit]
    ytr = label(tr.rul, H_PLOT)
    m, c = train_ensemble(tr[feat], ytr, RANDOM_SEED + int(H_PLOT), n_ensemble)
    base = m[0].named_steps["gb"]
    Xs = m[0].named_steps["scaler"].transform(te[feat])
    sv = shap.TreeExplainer(base).shap_values(Xs)
    sv = np.array(sv)
    if sv.ndim == 3: sv = sv[..., 1] if sv.shape[-1] == 2 else sv.reshape(sv.shape[0], -1)
    mabs = np.abs(sv).mean(0); mdir = sv.mean(0)
    order = np.argsort(-mabs)[:10]
    rows = []
    for r, i in enumerate(order, 1):
        d = "increase" if mdir[i] > 0 else "decrease"
        rows.append(dict(rank=r, feature=feat[i], mean_abs_shap=float(mabs[i]),
                         pct=float(mabs[i] / mabs[order].sum() * 100), direction=d))
        print(f"  {r:2d}. {feat[i]:20s} {mabs[i]:.4f}  ({d})")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "step3_shap_top10.csv"), index=False)
    return pd.DataFrame(rows)

def step4_alerts(F, feat, units, n_ensemble, out_dir):
    """Alert-window location vs true end-of-life on held-out unit.
    Decision: alerts should concentrate near true failure (high RUL fraction
    elapsed), not in early/mid life."""
    print(f"\n[STEP 4] Alert-window location (H={int(H_PLOT)}c, held-out unit)")
    test_unit = units[-1]
    tr = F[F.unit_id != test_unit]; te = F[F.unit_id == test_unit].reset_index(drop=True)
    ytr = label(tr.rul, H_PLOT)
    m, c = train_ensemble(tr[feat], ytr, RANDOM_SEED + int(H_PLOT), n_ensemble)
    p = predict_members(m, c, te[feat]).mean(0)
    alert = p >= ALERT_THRESHOLD
    # find persistent windows
    wins, s = [], None
    for i, fl in enumerate(alert):
        if fl and s is None: s = i
        elif not fl and s is not None:
            if i - s >= MIN_ALERT_CYCLES: wins.append((s, i-1))
            s = None
    if s is not None and len(alert) - s >= MIN_ALERT_CYCLES: wins.append((s, len(alert)-1))
    n = len(te)
    # life fraction elapsed at each alert-window centre (1.0 = failure)
    centres = [ (w[0]+w[1])/2 / n for w in wins ]
    print(f"  alert windows: {len(wins)}")
    if centres:
        late = sum(c_ > 0.6 for c_ in centres)
        print(f"  window centres (fraction of life elapsed): "
              f"{', '.join(f'{c_:.2f}' for c_ in centres)}")
        print(f"  windows in last 40% of life: {late}/{len(centres)} -> "
              f"{'PASS (alerts near end-of-life)' if late == len(centres) else 'CHECK'}")
    else:
        print("  no persistent alert windows at this threshold")
    pd.DataFrame([dict(start=w[0], end=w[1], centre_frac=cf)
                  for w, cf in zip(wins, centres)]).to_csv(
        os.path.join(out_dir, "step4_alerts.csv"), index=False)


# ============================================================
# main
# ============================================================
def main():
    global N_ESTIMATORS
    ap = argparse.ArgumentParser(
        description="CMAPSS FD001 validation-protocol contrast (single consolidated script).")
    ap.add_argument("--train_file", default="train_FD001.txt")
    ap.add_argument("--n_units", type=int, default=20,
                    help="Units used for LOUO (0 = all 100). Subset is fine for the contrast.")
    ap.add_argument("--n_ensemble", type=int, default=5,
                    help="Ensemble members (5 is plenty; 10 for the final reported table).")
    ap.add_argument("--n_estimators", type=int, default=N_ESTIMATORS,
                    help="GB trees per model. 100 default; raise to 250 only if you must.")
    ap.add_argument("--rul_cap", type=float, default=125.0)
    ap.add_argument("--out_dir", default="results_cmapss_fd001")
    ap.add_argument("--steps", default="1,2,3,4",
                    help="Comma list of steps to run, e.g. '3,4' or '1,2'. Default all.")
    ap.add_argument("--skip_shap", action="store_true",
                    help="Force-skip Step 3 even if requested (no shap installed).")
    ap.add_argument("--cache_features", action="store_true",
                    help="Cache engineered features to <out_dir>/_features.parquet and reuse.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    N_ESTIMATORS = args.n_estimators  # local override used via make_model closure

    steps = {int(x) for x in str(args.steps).split(",") if x.strip()}
    t0 = time.time()

    # ---- load + engineer (with optional cache) ----
    cache = os.path.join(args.out_dir, "_features.parquet")
    if args.cache_features and os.path.exists(cache):
        print(f"Loading cached features from {cache} ...")
        F = pd.read_parquet(cache)
    else:
        print("Loading + reshaping FD001 ...")
        df = load_fd001(args.train_file, rul_cap=args.rul_cap)
        print(f"  {len(df):,} rows, {df.unit_id.nunique()} units, "
              f"{len([c for c in df.columns if c.startswith('sensor_')])} sensors")
        print("Engineering features ...")
        F = engineer(df)
        if args.cache_features:
            try:
                F.to_parquet(cache); print(f"  cached -> {cache}")
            except Exception as e:
                print(f"  (cache write skipped: {e})")
    feat = [c for c in F.columns if c not in ("unit_id", "rul")]
    print(f"  {len(feat)} features  ({time.time()-t0:.0f}s)")

    units = sorted(F.unit_id.unique())
    if args.n_units and args.n_units > 0:
        units = units[: args.n_units]
    print(f"  units={len(units)}  ensemble={args.n_ensemble}  trees={N_ESTIMATORS}  "
          f"steps={sorted(steps)}")

    louo = None
    if 1 in steps:
        louo = step1_louo(F, feat, units, args.n_ensemble, args.out_dir)
    if 2 in steps:
        if louo is None:
            p = os.path.join(args.out_dir, "step1_louo_folds.csv")
            if os.path.exists(p):
                louo = pd.read_csv(p)
            else:
                print("  [STEP 2] needs Step 1 output; run step 1 first or together.")
        if louo is not None:
            step2_inversion(louo)
    if 3 in steps and not args.skip_shap:
        try:
            step3_shap(F, feat, units, args.n_ensemble, args.out_dir)
        except Exception as e:
            print(f"  [STEP 3] skipped ({e})")
    if 4 in steps:
        step4_alerts(F, feat, units, args.n_ensemble, args.out_dir)

    # summary table (only if step 1 ran or cached folds exist)
    if louo is not None:
        summ = (louo.groupby("H")
                    .agg(auc_mean=("auc","mean"), auc_std=("auc","std"),
                         auc_min=("auc","min"), auc_max=("auc","max"))
                    .reset_index())
        summ.to_csv(os.path.join(args.out_dir, "louo_summary.csv"), index=False)
        print("\n=== LOUO summary (for paper table) ===")
        print(summ.to_string(index=False))
    print(f"\nDone in {time.time()-t0:.0f}s. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
