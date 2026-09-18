"""
STEP 8 - PS-4: Response Time Prediction
=======================================

The odd one out. PS-1, PS-2 and PS-3 are all NLP problems on raw text. This one
is a classic TABULAR REGRESSION problem: predict a NUMBER - how many minutes the
company will take to reply.

WHY THIS ONE IS DIFFERENT
    - The target is real, not invented. response_time_minutes was computed from
      actual timestamps in script 01, so unlike PS-1 and PS-2 there is no
      weak-supervision problem here and no circularity to apologise for. These
      numbers mean exactly what they say.
    - The features are mostly NOT words. Message length, hour of day, day of
      week, the brand, and - as the brief requires - the PREDICTED intent and
      priority from PS-1 and PS-2. This is where the four problem statements
      finally join up into one system.
    - Metrics are errors, not accuracies. MAE and RMSE in minutes, plus R².

THE ONE MODELLING DECISION THAT MATTERS
    Response time is violently right-skewed: median ~18 minutes, mean ~103,
    maximum 1440. Fit a regressor to that directly and it spends all its effort
    on the slow tail and predicts badly everywhere else.
    So we train on log1p(minutes) and invert with expm1 before scoring. All
    reported errors are in real minutes - a model that is only good in log space
    is no use to a support manager.

REQUIRES
    outputs/ps1_intent_model.joblib   (run 03 first)
    outputs/ps2_priority_model.joblib (run 07 first)
    pip install xgboost               (optional - falls back to sklearn's
                                       HistGradientBoostingRegressor)

Run:  python 08_ps4_response_time.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\data"
OUT_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\outputs"
USE_SAMPLE = True
SUFFIX = "_sample" if USE_SAMPLE else ""

MAX_ROWS = 150_000          # regression on 1M rows is slow and adds little
TOP_BRANDS = 25             # rarer brands are pooled into "other"
RANDOM_STATE = 42
TUNE = True                 # RandomizedSearchCV on the Random Forest

os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"

URGENCY = (r"\b(urgent|asap|immediately|emergency|still waiting|been waiting|"
           r"no response|hours|days|unacceptable|ridiculous|cancel|refund now|"
           r"fraud|hacked|stolen|complaint|angry|furious)")


# ---------------------------------------------------------------------------
# 1. LOAD PAIRS
# ---------------------------------------------------------------------------
def load_pairs():
    path = os.path.join(DATA_DIR, f"support_pairs{SUFFIX}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}\nRun 01_build_pairs.py first.")

    df = pd.read_csv(path, parse_dates=["customer_time"])
    df = df.dropna(subset=["customer_text", "response_time_minutes"])
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)

    print(f"Loaded {len(df):,} pairs")
    d = df["response_time_minutes"].describe(percentiles=[.25, .5, .75, .9, .99])
    print("\nresponse_time_minutes:")
    print(f"  median {d['50%']:7.1f}   mean {d['mean']:7.1f}   "
          f"90th {d['90%']:7.1f}   max {d['max']:7.1f}")
    print(f"  skew   {df['response_time_minutes'].skew():.2f}  "
          f"-> log-transform the target (see docstring)")
    return df


# ---------------------------------------------------------------------------
# 2. FEATURES
#
# This is where PS-1 and PS-2 pay off: their models are loaded and used to
# PREDICT intent and priority for every ticket, and those predictions become
# columns here. The brief asks for exactly this - it is what turns four separate
# exercises into one system.
# ---------------------------------------------------------------------------
def add_model_predictions(df):
    ps1_path = os.path.join(OUT_DIR, "ps1_intent_model.joblib")
    ps2_path = os.path.join(OUT_DIR, "ps2_priority_model.joblib")
    for p, script in ((ps1_path, "03_ps1_intent_model.py"),
                      (ps2_path, "07_ps2_priority_model.py")):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}\nRun {script} first.")

    raw = df["customer_text"].fillna("")

    print("\npredicting intent with the PS-1 model ...")
    ps1 = joblib.load(ps1_path)
    df["pred_intent"] = ps1["model"].predict(ps1["vectorizer"].transform(raw))

    print("predicting priority with the PS-2 model ...")
    ps2 = joblib.load(ps2_path)
    letters = raw.str.count(r"[A-Za-z]").clip(lower=1)
    num = pd.DataFrame({
        "caps_ratio": raw.str.count(r"[A-Z]") / letters,
        "exclaims": raw.str.count(r"!"),
        "questions": raw.str.count(r"\?"),
        "char_len": raw.str.len(),
    }, index=df.index)
    if "sentiment" in ps2["feature_cols"]:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        num["sentiment"] = raw.map(lambda t: sia.polarity_scores(t)["compound"])
    X = hstack([ps2["vectorizer"].transform(raw),
                csr_matrix(ps2["scaler"].transform(num[ps2["feature_cols"]]))]).tocsr()
    df["pred_priority"] = ps2["model"].predict(X)

    print(f"  intent  : {df['pred_intent'].value_counts().to_dict()}")
    print(f"  priority: {df['pred_priority'].value_counts().to_dict()}")
    return df


def build_features(df):
    raw = df["customer_text"].fillna("")
    letters = raw.str.count(r"[A-Za-z]").clip(lower=1)

    feats = pd.DataFrame(index=df.index)

    # --- message shape ---
    feats["char_len"] = raw.str.len()
    feats["word_count"] = raw.str.split().str.len()
    feats["caps_ratio"] = raw.str.count(r"[A-Z]") / letters
    feats["exclaims"] = raw.str.count(r"!")
    feats["questions"] = raw.str.count(r"\?")
    feats["urgency_words"] = raw.str.lower().str.count(URGENCY)
    feats["has_url"] = raw.str.contains(r"http", regex=True).astype(int)
    feats["mentions"] = raw.str.count(r"@\w+")

    # --- WHEN it was sent ---
    # Hour and weekday are cyclical: hour 23 is adjacent to hour 0, but a tree
    # sees 23 and 0 as maximally far apart. Encoding each as a sine/cosine pair
    # restores that adjacency. Raw hour is kept too - trees can use either.
    t = df["customer_time"]
    feats["hour"] = t.dt.hour
    feats["dayofweek"] = t.dt.dayofweek
    feats["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    feats["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)

    # --- who they are talking to, and what about ---
    # Brand matters enormously: some support desks reply in minutes, others in
    # hours. Rare brands are pooled so we do not create 80 columns of noise.
    top = df["brand"].value_counts().head(TOP_BRANDS).index
    brand = df["brand"].where(df["brand"].isin(top), "other")
    feats = pd.concat([
        feats,
        pd.get_dummies(brand, prefix="brand", dtype=int),
        pd.get_dummies(df["pred_intent"], prefix="intent", dtype=int),
        pd.get_dummies(df["pred_priority"], prefix="priority", dtype=int),
    ], axis=1)

    print(f"\nbuilt {feats.shape[1]} features for {len(feats):,} rows")
    return feats


# ---------------------------------------------------------------------------
# 3. EVALUATE - always back in real minutes
# ---------------------------------------------------------------------------
def score(y_true_min, y_pred_log):
    """y_pred_log is in log1p space; invert before measuring anything."""
    pred_min = np.expm1(y_pred_log).clip(0, 24 * 60)
    return {
        "MAE_min": mean_absolute_error(y_true_min, pred_min),
        "RMSE_min": float(np.sqrt(mean_squared_error(y_true_min, pred_min))),
        "R2": r2_score(y_true_min, pred_min),
    }, pred_min


# ---------------------------------------------------------------------------
# 4. MODELS
# ---------------------------------------------------------------------------
def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1,
            random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.06, max_depth=7,
            subsample=0.85, colsample_bytree=0.85, n_jobs=-1,
            random_state=RANDOM_STATE, tree_method="hist")
        print("using XGBoost")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        models["HistGradientBoosting"] = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, random_state=RANDOM_STATE)
        print("xgboost not installed - using HistGradientBoosting instead "
              "(pip install xgboost to match the brief exactly)")
    return models


def tune_random_forest(Xtr, ytr):
    """The brief asks for hyperparameter tuning with cross-validation."""
    print("\ntuning RandomForest with 3-fold CV ...", flush=True)
    search = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 12, 20, 30],
            "min_samples_leaf": [1, 3, 5, 10],
            "max_features": ["sqrt", 0.3, 0.5],
        },
        n_iter=8, cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(Xtr, ytr)
    print(f"  best params: {search.best_params_}")
    print(f"  best CV MAE (log space): {-search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# 5. CHARTS
# ---------------------------------------------------------------------------
def plot_importance(model, names, path, top_n=18):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        title = "Feature importance (impurity reduction)"
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
        title = "Feature importance (|coefficient|)"
    else:
        return
    order = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(order) + 1.4))
    ax.barh(np.array(names)[order], imp[order], color=BLUE, height=0.65)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=12)
    ax.set_xlabel("Relative importance", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.grid(axis="x", color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pred_vs_actual(y_true, y_pred, path):
    """Log-log, because a linear plot is an unreadable blob near the origin."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=4, alpha=0.12, color=BLUE, edgecolors="none")
    lim = [1, 1440]
    ax.plot(lim, lim, color="#b3402f", lw=1.2, ls="--", label="perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual minutes", fontsize=9, color=MUTED)
    ax.set_ylabel("Predicted minutes", fontsize=9, color=MUTED)
    ax.set_title("Predicted vs actual response time", fontsize=11, color=INK,
                 loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(labelsize=9, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    df = load_pairs()
    df = add_model_predictions(df)
    X = build_features(df)
    y_min = df["response_time_minutes"].values
    y_log = np.log1p(y_min)

    Xtr, Xte, ytr_log, yte_log, ytr_min, yte_min = train_test_split(
        X, y_log, y_min, test_size=0.2, random_state=RANDOM_STATE)
    print(f"train {len(Xtr):,}   test {len(Xte):,}")

    # A baseline worth beating: always predict the median. If a model cannot
    # beat this, it has learned nothing.
    base_pred = np.full(len(yte_min), np.median(ytr_min))
    print(f"\nbaseline (always predict the median, {np.median(ytr_min):.0f} min):"
          f"  MAE {mean_absolute_error(yte_min, base_pred):.1f} min")

    models = get_models()
    if TUNE:
        models["RandomForest (tuned)"] = tune_random_forest(Xtr, ytr_log)

    rows, fitted = [], {}
    for name, model in models.items():
        print(f"\ntraining {name} ...", flush=True)
        model.fit(Xtr, ytr_log)
        metrics, _ = score(yte_min, model.predict(Xte))
        rows.append({"model": name, **metrics})
        fitted[name] = model
        print(f"  MAE {metrics['MAE_min']:.1f} min   "
              f"RMSE {metrics['RMSE_min']:.1f} min   R2 {metrics['R2']:.3f}")

    results = pd.DataFrame(rows).sort_values("MAE_min")
    print("\n--- MODEL COMPARISON (errors in minutes) ---")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    metrics, pred_min = score(yte_min, best.predict(Xte))

    print(f"\n=== BEST MODEL: {best_name} ===")
    print(f"  MAE  {metrics['MAE_min']:7.1f} minutes   "
          f"(average miss, in either direction)")
    print(f"  RMSE {metrics['RMSE_min']:7.1f} minutes   "
          f"(punishes large misses harder)")
    print(f"  R2   {metrics['R2']:7.3f}          "
          f"(share of variation explained; 0 = no better than the mean)")

    if metrics["R2"] < 0.25:
        print("\n  A low R2 here is normal and worth saying out loud: how fast a")
        print("  company replies depends mostly on THEIR staffing and queue at")
        print("  that moment, which is not in this dataset. The ticket text can")
        print("  only explain so much. That is a finding, not a failure.")

    plot_importance(best, list(X.columns),
                    os.path.join(OUT_DIR, "ps4_feature_importance.png"))
    plot_pred_vs_actual(yte_min, pred_min,
                        os.path.join(OUT_DIR, "ps4_pred_vs_actual.png"))

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns)
        print("\ntop 12 features:")
        print(imp.sort_values(ascending=False).head(12).to_string())

    joblib.dump({"model": best, "name": best_name, "columns": list(X.columns)},
                os.path.join(OUT_DIR, "ps4_response_time_model.joblib"))
    results.to_csv(os.path.join(OUT_DIR, "ps4_model_comparison.csv"), index=False)
    print(f"\nsaved model + charts + tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
