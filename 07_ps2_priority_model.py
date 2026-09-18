"""
STEP 7 - PS-2: Ticket Priority / Urgency Detection
==================================================

Same shape as PS-1, different question. PS-1 asked WHAT a ticket is about;
this asks HOW URGENT it sounds. Low, Medium or High.

WHAT IS DIFFERENT FROM SCRIPT 03 - and why it matters

1. TEXT ALONE IS NOT ENOUGH.
   Intent lives in the words: "refund" means Billing whoever wrote it. Urgency
   lives partly OUTSIDE the words - in SHOUTING, in "!!!", in a sentiment that
   TF-IDF cannot see because it throws away letter case. So this script bolts
   four numeric features onto the TF-IDF matrix, plus a sentiment score if
   VADER is available. That combination is what the brief asks for.

2. THE CLASS BALANCE IS MUCH WORSE.
   Roughly Medium 82% / Low 12% / High 6%. A model that predicts "Medium" for
   every single ticket scores 82% accuracy and is completely useless. Accuracy
   is a trap here - macro-F1 is the honest number.

3. ONE METRIC MATTERS MORE THAN THE REST.
   The brief is explicit: "pay extra attention to Recall on the High-priority
   class, since missing an urgent ticket is costlier than a false alarm."
   Recall on High is printed on its own line at the end. That is the number to
   put in your report, and the one you will be asked about.

4. THE CONFUSION MATRIX IS ORDERED Low -> Medium -> High.
   Not alphabetical. Priority is an ORDINAL scale, so the off-diagonal cells
   are only readable if the classes are in their natural order - it lets you
   see whether mistakes are near-misses (High called Medium) or serious ones
   (High called Low).

Run:  python 07_ps2_priority_model.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\data"
OUT_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\outputs"
USE_SAMPLE = True
SUFFIX = "_sample" if USE_SAMPLE else ""

MAX_ROWS = 200_000
RANDOM_STATE = 42

# Which model to keep when they disagree. "macro_f1" is the balanced choice;
# "high_recall" says catching urgent tickets matters more than overall tidiness.
# The brief leans towards the second. Whichever you pick, justify it in the
# report - the script prints the trade-off so you can see what it costs.
SELECT_BY = "macro_f1"

# Ordinal order. Everything - reports, matrices, charts - uses this, not sorted().
PRIORITY_ORDER = ["Low", "Medium", "High"]

os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", BLUE, "#123a68"])

NUMERIC_FEATURES = ["caps_ratio", "exclaims", "questions", "char_len"]


# ---------------------------------------------------------------------------
# 1. SENTIMENT (optional extra feature)
#
# VADER is a rule-based sentiment scorer built for social media - it understands
# emoji, slang, ALL CAPS and "!!!", which is exactly this problem. If nltk or the
# lexicon is not available the script carries on without it rather than dying;
# it just means one fewer feature.
# ---------------------------------------------------------------------------
def add_sentiment(df: pd.DataFrame) -> list:
    try:
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
    except Exception as exc:
        print(f"  VADER unavailable ({type(exc).__name__}) - continuing without "
              f"sentiment. To enable: pip install nltk")
        return []

    print("  scoring sentiment with VADER ...")
    # compound is a single -1 (very negative) to +1 (very positive) summary.
    df["sentiment"] = df["customer_text"].fillna("").map(
        lambda t: sia.polarity_scores(t)["compound"])
    return ["sentiment"]


# ---------------------------------------------------------------------------
# 2. LOAD
# ---------------------------------------------------------------------------
def load_data():
    path = os.path.join(DATA_DIR, f"train{SUFFIX}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"Not found: {path}\nRun 02_clean_and_label.py with the same "
            f"USE_SAMPLE setting first.")

    df = pd.read_csv(path).dropna(subset=["clean_text", "priority"])
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)

    print(f"Loaded {len(df):,} rows from {path}\n")
    counts = df["priority"].value_counts().reindex(PRIORITY_ORDER)
    share = counts / counts.sum()
    for cls in PRIORITY_ORDER:
        print(f"  {cls:<7} {counts[cls]:>8,}  ({share[cls]:.1%})")
    print(f"\n  Majority-class baseline accuracy: {share.max():.1%}")
    print("  Any model must beat that to be worth anything.\n")
    return df


# ---------------------------------------------------------------------------
# 3. FEATURES = TF-IDF  +  numeric urgency signals
#
# hstack glues the sparse text matrix to the dense numeric columns. The numeric
# ones are standardised first: without scaling, char_len (values in the hundreds)
# would dwarf caps_ratio (values between 0 and 1) in any distance- or
# gradient-based model.
# ---------------------------------------------------------------------------
def build_features(X_train_txt, X_test_txt, train_num, test_num):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6,
                          max_features=50_000, sublinear_tf=True,
                          strip_accents="unicode")
    Xtr_txt = vec.fit_transform(X_train_txt)
    Xte_txt = vec.transform(X_test_txt)

    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(train_num)
    Xte_num = scaler.transform(test_num)

    Xtr = hstack([Xtr_txt, csr_matrix(Xtr_num)]).tocsr()
    Xte = hstack([Xte_txt, csr_matrix(Xte_num)]).tocsr()

    print(f"vocabulary : {len(vec.vocabulary_):,} terms")
    print(f"features   : {Xtr_txt.shape[1]:,} text + {Xtr_num.shape[1]} numeric "
          f"= {Xtr.shape[1]:,}")
    return vec, scaler, Xtr, Xte


# ---------------------------------------------------------------------------
# 4. TRAIN AND COMPARE
#
# No MultinomialNB this time - it requires non-negative features, and the
# standardised numeric columns contain negatives. Worth knowing why a model
# gets dropped rather than silently omitting it.
# ---------------------------------------------------------------------------
def train_and_compare(Xtr, ytr, Xte, yte):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        print(f"\ntraining {name} ...", flush=True)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "high_recall": recall_score(yte, pred, labels=["High"],
                                        average="macro", zero_division=0),
        })
        fitted[name] = model
        print(f"  macro-F1 {rows[-1]['macro_f1']:.3f}   "
              f"HIGH recall {rows[-1]['high_recall']:.3f}   "
              f"accuracy {rows[-1]['accuracy']:.3f}")

    # Ranked by macro-F1, but High recall is shown alongside because that is the
    # metric the brief singles out. If two models are close on macro-F1, prefer
    # the one that catches more urgent tickets.
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n--- MODEL COMPARISON ---")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return results, fitted


# ---------------------------------------------------------------------------
# 5. CHARTS
# ---------------------------------------------------------------------------
def plot_comparison(results, path):
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    r = results.sort_values("macro_f1")
    y = np.arange(len(r))
    ax.barh(y - 0.19, r["macro_f1"], height=0.34, color=BLUE, label="Macro F1")
    ax.barh(y + 0.19, r["high_recall"], height=0.34, color="#9dc3ee",
            label="Recall (High)")
    for i, (f1, hr) in enumerate(zip(r["macro_f1"], r["high_recall"])):
        ax.text(f1 + 0.012, i - 0.19, f"{f1:.3f}", va="center", fontsize=8.5, color=INK)
        ax.text(hr + 0.012, i + 0.19, f"{hr:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, r["model"], fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", fontsize=9, color=MUTED)
    ax.set_title("PS-2 priority detection - overall vs urgent-ticket performance",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.tick_params(labelsize=9, colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.grid(axis="x", color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(yte, pred, path):
    cm = confusion_matrix(yte, pred, labels=PRIORITY_ORDER, normalize="true")
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(3), PRIORITY_ORDER, fontsize=9)
    ax.set_yticks(range(3), PRIORITY_ORDER, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("Actual", fontsize=9, color=MUTED)
    ax.set_title("Priority confusion matrix (row-normalised)",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if cm[i, j] > 0.55 else INK)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. ERROR ANALYSIS - with the dangerous errors called out separately
# ---------------------------------------------------------------------------
def error_analysis(texts, yte, pred, path):
    wrong = pd.DataFrame({"text": texts, "actual": yte.values, "predicted": pred})
    wrong = wrong[wrong["actual"] != wrong["predicted"]]
    wrong.to_csv(path, index=False)

    print(f"\n--- ERROR ANALYSIS ---  {len(wrong):,} misclassified "
          f"({len(wrong)/len(yte):.1%})  ->  {path}")
    print("\nmost common confusions:")
    print(wrong.groupby(["actual", "predicted"]).size()
          .sort_values(ascending=False).head(6).to_string())

    # The costly failure: a genuinely urgent ticket sent to the routine queue.
    missed = wrong[(wrong["actual"] == "High") & (wrong["predicted"] == "Low")]
    print(f"\nHigh tickets called Low (the expensive mistake): {len(missed)}")
    for t in missed["text"].head(5):
        print(f"  - {t[:140]}")


# ---------------------------------------------------------------------------
# 7. GOLD SET
# ---------------------------------------------------------------------------
def gold_eval(model, vec, scaler, use_sentiment):
    path = os.path.join(DATA_DIR, "gold_set_labeled.csv")
    if not os.path.exists(path):
        print("\n--- GOLD SET ---")
        print("  gold_set_labeled.csv not found. Run 06_gold_labeler.py.")
        return

    gold = pd.read_csv(path)
    gold = gold[gold["true_priority"].notna()
                & (gold["true_priority"].astype(str).str.strip() != "")]
    if gold.empty:
        print("\n--- GOLD SET --- file found but true_priority not filled in.")
        return

    raw = gold["customer_text"].fillna("")
    letters = raw.str.count(r"[A-Za-z]").clip(lower=1)
    num = pd.DataFrame({
        "caps_ratio": raw.str.count(r"[A-Z]") / letters,
        "exclaims": raw.str.count(r"!"),
        "questions": raw.str.count(r"\?"),
        "char_len": raw.str.len(),
    })
    if use_sentiment:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        num["sentiment"] = raw.map(lambda t: sia.polarity_scores(t)["compound"])

    X = hstack([vec.transform(raw), csr_matrix(scaler.transform(num))]).tocsr()
    pred = model.predict(X)

    print(f"\n--- GOLD SET ({len(gold)} hand-labelled tickets) ---")
    print(f"accuracy    {accuracy_score(gold['true_priority'], pred):.3f}")
    print(f"macro-F1    {f1_score(gold['true_priority'], pred, average='macro'):.3f}")
    print(f"HIGH recall {recall_score(gold['true_priority'], pred, labels=['High'], average='macro', zero_division=0):.3f}")
    print("\n" + classification_report(gold["true_priority"], pred, zero_division=0))


# ---------------------------------------------------------------------------
def main():
    df = load_data()
    extra = add_sentiment(df)
    feature_cols = NUMERIC_FEATURES + extra

    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"train.csv is missing {missing}. Re-run 02_clean_and_label.py.")

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=df["priority"])

    vec, scaler, Xtr, Xte = build_features(
        df.loc[train_idx, "clean_text"], df.loc[test_idx, "clean_text"],
        df.loc[train_idx, feature_cols], df.loc[test_idx, feature_cols])

    ytr, yte = df.loc[train_idx, "priority"], df.loc[test_idx, "priority"]
    results, fitted = train_and_compare(Xtr, ytr, Xte, yte)

    # --- the trade-off you have to decide, not the script -------------------
    top_f1 = results.iloc[0]["model"]
    top_recall = results.sort_values("high_recall", ascending=False).iloc[0]["model"]
    if top_f1 != top_recall:
        a = results.set_index("model")
        print("\n" + "=" * 68)
        print("DECISION REQUIRED - the two metrics disagree")
        print("=" * 68)
        print(f"  Best macro-F1     : {top_f1} "
              f"(F1 {a.loc[top_f1, 'macro_f1']:.3f}, "
              f"High recall {a.loc[top_f1, 'high_recall']:.3f})")
        print(f"  Best High recall  : {top_recall} "
              f"(F1 {a.loc[top_recall, 'macro_f1']:.3f}, "
              f"High recall {a.loc[top_recall, 'high_recall']:.3f})")
        print("\n  Trading "
              f"{a.loc[top_f1, 'macro_f1'] - a.loc[top_recall, 'macro_f1']:+.3f} macro-F1 "
              f"for {a.loc[top_recall, 'high_recall'] - a.loc[top_f1, 'high_recall']:+.3f} "
              "recall on urgent tickets.")
        print("\n  The brief says missing an urgent ticket costs more than a false")
        print("  alarm, which argues for the second. Set SELECT_BY at the top of")
        print("  this file, and explain the choice in your report.")
        print("=" * 68)

    best_name = (top_recall if SELECT_BY == "high_recall" else top_f1)
    best = fitted[best_name]
    pred = best.predict(Xte)

    print(f"\n=== SELECTED MODEL ({SELECT_BY}): {best_name} ===")
    print(classification_report(yte, pred, labels=PRIORITY_ORDER, zero_division=0))

    high_recall = recall_score(yte, pred, labels=["High"], average="macro",
                               zero_division=0)
    print(f">>> RECALL ON HIGH-PRIORITY TICKETS: {high_recall:.3f}")
    print("    This is the headline metric for PS-2. It says what share of")
    print("    genuinely urgent tickets the system would actually escalate.")

    plot_comparison(results, os.path.join(OUT_DIR, "ps2_model_comparison.png"))
    plot_confusion(yte, pred, os.path.join(OUT_DIR, "ps2_confusion_matrix.png"))
    error_analysis(df.loc[test_idx, "clean_text"].values, yte, pred,
                   os.path.join(OUT_DIR, "ps2_errors.csv"))
    gold_eval(best, vec, scaler, bool(extra))

    joblib.dump({"vectorizer": vec, "scaler": scaler, "model": best,
                 "name": best_name, "feature_cols": feature_cols},
                os.path.join(OUT_DIR, "ps2_priority_model.joblib"))
    results.to_csv(os.path.join(OUT_DIR, "ps2_model_comparison.csv"), index=False)
    print(f"\nsaved model + charts + tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
