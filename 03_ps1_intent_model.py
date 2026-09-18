"""
STEP 3 - PS-1: Ticket Intent Classification
===========================================

Now we actually build the thing the brief asks for: given a customer message,
predict which of five departments it belongs to.

WHAT THIS SCRIPT DOES
  1. Turns text into numbers (TF-IDF)
  2. Trains four classical classifiers and compares them honestly
  3. Reports accuracy / precision / recall / macro-F1
  4. Draws a confusion matrix
  5. Does error analysis - exports what it got wrong so you can read it
  6. Scores against your hand-labelled gold set, if you have filled it in
  7. Saves the model so PS-4 can reuse its predictions as a feature

READ THIS BEFORE YOU BELIEVE ANY NUMBER IT PRINTS
  The labels came from keyword rules in step 2. This model is therefore being
  scored on how well it imitates those rules. A high macro-F1 here means
  "the model learned my keywords", NOT "the model understands intent".
  The gold-set score at the bottom is the number that actually means something.
  Expect it to be meaningfully lower. That gap is a finding, not a failure -
  write about it.

Run:  python 03_ps1_intent_model.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")           # write PNGs without needing a display window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\data"
OUT_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\outputs"
USE_SAMPLE = True
SUFFIX = "_sample" if USE_SAMPLE else ""

# RandomForest gets slow on hundreds of thousands of sparse rows. Cap the
# training set while you are experimenting; raise it for the final run.
MAX_ROWS = 200_000
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

# Chart colours: one hue, light to dark, for the confusion matrix (it encodes
# MAGNITUDE, so a single-hue ramp is correct - never a rainbow, which invents
# category boundaries that are not in the data).
BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", BLUE, "#123a68"])


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data():
    path = os.path.join(DATA_DIR, f"train{SUFFIX}.csv")
    df = pd.read_csv(path).dropna(subset=["clean_text", "intent"])
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)
    print(f"Loaded {len(df):,} labelled rows from {path}")
    print("\nclass balance:")
    print(df["intent"].value_counts().to_string())
    print(f"\nsmallest class is {df['intent'].value_counts().min():,} rows "
          f"({df['intent'].value_counts(normalize=True).min():.1%})")
    return df


# ---------------------------------------------------------------------------
# 2. SPLIT
#
# stratify=y is essential here. Account is only ~6% of the data; a random split
# could easily leave the test set with barely any of it, and then its recall
# score would be noise. Stratifying keeps each class at the same proportion in
# both halves.
# ---------------------------------------------------------------------------
def split(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["intent"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["intent"],
    )
    print(f"\ntrain {len(X_train):,}   test {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. VECTORISE - turning text into numbers
#
# TF-IDF = Term Frequency x Inverse Document Frequency. Two ideas multiplied:
#   TF  - how often a word appears in THIS ticket (frequent here = important here)
#   IDF - how rare the word is across ALL tickets (rare overall = informative)
# So "refund" scores high: common in the ticket it appears in, rare overall.
# "the" scores near zero: it is everywhere, so it distinguishes nothing.
#
# Parameter choices, and why:
#   ngram_range=(1,2) - single words AND adjacent pairs. Crucial here: "not
#                       working" and "working" mean opposite things, and
#                       unigrams alone cannot tell them apart.
#   min_df=3          - ignore terms in fewer than 3 tickets. Kills typos and
#                       one-off usernames that would otherwise be memorised.
#   max_df=0.6        - ignore terms in >60% of tickets. This is how we drop
#                       stopwords WITHOUT a stopword list - important, because
#                       standard lists remove "not", "no" and "can't", which
#                       carry real meaning in support tickets.
#   sublinear_tf=True - use 1+log(count) instead of raw count, so a word said
#                       ten times is not treated as ten times more important.
# ---------------------------------------------------------------------------
def vectorise(X_train, X_test):
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.6,
        max_features=50_000, sublinear_tf=True, strip_accents="unicode",
    )
    Xtr = vec.fit_transform(X_train)   # fit on TRAIN ONLY
    Xte = vec.transform(X_test)        # test just gets transformed
    print(f"\nvocabulary: {len(vec.vocabulary_):,} terms")
    print(f"matrix: {Xtr.shape[0]:,} x {Xtr.shape[1]:,}, "
          f"{Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1]):.4%} non-zero (sparse)")
    return vec, Xtr, Xte


# ---------------------------------------------------------------------------
# 4. TRAIN AND COMPARE
#
# Why these four:
#   LogisticRegression - the honest baseline. Linear, fast, and you can read
#                        its coefficients to see WHICH words drove a decision.
#   LinearSVC          - usually the strongest classical model on sparse text.
#   MultinomialNB      - the classic text baseline. Very fast, often weaker,
#                        but if it beats the others your features are too thin.
#   RandomForest       - non-linear. Tends to underperform on high-dimensional
#                        sparse text, and it is worth showing that in a report
#                        rather than assuming "trees are better".
#
# class_weight="balanced" makes each class contribute equally to the loss
# regardless of size. Without it, Account (6% of rows) gets ignored - the model
# learns it can safely never predict it.
#
# JUDGE ON MACRO-F1, NOT ACCURACY. Macro-F1 averages the F1 of each class
# equally, so failing on the small class actually costs you. Accuracy would let
# a model ignore Account entirely and still look fine.
# ---------------------------------------------------------------------------
def train_and_compare(Xtr, ytr, Xte, yte):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "MultinomialNB": MultinomialNB(alpha=0.3),
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
            "weighted_f1": f1_score(yte, pred, average="weighted"),
        })
        fitted[name] = model
        print(f"  macro-F1 {rows[-1]['macro_f1']:.3f}   "
              f"accuracy {rows[-1]['accuracy']:.3f}")

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n--- MODEL COMPARISON ---")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return results, fitted


# ---------------------------------------------------------------------------
# 5. CHARTS
# ---------------------------------------------------------------------------
def plot_comparison(results, path):
    """One series, so no legend - the title names it. Values labelled directly."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    r = results.sort_values("macro_f1")
    bars = ax.barh(r["model"], r["macro_f1"], color=BLUE, height=0.6)
    for bar, v in zip(bars, r["macro_f1"]):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, min(1.0, r["macro_f1"].max() + 0.12))
    ax.set_xlabel("Macro F1", fontsize=9, color=MUTED)
    ax.set_title("PS-1 intent classification - model comparison",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.tick_params(labelsize=9, colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.grid(axis="x", color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(yte, pred, labels, path):
    """
    Row-normalised: each row sums to 1, so a cell reads "of all real Billing
    tickets, this share was predicted as X". That IS recall on the diagonal.
    Raw counts would just show you which class is biggest.
    """
    cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("Actual", fontsize=9, color=MUTED)
    ax.set_title("Confusion matrix (row-normalised = recall on diagonal)",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > 0.55 else INK)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. WHAT DID IT LEARN? (interpretability)
#
# For a linear model, each class has a coefficient per term. The largest ones
# are literally the words pushing a ticket into that class. Print them and
# compare against your step-2 lexicon: if they are identical, the model has
# just memorised your rules and learned nothing extra. Some overlap is
# expected; total overlap is a warning sign.
# ---------------------------------------------------------------------------
def top_terms(model, vec, n=15):
    if not hasattr(model, "coef_"):
        return
    names = np.array(vec.get_feature_names_out())
    print("\n--- TOP TERMS PER CLASS ---")
    for idx, cls in enumerate(model.classes_):
        top = np.argsort(model.coef_[idx])[-n:][::-1]
        print(f"\n{cls:10s}: {', '.join(names[top])}")


# ---------------------------------------------------------------------------
# 7. ERROR ANALYSIS
#
# The brief explicitly asks for this, and it is the easiest place to earn marks.
# Do not just report the number - read the mistakes. Most will fall into a few
# recognisable buckets (e.g. billing complaints landing in Complaint, which is
# arguably the rules' fault rather than the model's).
# ---------------------------------------------------------------------------
def error_analysis(X_test, yte, pred, path, n_show=12):
    wrong = pd.DataFrame({
        "text": X_test.values, "actual": yte.values, "predicted": pred,
    })
    wrong = wrong[wrong["actual"] != wrong["predicted"]]
    wrong.to_csv(path, index=False)
    print(f"\n--- ERROR ANALYSIS ---  {len(wrong):,} misclassified "
          f"({len(wrong) / len(yte):.1%})  ->  {path}")
    print("\nmost common confusions:")
    print((wrong.groupby(["actual", "predicted"]).size()
           .sort_values(ascending=False).head(8).to_string()))
    print("\nexamples:")
    for _, r in wrong.sample(min(n_show, len(wrong)), random_state=1).iterrows():
        print(f"\n  actual={r['actual']:10s} predicted={r['predicted']:10s}")
        print(f"  {r['text'][:150]}")


# ---------------------------------------------------------------------------
# 8. GOLD-SET EVALUATION - the only score that is not self-referential
#
# Fill in gold_set_TO_LABEL.csv by hand, save it as gold_set_labeled.csv,
# and this block evaluates the model against YOUR judgement instead of
# against the keyword rules.
# ---------------------------------------------------------------------------
def gold_eval(model, vec):
    path = os.path.join(DATA_DIR, "gold_set_labeled.csv")
    if not os.path.exists(path):
        print("\n--- GOLD SET ---")
        print("  gold_set_labeled.csv not found - skipping.")
        print("  Fill in true_intent in gold_set_TO_LABEL.csv, save it under")
        print("  that name, and re-run. This is the score that matters.")
        return

    gold = pd.read_csv(path)
    gold = gold[gold["true_intent"].notna() & (gold["true_intent"].str.strip() != "")]
    if gold.empty:
        print("\n--- GOLD SET --- file found but no labels filled in yet.")
        return

    from_text = gold["customer_text"].fillna("")
    pred = model.predict(vec.transform(from_text))
    print(f"\n--- GOLD SET ({len(gold)} hand-labelled tickets) ---")
    print(f"accuracy  {accuracy_score(gold['true_intent'], pred):.3f}")
    print(f"macro-F1  {f1_score(gold['true_intent'], pred, average='macro'):.3f}")
    print("\n" + classification_report(gold["true_intent"], pred, zero_division=0))
    print("Compare this with the test-set score above. The gap is your finding.")


# ---------------------------------------------------------------------------
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    vec, Xtr, Xte = vectorise(X_train, X_test)
    results, fitted = train_and_compare(Xtr, y_train, Xte, y_test)

    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(Xte)
    labels = sorted(y_train.unique())

    print(f"\n=== BEST MODEL: {best_name} ===")
    print(classification_report(y_test, pred, zero_division=0))

    plot_comparison(results, os.path.join(OUT_DIR, "ps1_model_comparison.png"))
    plot_confusion(y_test, pred, labels,
                   os.path.join(OUT_DIR, "ps1_confusion_matrix.png"))
    top_terms(best, vec)
    error_analysis(X_test, y_test, pred,
                   os.path.join(OUT_DIR, "ps1_errors.csv"))
    gold_eval(best, vec)

    joblib.dump({"vectorizer": vec, "model": best, "name": best_name},
                os.path.join(OUT_DIR, "ps1_intent_model.joblib"))
    results.to_csv(os.path.join(OUT_DIR, "ps1_model_comparison.csv"), index=False)
    print(f"\nsaved model + charts + tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
