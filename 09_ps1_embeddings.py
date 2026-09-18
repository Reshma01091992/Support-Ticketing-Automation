"""
STEP 9 - PS-1: TF-IDF vs word embeddings
========================================

The brief asks you to "build features with TF-IDF and n-grams as a baseline,
then compare against Word2Vec / GloVe embeddings". Script 03 did the baseline.
This does the comparison, and it is the last missing piece of PS-1.

WHAT IS ACTUALLY DIFFERENT ABOUT EMBEDDINGS

    TF-IDF gives every word its own column. "refund" and "reimbursement" are two
    unrelated columns - the model has no way to know they mean nearly the same
    thing, and a ticket saying "reimbursement" gets no credit from all the
    training examples that said "refund".

    Word2Vec learns a dense vector for each word from the company it keeps. Words
    used in similar contexts end up close together, so "refund" and
    "reimbursement" land near each other and the model can generalise between
    them.

    The catch: a ticket is a sentence, not a word. To get one vector per ticket
    you have to combine the word vectors, and the simplest way - averaging - is
    lossy. "not working" and "working" average to almost the same thing, because
    averaging discards word order entirely. That is the core trade-off:

        TF-IDF     : sparse, no notion of meaning, keeps exact words and bigrams
        embeddings : dense, knows word similarity, blurs the sentence together

THREE FEATURE SETS ARE COMPARED
    1. Word2Vec, mean-pooled          - the naive baseline
    2. Word2Vec, TF-IDF-weighted mean - rare informative words count for more,
                                        which partly fixes the blurring
    3. GloVe (pretrained, optional)   - vectors trained on billions of words of
                                        general text rather than 18k tweets

    Results are then set against the TF-IDF scores script 03 already saved.

REQUIRES
    pip install gensim
    Optional: set USE_GLOVE = True to also download pretrained GloVe (~130 MB
    the first time, needs internet).

Run:  python 09_ps1_embeddings.py
"""

import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

DATA_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\data"
OUT_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\outputs"
USE_SAMPLE = True
SUFFIX = "_sample" if USE_SAMPLE else ""

MAX_ROWS = 200_000
RANDOM_STATE = 42
VECTOR_SIZE = 200          # dimensions per word vector
W2V_EPOCHS = 10
USE_GLOVE = False          # True downloads glove-twitter-200 (~130 MB) once

os.makedirs(OUT_DIR, exist_ok=True)
BLUE = "#2a78d6"
PALE = "#9dc3ee"
INK = "#0b0b0b"
MUTED = "#52514e"


# ---------------------------------------------------------------------------
# TOKENIZER - and why it needs to exist
#
# The obvious thing is text.lower().split(). Do that here and you get a nasty
# silent problem: "refund", "refund.", "refund?" and "refund!" become four
# separate words, each with its own vector trained on a quarter of the data.
# The first version of this script did exactly that, and the nearest neighbours
# of "refund" came back as: refund., product., refund?, refund!  - the model had
# learned punctuation, not meaning.
#
# sklearn's TfidfVectorizer strips punctuation for you (its default token
# pattern is \w\w+), which is why script 03 was never affected. Anything using
# a bare .split() is - INCLUDING the LSTM and Seq2Seq in notebook 05.
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN.findall(str(text).lower())


# ---------------------------------------------------------------------------
def load_data():
    path = os.path.join(DATA_DIR, f"train{SUFFIX}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}\nRun 02_clean_and_label.py first.")
    df = pd.read_csv(path).dropna(subset=["clean_text", "intent"])
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)
    print(f"Loaded {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# 1. TRAIN WORD2VEC
#
# Trained on the TRAINING TEXTS ONLY. Training it on everything would let test
# vocabulary influence the vectors - a quiet form of leakage that inflates the
# score without ever looking like cheating.
#
# sg=1 selects skip-gram (predict context from a word) over CBOW. Skip-gram is
# slower but better on small corpora and rare words, which is what you have.
# ---------------------------------------------------------------------------
def train_word2vec(train_texts):
    from gensim.models import Word2Vec
    sentences = [tokenize(t) for t in train_texts]
    print(f"\ntraining Word2Vec on {len(sentences):,} tickets "
          f"({VECTOR_SIZE}d, skip-gram, {W2V_EPOCHS} epochs) ...")
    model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=5, min_count=3,
                     sg=1, workers=4, epochs=W2V_EPOCHS, seed=RANDOM_STATE)
    print(f"  vocabulary: {len(model.wv):,} words")

    # A quick sanity check you should paste into your report - it shows the
    # embedding learned domain meaning rather than generic English.
    print("\n  nearest neighbours (does it understand support language?):")
    for probe in ("refund", "broken", "password", "delayed"):
        if probe in model.wv:
            sims = ", ".join(w for w, _ in model.wv.most_similar(probe, topn=6))
            print(f"    {probe:<10} -> {sims}")
    return model.wv


# ---------------------------------------------------------------------------
# 2. TURN WORD VECTORS INTO TICKET VECTORS
# ---------------------------------------------------------------------------
def mean_pool(texts, kv):
    """Plain average of the word vectors present in the vocabulary."""
    dim = kv.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in tokenize(t) if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def tfidf_weighted_pool(texts, kv, idf_map, default_idf):
    """
    Weighted average, where each word's weight is its IDF.

    Why bother: in a plain average, "the" counts as much as "refund". IDF is
    small for words that appear everywhere and large for rare informative ones,
    so this pulls the ticket vector towards the words that actually carry intent.
    """
    dim = kv.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs, weights = [], []
        for w in tokenize(t):
            if w in kv:
                vecs.append(kv[w])
                weights.append(idf_map.get(w, default_idf))
        if vecs:
            w = np.array(weights, dtype=np.float32)
            out[i] = np.average(np.array(vecs), axis=0, weights=w)
    return out


# ---------------------------------------------------------------------------
# 3. TRAIN CLASSIFIERS ON A GIVEN FEATURE SET
#
# Embeddings are dense and roughly zero-centred, so they get standardised - and
# unlike TF-IDF they work fine with models that assume continuous features.
# The same three classifiers are used throughout so the comparison is fair.
# ---------------------------------------------------------------------------
def evaluate(name, Xtr, Xte, ytr, yte, rows, store):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced", n_jobs=-1),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE),
    }
    for mname, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        f1 = f1_score(yte, pred, average="macro")
        rows.append({"features": name, "model": mname, "macro_f1": f1})
        print(f"  {mname:<20} macro-F1 {f1:.4f}")
        key = (name, mname)
        store[key] = (model, scaler, pred)
    return rows


# ---------------------------------------------------------------------------
def plot_comparison(results, tfidf_best, path):
    """Grouped bars: one group per feature set, one bar per classifier."""
    pivot = results.pivot(index="features", columns="model", values="macro_f1")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    n = len(pivot.columns)
    y = np.arange(len(pivot))
    colours = [BLUE, PALE, "#c9dcf5"]
    for i, col in enumerate(pivot.columns):
        off = (i - (n - 1) / 2) * 0.26
        ax.barh(y + off, pivot[col], height=0.24, color=colours[i % 3], label=col)
        for yy, v in zip(y + off, pivot[col]):
            ax.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=8, color=INK)

    if tfidf_best is not None:
        ax.axvline(tfidf_best, color="#b3402f", lw=1.4, ls="--",
                   label=f"TF-IDF baseline ({tfidf_best:.3f})")

    ax.set_yticks(y, pivot.index, fontsize=9)
    ax.set_xlabel("Macro F1", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1.14)
    ax.set_title("PS-1: embedding features vs the TF-IDF baseline",
                 fontsize=11, color=INK, loc="left", pad=12)
    # legend below the plot - inside it, it lands on top of the bars
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.tick_params(labelsize=9, colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.grid(axis="x", color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values, df["intent"].values,
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["intent"].values)
    print(f"train {len(X_train):,}   test {len(X_test):,}")

    try:
        kv = train_word2vec(X_train)
    except ImportError as exc:
        # Never swallow the real message here. "gensim is not installed" is only
        # one of several things that raise ImportError - a version clash between
        # gensim, numpy and scipy raises it too, and so does having gensim
        # installed into a different interpreter than the one running this file.
        raise SystemExit(
            f"Could not import gensim.\n\n"
            f"  actual error: {exc}\n\n"
            f"  running on : {sys.executable}\n"
            f"               Python {sys.version.split()[0]}\n\n"
            f"  If gensim IS installed, it went to a different interpreter.\n"
            f"  Install it into THIS one:\n"
            f"      \"{sys.executable}\" -m pip install gensim\n")

    # IDF weights, fitted on train only, reused for the weighted pooling
    tfidf = TfidfVectorizer(min_df=3, max_df=0.6)
    tfidf.fit(X_train)
    idf_map = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    default_idf = float(np.median(tfidf.idf_))

    rows, store = [], {}

    print("\n--- Word2Vec, mean-pooled ---")
    evaluate("W2V mean", mean_pool(X_train, kv), mean_pool(X_test, kv),
             y_train, y_test, rows, store)

    print("\n--- Word2Vec, TF-IDF weighted ---")
    evaluate("W2V tfidf-weighted",
             tfidf_weighted_pool(X_train, kv, idf_map, default_idf),
             tfidf_weighted_pool(X_test, kv, idf_map, default_idf),
             y_train, y_test, rows, store)

    if USE_GLOVE:
        print("\n--- GloVe (pretrained glove-twitter-200) ---")
        try:
            import gensim.downloader as api
            glove = api.load("glove-twitter-200")
            evaluate("GloVe pretrained", mean_pool(X_train, glove),
                     mean_pool(X_test, glove), y_train, y_test, rows, store)
        except Exception as exc:
            print(f"  skipped ({type(exc).__name__}: {exc})")

    results = pd.DataFrame(rows)

    # --- set it against the TF-IDF baseline script 03 saved -----------------
    tfidf_path = os.path.join(OUT_DIR, "ps1_model_comparison.csv")
    tfidf_best = None
    if os.path.exists(tfidf_path):
        base = pd.read_csv(tfidf_path)
        tfidf_best = float(base["macro_f1"].max())
        best_row = base.loc[base["macro_f1"].idxmax(), "model"]
        print(f"\nTF-IDF baseline (script 03): {best_row} = {tfidf_best:.4f}")
    else:
        print("\nps1_model_comparison.csv not found - run 03 for the baseline.")

    print("\n--- ALL EMBEDDING RESULTS ---")
    print(results.sort_values("macro_f1", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = results.loc[results["macro_f1"].idxmax()]
    print(f"\nbest embedding setup: {best['features']} + {best['model']} "
          f"= {best['macro_f1']:.4f}")

    if tfidf_best is not None:
        gap = tfidf_best - best["macro_f1"]
        print(f"gap to TF-IDF: {gap:+.4f}")
        if gap > 0:
            print("\n  TF-IDF wins. The explanation for your report:")
            print("  the labels were generated by keyword rules, so the signal IS")
            print("  the exact words - which is precisely what TF-IDF encodes and")
            print("  what averaging into a single vector destroys. Embeddings help")
            print("  when meaning matters more than wording; here it does not.")
        else:
            print("\n  Embeddings win - report which pooling strategy did it and why.")

    _, _, pred = store[(best["features"], best["model"])]
    print("\nper-class detail for the best embedding model:")
    print(classification_report(y_test, pred, zero_division=0))

    plot_comparison(results, tfidf_best,
                    os.path.join(OUT_DIR, "ps1_embeddings_comparison.png"))
    results.to_csv(os.path.join(OUT_DIR, "ps1_embeddings_comparison.csv"),
                   index=False)
    print(f"\nsaved chart + table -> {OUT_DIR}")


if __name__ == "__main__":
    main()
