"""
STEP 4 - Export trimmed files for Colab
=======================================

Run this LOCALLY, once, before you open the notebook.

Why not just upload support_pairs.csv? It is 374 MB. Uploading that through a
browser is slow, and Colab's disk is wiped every time the session dies - so you
would be re-uploading it constantly. Instead we cut it down to only the columns
and rows the deep-learning models actually need:

    colab_ps1.csv  ~ 25 MB   text + intent label      (PS-1 LSTM)
    colab_ps3.csv  ~ 30 MB   query + response pairs   (PS-3 Seq2Seq)

Upload those two to Google Drive once and you never touch them again.

--------------------------------------------------------------------------
PREREQUISITE - read this if you got a FileNotFoundError
--------------------------------------------------------------------------
This script reads the OUTPUT of 02_clean_and_label.py, so 02 must have been run
with the SAME setting of USE_SAMPLE that you set below.

    USE_SAMPLE = True   needs  train_sample.csv  +  support_pairs_sample.csv
    USE_SAMPLE = False  needs  train.csv         +  support_pairs.csv

If you have only ever run 02 on the sample, either set USE_SAMPLE = True here,
or (better) go back and run 02 once with USE_SAMPLE = False so the deep-learning
models get the full dataset. The pre-flight check below tells you which files
are missing rather than dumping a traceback.

Run:  python 04_export_for_colab.py
"""

import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\data"
OUT_DIR = os.path.join(DATA_DIR, "colab")

# Must match what you used in 02_clean_and_label.py. See the note above.
USE_SAMPLE = True

SUFFIX = "_sample" if USE_SAMPLE else ""
TRAIN_FILE = os.path.join(DATA_DIR, f"train{SUFFIX}.csv")
PAIRS_FILE = os.path.join(DATA_DIR, f"support_pairs{SUFFIX}.csv")

# How many rows to keep. These caps are about training TIME, not disk space.
# 200k classification rows and 150k generation pairs are plenty for a project
# like this, and both fit comfortably on a free T4.
N_PS1 = 200_000
N_PS3 = 150_000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# PRE-FLIGHT CHECK
#
# Failing early with a readable message beats a traceback forty lines down.
# ---------------------------------------------------------------------------
def preflight():
    missing = [p for p in (TRAIN_FILE, PAIRS_FILE) if not os.path.exists(p)]
    if not missing:
        return

    print("Cannot run - required input file(s) not found:\n")
    for p in missing:
        print(f"   missing: {p}")

    print(f"\nUSE_SAMPLE is currently {USE_SAMPLE}, so this script expects the "
          f"'{SUFFIX or 'full'}' outputs of 02_clean_and_label.py.")

    present = sorted(f for f in os.listdir(DATA_DIR)
                     if f.endswith(".csv")) if os.path.isdir(DATA_DIR) else []
    if present:
        print("\nWhat is actually in your data folder:")
        for f in present:
            size = os.path.getsize(os.path.join(DATA_DIR, f)) / 1e6
            print(f"   {f:<32} {size:8.1f} MB")

    print("\nFix it one of two ways:")
    print("  A) Set USE_SAMPLE = True in this file, to work from the 50k sample.")
    print("  B) Open 02_clean_and_label.py, set USE_SAMPLE = False, run it once")
    print("     to build the full train.csv, then set USE_SAMPLE = False here.")
    print("\n  B is better - the LSTM and Seq2Seq models want the full dataset.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLEANING (the brand replies need it too - see below)
# ---------------------------------------------------------------------------
RE_MENTION = re.compile(r"@\w+")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_ENTITY = re.compile(r"&amp;|&gt;|&lt;|&quot;|&#\d+;")
RE_SPACE = re.compile(r"\s+")


def clean(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    for rx in (RE_URL, RE_MENTION, RE_ENTITY):
        s = s.str.replace(rx, " ", regex=True)
    return s.str.replace(RE_SPACE, " ", regex=True).str.strip()


# ---------------------------------------------------------------------------
def export_ps1():
    """text + intent, stratified down to N_PS1 rows."""
    cols = pd.read_csv(TRAIN_FILE, nrows=0).columns
    want = [c for c in ("clean_text", "intent", "priority") if c in cols]
    df = pd.read_csv(TRAIN_FILE, usecols=want).dropna(subset=["clean_text", "intent"])

    if len(df) > N_PS1:
        # groupby(...).sample(frac=) keeps each class at its original share, so
        # the small Account class is not wiped out by a plain random sample.
        frac = N_PS1 / len(df)
        df = df.groupby("intent", group_keys=False).sample(
            frac=frac, random_state=RANDOM_STATE)

    path = os.path.join(OUT_DIR, "colab_ps1.csv")
    df.to_csv(path, index=False)
    print(f"PS-1: {len(df):,} rows -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    print(df["intent"].value_counts().to_string())
    return path


def export_ps3():
    """customer query -> company response pairs."""
    pairs = pd.read_csv(PAIRS_FILE, usecols=["customer_text", "response_text"])
    pairs["customer_text"] = clean(pairs["customer_text"])
    pairs["response_text"] = clean(pairs["response_text"])

    # The brand replies need cleaning too. They are full of "@123456" handles and
    # tracking links, and a Seq2Seq model will happily learn to generate those -
    # giving fluent-looking output that is pure noise.
    #
    # Then drop pairs too short to be a real exchange, or so long they inflate the
    # sequence length for every other row in the batch.
    ok = (pairs["customer_text"].str.split().str.len().between(3, 40) &
          pairs["response_text"].str.split().str.len().between(3, 40))
    pairs = pairs[ok]
    print(f"\nPS-3: {len(pairs):,} pairs survive the 3-40 word filter")

    if len(pairs) > N_PS3:
        pairs = pairs.sample(N_PS3, random_state=RANDOM_STATE)

    path = os.path.join(OUT_DIR, "colab_ps3.csv")
    pairs.to_csv(path, index=False)
    print(f"PS-3: {len(pairs):,} rows -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def main():
    preflight()
    os.makedirs(OUT_DIR, exist_ok=True)
    export_ps1()
    export_ps3()
    print(f"\nDone. Upload both files in {OUT_DIR} to Google Drive, "
          f"into a folder called 'support_ticket_project'.")


if __name__ == "__main__":
    main()
