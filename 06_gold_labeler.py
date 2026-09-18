"""
STEP 6 - Gold set annotation tool
=================================

A small Streamlit app for hand-labelling the 300-row gold set.

WHY THIS EXISTS
    The brief requires a hand-labelled gold set for evaluation only. No code can
    produce it - if a program could label these correctly, you would ship the
    program instead of training a model. Your judgement IS the ground truth.

    What code CAN do is make supplying that judgement fast and consistent. This
    app shows one ticket at a time, takes two clicks, autosaves after every row,
    and resumes where you left off if you close it.

    ~300 rows at 2 clicks each is about 20-25 minutes, versus roughly two hours
    of scrolling a spreadsheet.

HOW TO RUN
    pip install streamlit
    streamlit run 06_gold_labeler.py

    It opens in your browser. Label until the progress bar hits 100%, then close
    it. The output file gold_set_labeled.csv is what script 03 looks for.

BEFORE YOU START
    Read the category definitions in the sidebar and stick to them. Consistency
    matters more than being "right" - if you decide angry billing tickets go to
    Billing rather than Complaint, apply that every single time. Write your rule
    down; it becomes a paragraph in your report, and an examiner will ask.
"""

import os

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\ReshmaG\Downloads\GUVI\support_ticket_project\data"
SRC = os.path.join(DATA_DIR, "gold_set_TO_LABEL.csv")
OUT = os.path.join(DATA_DIR, "gold_set_labeled.csv")

INTENTS = ["Billing", "Technical", "Account", "Complaint", "General"]
PRIORITIES = ["Low", "Medium", "High"]

GUIDELINES = {
    "Billing": "Money. Charges, refunds, invoices, subscriptions, prices, payment failures.",
    "Technical": "Something is broken. App crashes, no internet, errors, device or service faults.",
    "Account": "Access and identity. Login, password, verification, locked or hacked accounts.",
    "Complaint": "Dissatisfaction is the point of the message, not the topic. Rudeness, being ignored, repeated failures.",
    "General": "Everything genuinely else. Questions, information requests, tracking, thanks and praise.",
}

PRIORITY_GUIDE = {
    "High": "Angry, threatening to leave, waiting a long time, security or safety, service outage, demands escalation.",
    "Medium": "A real problem stated calmly. The default for most tickets.",
    "Low": "No problem to solve - thanks, praise, or an idle question with no urgency.",
}


# ---------------------------------------------------------------------------
# IO - kept as plain functions so the logic is testable without Streamlit
# ---------------------------------------------------------------------------
def load_work(src_path: str, out_path: str) -> pd.DataFrame:
    """Load the gold set, merging in any labels already saved."""
    df = pd.read_csv(src_path)
    for col in ("true_intent", "true_priority", "note"):
        if col not in df.columns:
            df[col] = ""
    df[["true_intent", "true_priority", "note"]] = (
        df[["true_intent", "true_priority", "note"]].fillna("").astype(str))

    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        for col in ("true_intent", "true_priority", "note"):
            if col not in done.columns:
                done[col] = ""
        done = done.set_index("customer_tweet_id")
        for col in ("true_intent", "true_priority", "note"):
            mapped = df["customer_tweet_id"].map(done[col]).fillna("").astype(str)
            df[col] = mapped.where(mapped.str.strip() != "", df[col])
    return df


def next_unlabelled(df: pd.DataFrame) -> int:
    """Index of the first row still missing an intent, or -1 when finished."""
    todo = df.index[df["true_intent"].str.strip() == ""]
    return int(todo[0]) if len(todo) else -1


def save(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Gold set labeller", layout="centered")

    if "df" not in st.session_state:
        if not os.path.exists(SRC):
            st.error(f"Cannot find {SRC}. Run 02_clean_and_label.py first.")
            st.stop()
        st.session_state.df = load_work(SRC, OUT)
        st.session_state.i = next_unlabelled(st.session_state.df)

    df = st.session_state.df
    total = len(df)
    labelled = int((df["true_intent"].str.strip() != "").sum())

    # ---- sidebar: the definitions you are labelling against -----------------
    with st.sidebar:
        st.subheader("Intent definitions")
        for k, v in GUIDELINES.items():
            st.markdown(f"**{k}** — {v}")
        st.divider()
        st.subheader("Priority definitions")
        for k, v in PRIORITY_GUIDE.items():
            st.markdown(f"**{k}** — {v}")
        st.divider()
        st.caption(
            "Pick the *primary* purpose of the message. When two fit, choose the "
            "one the customer wants acted on, and apply that rule consistently.")

    st.title("Gold set labeller")
    st.progress(labelled / total, text=f"{labelled} of {total} labelled")

    if st.session_state.i < 0:
        st.success("All rows labelled.")
        st.write("**Intent distribution**")
        st.write(df["true_intent"].value_counts())
        st.write("**Priority distribution**")
        st.write(df["true_priority"].value_counts())
        st.info(f"Saved to {OUT} — now re-run 03_ps1_intent_model.py.")
        st.download_button("Download labelled CSV", df.to_csv(index=False),
                           file_name="gold_set_labeled.csv")
        return

    row = df.loc[st.session_state.i]

    st.caption(f"Row {st.session_state.i + 1} of {total}  ·  brand: {row['brand']}")
    st.markdown(
        f"<div style='background:#f1f3f5;padding:18px;border-radius:6px;"
        f"font-size:17px;line-height:1.5;color:#111'>{row['customer_text']}</div>",
        unsafe_allow_html=True)
    st.write("")

    # ---- intent -------------------------------------------------------------
    st.write("**Intent**")
    cols = st.columns(len(INTENTS))
    for col, label in zip(cols, INTENTS):
        kind = "primary" if st.session_state.get("pick_intent") == label else "secondary"
        if col.button(label, key=f"i_{label}", use_container_width=True, type=kind):
            st.session_state.pick_intent = label
            st.rerun()

    # ---- priority -----------------------------------------------------------
    st.write("**Priority**")
    cols = st.columns(len(PRIORITIES))
    for col, label in zip(cols, PRIORITIES):
        kind = "primary" if st.session_state.get("pick_priority") == label else "secondary"
        if col.button(label, key=f"p_{label}", use_container_width=True, type=kind):
            st.session_state.pick_priority = label
            st.rerun()

    note = st.text_input(
        "Note (optional)",
        placeholder="e.g. 'ambiguous - billing complaint, filed under Billing'",
        key=f"note_{st.session_state.i}")

    st.write("")
    c1, c2, c3 = st.columns([2, 1, 1])

    ready = st.session_state.get("pick_intent") and st.session_state.get("pick_priority")
    if c1.button("Save and next", type="primary", disabled=not ready,
                 use_container_width=True):
        df.at[st.session_state.i, "true_intent"] = st.session_state.pick_intent
        df.at[st.session_state.i, "true_priority"] = st.session_state.pick_priority
        df.at[st.session_state.i, "note"] = note
        save(df, OUT)                       # autosave every single row
        st.session_state.pop("pick_intent", None)
        st.session_state.pop("pick_priority", None)
        st.session_state.i = next_unlabelled(df)
        st.rerun()

    if c2.button("Skip", use_container_width=True):
        # Skipping is fine and honest - a tweet you cannot classify is a finding.
        # Note it in the report rather than forcing a label you do not believe.
        st.session_state.pop("pick_intent", None)
        st.session_state.pop("pick_priority", None)
        remaining = df.index[(df["true_intent"].str.strip() == "")
                             & (df.index > st.session_state.i)]
        st.session_state.i = int(remaining[0]) if len(remaining) else -1
        st.rerun()

    if c3.button("Back", use_container_width=True,
                 disabled=st.session_state.i == 0):
        st.session_state.pop("pick_intent", None)
        st.session_state.pop("pick_priority", None)
        st.session_state.i -= 1
        st.rerun()


if __name__ == "__main__":
    main()
