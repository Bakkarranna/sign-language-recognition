"""Interactive replay console for the committed SignGRU, over real WLASL100 instances.

Run it from the repo root (the package must be installed, as in the quickstart):

    .venv/Scripts/python.exe -m pip install -e .          # once
    .venv/Scripts/python.exe -m streamlit run streamlit_app.py

Everything the page renders comes from the local artefacts: the committed
``results/sign_gru.pt`` checkpoint and the ``data/processed/landmarks`` cache.
Nothing is retrained, so the replay cannot disagree with ``results/``.
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sign_language import config, replay
from sign_language import dataset as ds

st.set_page_config(page_title="SignGRU replay - WLASL100", layout="wide")

REPLAY_DELAY = 0.08      # seconds between frames in the motion replay


@st.cache_resource
def load_model():
    """The checkpoint, loaded once per browser session."""
    return replay.load_model()


@st.cache_data
def replay_cached(split: str) -> list[replay.Prediction]:
    return replay.replay_split(split=split)


@st.cache_data
def frames_cached(video_id: str) -> np.ndarray:
    """(32, 75, 3) normalised landmarks for one instance."""
    return np.asarray(ds.load_landmarks(video_id, config.LANDMARK_DIR),
                      dtype=np.float32)


def metrics_file() -> dict:
    """The training run's summary, when it is on disk."""
    path = config.RESULTS_DIR / "sign_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_landmarks(seq, frame: int, pred: replay.Prediction) -> None:
    """Draw one frame of the replayed instance."""
    fig = replay.landmark_figure(seq, frame=frame, gloss=pred.gloss,
                                 predicted=pred.predicted)
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def render_topk(pred: replay.Prediction) -> None:
    fig = replay.topk_figure(pred)
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("SignGRU replay - WLASL100")
st.caption(
    "The committed checkpoint replayed on real cached instances: pick a clip, scrub "
    "through its 32 normalised landmark frames, and see the gloss the biGRU predicted."
)

try:
    model, labels = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

summary = metrics_file().get("test", {})
header = ["**model** SignGRU (biGRU, 2x256)",
          f"**top-1** {summary.get('accuracy', float('nan')):.4f}",
          f"**top-5** {summary.get('top5', float('nan')):.4f}" if 'top5' in summary else None,
          f"**chance** {1.0 / config.N_CLASSES:.2f}"]
st.markdown(" · ".join(h for h in header if h))

with st.sidebar:
    st.header("Instance")
    split = st.radio("split", ["test", "val", "train"], index=0,
                     format_func=str.capitalize)
    try:
        predictions = replay_cached(split)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    stats = replay.summary(predictions)
    st.caption(
        f"{stats['n']} instances · top-1 {stats['top1']:.3f} · "
        f"top-5 {stats['top5']:.3f} (chance {stats['chance_top1']:.2f})"
    )
    only_wrong = st.checkbox("only incorrect predictions", value=False)
    gloss_query = st.text_input("filter by true gloss", value="").strip().lower()

    options = list(predictions)
    if only_wrong:
        options = [p for p in options if not p.correct]
    if gloss_query:
        options = [p for p in options if gloss_query in p.gloss.lower()]
    if not options:
        st.warning("no instances match that filter")
        st.stop()

    def label(p: replay.Prediction) -> str:
        flag = "correct" if p.correct else "WRONG"
        return f"{p.gloss} · {p.video_id} · {flag}"

    st.caption("recomputed from the checkpoint on every session, not read from "
               "sign_predictions.npz")

    pred = st.selectbox("instance", options, format_func=label,
                        help="the replayed prediction for each clip")

# --------------------------------------------------------------------------
# Selected instance: landmarks, prediction, top-5
# --------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Landmark replay")
    seq = frames_cached(pred.video_id)
    frame = st.slider("frame", 0, config.N_FRAMES - 1, 0,
                      help="32 frames, resampled at extraction time")
    render_landmarks(seq, frame, pred)
    if st.button("▶ Play 32 frames", use_container_width=True):
        holder = st.empty()
        for f in range(config.N_FRAMES):
            fig = replay.landmark_figure(seq, frame=f, gloss=pred.gloss,
                                         predicted=pred.predicted)
            holder.pyplot(fig)
            plt.close(fig)
            time.sleep(REPLAY_DELAY)
        holder.empty()
        render_landmarks(seq, config.N_FRAMES - 1, pred)
    st.caption("pose (grey), left hand (blue), right hand (red); bands are the "
               "wrist trails. Shoulder-centred and shoulder-width scaled.")

with right:
    st.subheader("Prediction")
    verdict = f"✅ {pred.predicted}" if pred.correct else f"❌ {pred.predicted}"
    st.markdown(f"**true gloss:** {pred.gloss}  ·  **predicted:** {verdict}")
    st.caption(f"instance {pred.video_id} · signer {pred.signer_id} · "
               f"true gloss ranked #{pred.rank} of {config.N_CLASSES}")
    render_topk(pred)
    st.caption("probabilities from the softmax over the 100 WLASL100 glosses")

# --------------------------------------------------------------------------
# Whole split
# --------------------------------------------------------------------------
st.divider()
rows = [p.as_row() for p in predictions]
table = pd.DataFrame(rows, columns=["video_id", "true", "predicted",
                                    "correct", "rank", "top5", "signer"])
col1, col2, col3 = st.columns(3)
col1.metric("instances", stats["n"])
col2.metric("top-1", f"{stats['top1']:.3f}", f"{stats['n_correct']} correct")
col3.metric("top-5", f"{stats['top5']:.3f}")
view = st.radio("table", ["all", "correct", "incorrect"], index=0,
                horizontal=True, format_func=str.capitalize,
                label_visibility="collapsed")
mask = {"all": table["correct"], "correct": table["correct"],
        "incorrect": ~table["correct"]}[view]
st.dataframe(table[mask], use_container_width=True, hide_index=True,
             column_config={"top5": st.column_config.TextColumn("top-5", width="large")})
