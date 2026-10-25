"""
Casting Defect Detection: demo app for QC operators.

Upload one or more images of cast parts -> OK / DEFECTIVE + confidence + Grad-CAM heatmap.
Every inspection is added to a session log with a simple summary (total inspected, defect rate).

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

The app needs:
    models/resnet50v2.keras   (from notebook 02)
    models/threshold.json     (from notebook 03, optional; default threshold 0.5)

If the model is too large for GitHub, upload both files to a Hugging Face model repo and set
HF_MODEL_REPO (environment variable or Streamlit secret), e.g. HF_MODEL_REPO = "gbrlhf/casting-defect-resnet50v2".
The app then downloads them on first start.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "resnet50v2.keras"))
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Casting Defect Inspection", page_icon="🔍", layout="wide")


def get_setting(name):
    """Read a setting from an environment variable or Streamlit secrets (if any)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file configured
        return None


@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def download_from_hub(repo_id: str):
    """Download resnet50v2.keras (+ threshold.json if present) from a Hugging Face model repo."""
    from huggingface_hub import hf_hub_download

    model_file = hf_hub_download(repo_id=repo_id, filename="resnet50v2.keras")
    try:
        threshold_file = hf_hub_download(repo_id=repo_id, filename="threshold.json")
    except Exception:
        threshold_file = None
    return model_file, threshold_file


HF_MODEL_REPO = get_setting("HF_MODEL_REPO")
if not MODEL_PATH.exists() and HF_MODEL_REPO:
    try:
        hub_model, hub_threshold = download_from_hub(HF_MODEL_REPO)
        MODEL_PATH = Path(hub_model)
        if hub_threshold and not THRESHOLD_PATH.exists():
            THRESHOLD_PATH = Path(hub_threshold)
    except Exception as exc:
        st.error(f"Could not download the model from Hugging Face repo `{HF_MODEL_REPO}`: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    import keras  # imported here so the page renders quickly before TensorFlow loads

    if not Path(model_path).exists():
        return None, None
    model = keras.models.load_model(model_path)
    return model, split_model(model)


def split_model(model):
    """Split a transfer-learning model from notebook 02 into (pre-layers, backbone, head-layers)."""
    import keras

    layer_list = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    backbone_idx = next(
        (i for i, l in enumerate(layer_list) if isinstance(l, keras.Model) and l.name != "augmentation"),
        None,
    )
    if backbone_idx is None:
        return None  # e.g. the baseline CNN: prediction still works, Grad-CAM is skipped
    pre = [l for l in layer_list[:backbone_idx] if l.name != "augmentation"]
    return pre, layer_list[backbone_idx], layer_list[backbone_idx + 1:]


def load_default_threshold():
    try:
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Prediction and Grad-CAM (same method as notebook 03)
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 RGB with raw 0-255 values; the model has its own scaling built in."""
    return np.asarray(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32)


def predict_with_gradcam(model, parts, img_array):
    """Return (p_defect, heatmap or None)."""
    import tensorflow as tf

    if parts is None:
        p = float(model(img_array[None, ...], training=False).numpy().ravel()[0])
        return p, None

    pre, backbone, head = parts
    x = tf.convert_to_tensor(img_array[None, ...])
    for layer in pre:
        x = layer(x)
    with tf.GradientTape() as tape:
        features = backbone(x, training=False)
        tape.watch(features)
        y = features
        for layer in head[:-1]:
            y = layer(y, training=False)
        dense = head[-1]
        logit = tf.matmul(y, dense.kernel) + dense.bias  # score before sigmoid, avoids vanishing gradients
        score = logit[:, 0]  # must be computed inside the tape so the gradient can be traced
    grads = tape.gradient(score, features)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return float(tf.sigmoid(logit[0, 0])), cam.numpy()


def overlay(img_array, heatmap, alpha=0.4):
    heat = np.asarray(Image.fromarray(np.uint8(heatmap * 255)).resize(IMG_SIZE, Image.BILINEAR)) / 255.0
    colored = cm.jet(heat)[..., :3] * 255
    return np.clip((1 - alpha) * img_array + alpha * colored, 0, 255).astype("uint8")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = []

st.title("🔍 Casting Defect Inspection")
st.caption(
    "AI-assisted visual quality inspection for cast metal parts. "
    "Upload top-view images to classify each part as **OK** or **DEFECTIVE**, "
    "with a Grad-CAM heatmap showing where the model looked."
)

model, parts = load_model(str(MODEL_PATH))
if model is None:
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Put `resnet50v2.keras` from notebook 02 in the `models/` folder, "
        "or set `HF_MODEL_REPO` to a Hugging Face repo that contains it."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    default_t = load_default_threshold()
    threshold = st.slider(
        "Decision threshold",
        min_value=0.01, max_value=0.99, value=default_t, step=0.01,
        help="A part is flagged DEFECTIVE if its defect probability is at or above this value. "
             "Lower = fewer missed defects but more false alarms.",
    )
    st.caption(f"Tuned threshold from validation: **{default_t:.2f}**")
    show_cam = st.toggle("Show Grad-CAM heatmap", value=parts is not None, disabled=parts is None)
    st.divider()
    st.markdown(
        "**Model:** `{}`  \n**Input:** 224×224 RGB  \n"
        "**Note:** trained on pump-impeller casting images; real production use needs validation on new data."
        .format(model.name)
    )

files = st.file_uploader(
    "Upload product images", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)

if files:
    for f in files:
        image = Image.open(f)
        img_array = preprocess(image)
        p_defect, heatmap = predict_with_gradcam(model, parts, img_array)
        is_defect = p_defect >= threshold
        label = "DEFECTIVE" if is_defect else "OK"
        confidence = p_defect if is_defect else 1 - p_defect

        with st.container(border=True):
            cols = st.columns([1, 1, 1.2]) if show_cam and heatmap is not None else st.columns([1, 1.2])
            cols[0].image(img_array.astype("uint8"), caption=f.name, use_container_width=True)
            if show_cam and heatmap is not None:
                cols[1].image(overlay(img_array, heatmap), caption="Grad-CAM", use_container_width=True)
            result_col = cols[-1]
            if is_defect:
                result_col.error(f"### ❌ {label}")
            else:
                result_col.success(f"### ✅ {label}")
            result_col.metric("Confidence", f"{confidence:.1%}")
            result_col.progress(p_defect, text=f"Defect probability: {p_defect:.3f} (threshold {threshold:.2f})")

        # Log each file once per upload
        key = (f.name, f.size, round(threshold, 2))
        if key not in {(r["file"], r["size"], r["threshold"]) for r in st.session_state.log}:
            st.session_state.log.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": f.name, "size": f.size, "result": label,
                "p_defect": round(p_defect, 4), "threshold": round(threshold, 2),
            })
else:
    st.info("Upload one or more images to start the inspection.")

# ---------------------------------------------------------------------------
# Inspection log and summary
# ---------------------------------------------------------------------------
if st.session_state.log:
    st.divider()
    st.subheader("📋 Inspection log (this session)")
    log_df = pd.DataFrame(st.session_state.log).drop(columns="size")
    total = len(log_df)
    n_def = int((log_df["result"] == "DEFECTIVE").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total inspected", total)
    c2.metric("OK", total - n_def)
    c3.metric("Defective", n_def)
    c4.metric("Defect rate", f"{n_def / total:.1%}")

    st.dataframe(log_df, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    left.download_button(
        "⬇️ Download log (CSV)", log_df.to_csv(index=False).encode(),
        file_name=f"inspection_log_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
    )
    if right.button("🗑️ Clear log"):
        st.session_state.log = []
        st.rerun()
