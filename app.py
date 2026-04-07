import json
from pathlib import Path

def prepare_uploaded_image(uploaded_file, image_size: int):
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = image.resize((image_size, image_size))

    image_np = np.array(image).astype(np.float32) / 255.0
    reshaped = image_np.reshape(1, image_size, image_size, 1)

    return image, reshaped
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩻",
    layout="wide"
)

# =========================
# Styling
# =========================
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1c2333;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 12px;
    }
    .prediction-normal {
        color: #4CAF50;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .prediction-pneumonia {
        color: #FF6B6B;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .small-note {
        color: #B0B7C3;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Constants
# =========================
MODEL_PATH = "artifacts/pneumonia_model.keras"
METADATA_PATH = "artifacts/model_metadata.json"

# =========================
# Load Model + Metadata
# =========================
@st.cache_resource
def load_trained_model(model_path: str):
    return load_model(model_path)

@st.cache_data
def load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_required_files():
    missing = []
    if not Path(MODEL_PATH).exists():
        missing.append(MODEL_PATH)
    if not Path(METADATA_PATH).exists():
        missing.append(METADATA_PATH)
    return missing

# =========================
# Image Prep
# =========================
def prepare_uploaded_image(uploaded_file, image_size: int):
    # Read uploaded file as PIL image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image_np = np.array(image)

    # Resize for model
    resized = cv2.resize(image_np, (image_size, image_size))

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Shape for model: (1, H, W, 1)
    reshaped = normalized.reshape(1, image_size, image_size, 1)

    return image, reshaped

def predict_image(model, processed_image, class_names):
    prediction = model.predict(processed_image, verbose=0)
    score = float(prediction[0][0])

    predicted_label = class_names[1] if score > 0.5 else class_names[0]
    normal_prob = 1.0 - score
    pneumonia_prob = score

    return predicted_label, normal_prob, pneumonia_prob

# =========================
# Sidebar
# =========================
st.sidebar.title("Upload X-Ray Images")
st.sidebar.markdown("Upload one or more chest X-ray images below.")

uploaded_files = st.sidebar.file_uploader(
    "Choose image files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

show_prob_bars = st.sidebar.checkbox("Show probability bars", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Files")
st.sidebar.code(MODEL_PATH, language="bash")
st.sidebar.code(METADATA_PATH, language="bash")

# =========================
# Main Title
# =========================
st.title("Chest X-Ray Pneumonia Detection")
st.markdown(
    "<p class='small-note'>Upload chest X-ray images from the sidebar to classify them as NORMAL or PNEUMONIA.</p>",
    unsafe_allow_html=True
)

# =========================
# Validate Model Files
# =========================
missing_files = check_required_files()
if missing_files:
    st.error("Missing required model files:")
    for file in missing_files:
        st.code(file)
    st.stop()

# Load model + metadata
model = load_trained_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

image_size = metadata.get("img_size", [224, 224])[0]
class_names = metadata.get("class_names", ["NORMAL", "PNEUMONIA"])

# =========================
# No Upload Yet
# =========================
if not uploaded_files:
    st.info("Upload one or more chest X-ray images from the sidebar to begin.")
    st.stop()

# =========================
# Predictions
# =========================
st.subheader("Predictions")

for uploaded_file in uploaded_files:
    try:
        original_image, processed_image = prepare_uploaded_image(uploaded_file, image_size)
        predicted_label, normal_prob, pneumonia_prob = predict_image(
            model, processed_image, class_names
        )

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(original_image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.markdown(f"<div class='metric-card'><strong>File:</strong> {uploaded_file.name}</div>", unsafe_allow_html=True)

            if predicted_label.upper() == "PNEUMONIA":
                st.markdown(
                    f"<div class='metric-card'><span class='prediction-pneumonia'>Prediction: {predicted_label}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='metric-card'><span class='prediction-normal'>Prediction: {predicted_label}</span></div>",
                    unsafe_allow_html=True
                )

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Normal Probability", f"{normal_prob:.2%}")
            with c2:
                st.metric("Pneumonia Probability", f"{pneumonia_prob:.2%}")

            if show_prob_bars:
                st.progress(float(normal_prob), text=f"Normal: {normal_prob:.2%}")
                st.progress(float(pneumonia_prob), text=f"Pneumonia: {pneumonia_prob:.2%}")

        st.markdown("---")

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")

# =========================
# Footer Note
# =========================
st.markdown(
    """
    <p class='small-note'>
    Note: This app is for model demonstration purposes only and should not be used for medical diagnosis.
    </p>
    """,
    unsafe_allow_html=True
)
