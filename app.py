from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 64
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "CNN.keras"
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", layout="centered")

st.markdown(
    """
    <style>
    :root {
        --bg: #071A2D;
        --panel: #0B2742;
        --panel-light: #123A5C;
        --accent: #38E8D0;
        --accent-2: #58A6FF;
        --text: #F4FAFF;
        --muted: #B7C9D9;
        --danger: #FF5C7A;
        --success: #43E88D;
    }

    /* -------- REMOVE STREAMLIT UI -------- */
    header[data-testid="stHeader"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        display: none;
    }

    /* Remove top padding gap */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 3rem;
    }

    section.main > div {
        padding-top: 0rem;
    }

    /* -------- FORCE DARK THEME -------- */
    html, body, [class*="css"] {
        color: var(--text) !important;
    }

    [data-testid="stAppViewContainer"] {
        background: none !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56,232,208,0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(88,166,255,0.16), transparent 28%),
            linear-gradient(135deg, #061525 0%, #071A2D 50%, #081E33 100%) !important;
    }

    /* -------- HEADER -------- */
    .header-banner {
        background:
            linear-gradient(90deg, rgba(56,232,208,0.95), rgba(88,166,255,0.95));
        padding: 18px 24px;
        border-radius: 0px 0px 22px 22px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }

    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 36px;
        font-weight: 800;
    }

    .header-banner p {
        color: rgba(255,255,255,0.9);
        margin: 6px 0 0 0;
    }

    /* -------- SIDEBAR -------- */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #061525 0%, #0B2742 100%);
        border-right: 1px solid rgba(56,232,208,0.25);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2 {
        color: var(--accent);
    }

    /* -------- TABS -------- */
    button[data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.06);
        color: var(--muted);
        border-radius: 14px 14px 0 0;
        padding: 10px 18px;
        margin-right: 6px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
    }

    button[data-baseweb="tab"]:hover {
        background-color: rgba(56,232,208,0.14);
        color: white;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        border: none;
    }

    /* -------- BUTTONS -------- */
    div.stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        border-radius: 14px;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(56,232,208,0.25);
        transition: 0.2s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(88,166,255,0.35);
    }

    /* -------- FILE UPLOADER -------- */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.06);
        border: 2px dashed var(--accent);
        border-radius: 18px;
    }

    /* -------- METRICS -------- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(56,232,208,0.35);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }

    div[data-testid="stMetricValue"] {
        color: var(--accent);
        font-weight: 800;
    }

    /* -------- PROGRESS -------- */
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }

    /* -------- ALERTS -------- */
    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    /* -------- EXPANDERS -------- */
    details {
        background-color: rgba(255,255,255,0.06);
        border: 1px solid rgba(56,232,208,0.25);
        border-radius: 16px;
    }

    summary {
        color: var(--accent);
        font-weight: 700;
    }

    /* -------- IMAGES -------- */
    img {
        border-radius: 18px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.32);
    }

    /* -------- TEXT -------- */
    h1, h2, h3, h4, h5, h6, p, label {
        color: var(--text);
    }

    .stCaption {
        color: var(--muted);
    }

    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_cnn_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    model_input = np.expand_dims(normalized, axis=-1)
    model_input = np.expand_dims(model_input, axis=0)

    return image_np, model_input


def predict_image(model, model_input: np.ndarray, threshold: float = 0.5):
    raw_pred = model.predict(model_input, verbose=0)
    pneumonia_prob = float(raw_pred[0][0])
    normal_prob = 1.0 - pneumonia_prob

    predicted_class = 1 if pneumonia_prob >= threshold else 0
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = pneumonia_prob if predicted_class == 1 else normal_prob

    return {
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": normal_prob,
    }

# ---------- Header ----------
st.markdown(
    """
    <div class="header-banner">
        <h1>Chest X-Ray Pneumonia Detector</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Load Model ----------
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_cnn_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("Chest X-Ray Pneumonia Detector")

uploaded_files = st.sidebar.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="If pneumonia probability is greater than or equal to this threshold, the model predicts PNEUMONIA.",
)

with st.sidebar.expander("Important notes"):
    st.markdown(
        """
        - Upload a chest X-ray image in JPG, JPEG, or PNG format
        - The image is automatically converted to grayscale and resized to 64x64
        - The threshold controls when the model calls an image pneumonia
        - This tool is for demonstration only and not medical diagnosis
        """
    )

# ---------- Tabs ----------
tab1, tab2, tab3, tab4= st.tabs(
    ["Run Model", "About the Model", "About Pneumonia", "Trial"]
)

with tab1:
    st.write("Testing Tab.")

    if not uploaded_files:
        st.info("Use the sidebar to upload one or more X-ray images.")
    else:
        if "image_index" not in st.session_state:
            st.session_state.image_index = 0

        if st.session_state.image_index >= len(uploaded_files):
            st.session_state.image_index = 0

        uploaded_file = uploaded_files[st.session_state.image_index]

        try:
            display_image, model_input = preprocess_uploaded_image(uploaded_file)
            results = predict_image(model, model_input, threshold)

            st.markdown(
                f"<h4 style='text-align:center;'>Image "
                f"{st.session_state.image_index + 1} of {len(uploaded_files)}</h4>",
                unsafe_allow_html=True
            )

            col_left, col_img, col_right = st.columns([1, 8, 1])

            with col_left:
                st.write("")
                st.write("")
                st.write("")
                if st.button("<-", use_container_width=True):
                    st.session_state.image_index -= 1
                    if st.session_state.image_index < 0:
                        st.session_state.image_index = len(uploaded_files) - 1
                    st.rerun()

            with col_img:
                st.image(
                    display_image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            with col_right:
                st.write("")
                st.write("")
                st.write("")
                if st.button("->", use_container_width=True):
                    st.session_state.image_index += 1
                    if st.session_state.image_index >= len(uploaded_files):
                        st.session_state.image_index = 0
                    st.rerun()

            st.subheader("Prediction Result")

            if results["predicted_label"] == "PNEUMONIA":
                st.error(f"Prediction: {results['predicted_label']}")
            else:
                st.success(f"Prediction: {results['predicted_label']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{results['confidence']:.2%}")
            col2.metric("Pneumonia Probability", f"{results['pneumonia_prob']:.2%}")
            col3.metric("Normal Probability", f"{results['normal_prob']:.2%}")

            st.progress(float(results["pneumonia_prob"]))
            st.caption(
                f"Pneumonia score: {results['pneumonia_prob']:.4f} | "
                f"Threshold: {threshold:.2f}"
            )

        except Exception as exc:
            st.error(f"Error processing {uploaded_file.name}: {exc}")

with tab2:
    st.header("About the Model")
    st.write("Add model description.")

with tab3:
    st.header("About Pneumonia")
    st.write("Add pneumonia information.")

with tab4:
    st.write("Testing Tab.")

    if not uploaded_files:
        st.info("Use the sidebar to upload one or more X-ray images.")
    else:
        if "image_index" not in st.session_state:
            st.session_state.image_index = 0

        if st.session_state.image_index >= len(uploaded_files):
            st.session_state.image_index = 0

        uploaded_file = uploaded_files[st.session_state.image_index]

        try:
            display_image, model_input = preprocess_uploaded_image(uploaded_file)
            results = predict_image(model, model_input, threshold)

            st.markdown(
                f"<h4 style='text-align:center;'>Image "
                f"{st.session_state.image_index + 1} of {len(uploaded_files)}</h4>",
                unsafe_allow_html=True
            )

            col_left, col_img, col_right = st.columns([1, 8, 1])

            with col_left:
                st.write("")
                st.write("")
                st.write("")
                if st.button("<-", use_container_width=True):
                    st.session_state.image_index -= 1
                    if st.session_state.image_index < 0:
                        st.session_state.image_index = len(uploaded_files) - 1
                    st.rerun()

            with col_img:
                st.image(
                    display_image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            with col_right:
                st.write("")
                st.write("")
                st.write("")
                if st.button("->", use_container_width=True):
                    st.session_state.image_index += 1
                    if st.session_state.image_index >= len(uploaded_files):
                        st.session_state.image_index = 0
                    st.rerun()

            st.subheader("Prediction Result")

            if results["predicted_label"] == "PNEUMONIA":
                st.error(f"Prediction: {results['predicted_label']}")
            else:
                st.success(f"Prediction: {results['predicted_label']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{results['confidence']:.2%}")
            col2.metric("Pneumonia Probability", f"{results['pneumonia_prob']:.2%}")
            col3.metric("Normal Probability", f"{results['normal_prob']:.2%}")

            st.progress(float(results["pneumonia_prob"]))
            st.caption(
                f"Pneumonia score: {results['pneumonia_prob']:.4f} | "
                f"Threshold: {threshold:.2f}"
            )

        except Exception as exc:
            st.error(f"Error processing {uploaded_file.name}: {exc}")


