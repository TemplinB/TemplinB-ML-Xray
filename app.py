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

# ---------- Custom App Styling ----------
st.markdown(
    """
    <style>
    :root {
        --bg: #061A1F;
        --panel: #0B2A30;
        --panel-soft: #123B42;
        --primary: #7FE8D7;
        --primary-dark: #38BFAE;
        --primary-glow: rgba(127, 232, 215, 0.35);
        --secondary: #9ED8FF;
        --text: #F4FFFD;
        --muted: #B7D7D3;
        --danger: #FF6B8A;
        --success: #7FE8A5;
    }

    /* Hide Streamlit default UI */
    header[data-testid="stHeader"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        display: none;
    }

    /* Remove top spacing */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 3rem;
    }

    section.main > div {
        padding-top: 0rem;
    }

    /* Force custom theme */
    html, body, [class*="css"] {
        color: var(--text) !important;
    }

    [data-testid="stAppViewContainer"] {
        background: none !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(127,232,215,0.24), transparent 30%),
            radial-gradient(circle at bottom right, rgba(158,216,255,0.14), transparent 34%),
            linear-gradient(135deg, #041316 0%, #061A1F 50%, #09262B 100%) !important;
        color: var(--text);
    }

    /* Header banner */
    .header-banner {
        background:
            linear-gradient(90deg, #7FE8D7 0%, #38BFAE 50%, #9ED8FF 100%);
        padding: 18px 24px;
        border-radius: 0px 0px 24px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.35);
    }

    .header-banner h1 {
        color: #032326;
        margin: 0;
        font-size: 36px;
        font-weight: 900;
        letter-spacing: 0.4px;
    }

    .header-banner p {
        color: #06393C;
        margin: 6px 0 0 0;
        font-size: 16px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #041316 0%, #0B2A30 100%);
        border-right: 1px solid rgba(127,232,215,0.3);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--primary);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.06);
        color: var(--muted);
        border-radius: 14px 14px 0 0;
        padding: 10px 18px;
        margin-right: 6px;
        font-weight: 800;
        border: 1px solid rgba(127,232,215,0.18);
    }

    button[data-baseweb="tab"]:hover {
        background-color: rgba(127,232,215,0.14);
        color: var(--text);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: #032326;
        border: none;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: #032326;
        border: none;
        border-radius: 14px;
        font-weight: 900;
        box-shadow: 0 8px 22px var(--primary-glow);
        transition: 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(127,232,215,0.45);
        color: #032326;
        border: none;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.06);
        border: 2px dashed var(--primary);
        border-radius: 18px;
        padding: 16px;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(127,232,215,0.1);
        border-color: var(--secondary);
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: var(--text);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(127,232,215,0.35);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.24);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary);
        font-weight: 900;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(127,232,215,0.2);
    }

    /* Expanders */
    details {
        background-color: rgba(255,255,255,0.06);
        border: 1px solid rgba(127,232,215,0.25);
        border-radius: 16px;
        padding: 8px;
    }

    summary {
        color: var(--primary);
        font-weight: 800;
    }

    /* Images */
    img {
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.34);
    }

    /* Text */
    h1, h2, h3, h4, h5, h6, p, label {
        color: var(--text);
    }

    .stCaption {
        color: var(--muted);
    }

    /* Slider label */
    .stSlider label {
        color: var(--text);
        font-weight: 700;
    }

    hr {
        border-color: rgba(127,232,215,0.25);
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


