from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MAIN_IMAGE_SIZE = 64
TRIAL_IMAGE_SIZE = 128

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "CNN.keras"
TRIAL_MODEL_PATH = APP_DIR / "CNN_MNV2.keras"

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", layout="centered")

# ---------- Custom App Styling ----------
st.markdown(
    """
    <style>
    :root {
        --primary: #7FE8D7;
        --primary-dark: #44C7B7;
        --bg: #F6FBFA;
        --panel: #FFFFFF;
        --sidebar-bg: #EFFFFC;
        --text: #102A2F;
        --muted: #5E7477;
        --border: #D7EFEB;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        display: none;
    }

    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        display: block !important;
        min-width: 300px !important;
        max-width: 300px !important;
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
        box-shadow: inset -2px 0px 8px rgba(16, 42, 47, 0.05);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--primary-dark) !important;
    }

    section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF;
        border: 2px dashed var(--primary-dark);
        border-radius: 14px;
    }

    .block-container {
        padding-top: 0rem;
        padding-bottom: 3rem;
    }

    section.main > div {
        padding-top: 0rem;
    }

    .stApp {
        background: var(--bg) !important;
        color: var(--text);
    }

    .header-banner {
        background: var(--primary);
        padding: 18px 24px;
        border-radius: 0px 0px 18px 18px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 6px 18px rgba(16, 42, 47, 0.16);
    }

    .header-banner h1 {
        color: var(--text);
        margin: 0;
        font-size: 34px;
        font-weight: 800;
    }

    .header-banner p {
        color: #17383E;
        margin: 6px 0 0 0;
        font-size: 15px;
    }

    button[data-baseweb="tab"] {
        background-color: #FFFFFF;
        color: var(--muted);
        border-radius: 12px 12px 0 0;
        padding: 10px 18px;
        margin-right: 5px;
        font-weight: 700;
        border: 1px solid var(--border);
    }

    button[data-baseweb="tab"]:hover {
        background-color: #EFFFFC;
        color: var(--text);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary);
        color: var(--text);
        border: 1px solid var(--primary);
    }

    div.stButton > button {
        background: var(--primary);
        color: var(--text);
        border: 1px solid var(--primary-dark);
        border-radius: 12px;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(68, 199, 183, 0.25);
    }

    div.stButton > button:hover {
        background: var(--primary-dark);
        color: white;
        border: 1px solid var(--primary-dark);
    }

    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF;
        border: 2px dashed var(--primary-dark);
        border-radius: 14px;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: var(--text);
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 14px rgba(16, 42, 47, 0.08);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-dark);
        font-weight: 800;
    }

    div[data-testid="stProgress"] > div > div > div {
        background-color: var(--primary-dark);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    details {
        background-color: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 6px;
    }

    summary {
        color: var(--text);
        font-weight: 700;
    }

    img {
        border-radius: 14px;
        box-shadow: 0 8px 20px rgba(16, 42, 47, 0.14);
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


def preprocess_uploaded_image(uploaded_file, image_size):
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    resized = cv2.resize(image_np, (image_size, image_size))
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


# ---------- Load Models ----------
if not MODEL_PATH.exists():
    st.error(f"Main model file not found: {MODEL_PATH}")
    st.stop()

if not TRIAL_MODEL_PATH.exists():
    st.error(f"Trial model file not found: {TRIAL_MODEL_PATH}")
    st.stop()

try:
    model = load_cnn_model(str(MODEL_PATH))
    trial_model = load_cnn_model(str(TRIAL_MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load models: {exc}")
    st.stop()


# ---------- Sidebar ----------
st.sidebar.header("Chest X-Ray Pneumonia Detector")

uploaded_files = st.sidebar.file_uploader(
    "Upload chest X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

selected_model_name = st.sidebar.radio(
    "Choose model",
    ["CNN Model", "MobileNetV2 Trial Model"],
    index=0,
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
        - Upload chest X-ray images in JPG, JPEG, or PNG format
        - CNN Model uses grayscale 64x64 images
        - MobileNetV2 Trial Model uses grayscale 128x128 images
        - Switching models will re-run the prediction on the same uploaded image
        - This tool is for demonstration only and not medical diagnosis
        """
    )


# ---------- Shared Display Function ----------
def show_model_results(model_to_use, image_size, image_index_key, tab_label):
    if not uploaded_files:
        st.info("Use the sidebar to upload one or more X-ray images.")
        return

    if image_index_key not in st.session_state:
        st.session_state[image_index_key] = 0

    if st.session_state[image_index_key] >= len(uploaded_files):
        st.session_state[image_index_key] = 0

    uploaded_file = uploaded_files[st.session_state[image_index_key]]

    try:
        display_image, model_input = preprocess_uploaded_image(
            uploaded_file,
            image_size
        )

        results = predict_image(model_to_use, model_input, threshold)

        st.markdown(
            f"<h4 style='text-align:center;'>Image "
            f"{st.session_state[image_index_key] + 1} of {len(uploaded_files)}</h4>",
            unsafe_allow_html=True
        )

        col_left, col_img, col_right = st.columns([1, 8, 1])

        with col_left:
            st.write("")
            st.write("")
            st.write("")
            if st.button(f"<- {tab_label}", use_container_width=True):
                st.session_state[image_index_key] -= 1
                if st.session_state[image_index_key] < 0:
                    st.session_state[image_index_key] = len(uploaded_files) - 1
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
            if st.button(f"{tab_label} ->", use_container_width=True):
                st.session_state[image_index_key] += 1
                if st.session_state[image_index_key] >= len(uploaded_files):
                    st.session_state[image_index_key] = 0
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


# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(
    ["Diagnosis", "About the Model", "About Pneumonia"]
)

with tab1:

    if selected_model_name == "CNN Model":
        active_model = model
        active_image_size = MAIN_IMAGE_SIZE
        active_model_file = "CNN.keras"
    else:
        active_model = trial_model
        active_image_size = TRIAL_IMAGE_SIZE
        active_model_file = "CNN_MNV2.keras"

    st.caption(
        f"Current model: {active_model_file} | "
        f"Input size: {active_image_size}x{active_image_size} grayscale"
    )

    show_model_results(
        model_to_use=active_model,
        image_size=active_image_size,
        image_index_key="main_image_index",
        tab_label="Diagnosis"
    )

with tab2:
    st.write("Add model description.")

with tab3:
    st.write("Pneumonia is an infection of the lungs that causes inflammation and fluid or pus to fill the air sacs (alveoli), making it harder for oxygen to pass into the bloodstream. On a chest X-ray, it typically appears as areas of increased opacity (white or cloudy patches) where air should normally look dark, often localized to a lobe. Clinically, it’s generally safer to be overly cautious and treat a suspected case of pneumonia, even if it turns out not to be present, because untreated pneumonia can rapidly worsen, leading to serious complications like respiratory failure or sepsis, whereas the risks of short-term treatment such as antibiotics when indicated are usually much lower than the potential harm of missing a true infection.")
    st.write("There are two types of pneumonia: bacterial and viral. Both infect the lungs but differ in cause, severity, and treatment. Bacterial pneumonia is commonly caused by organisms like Streptococcus pneumoniae and often develops suddenly with high fever, coughing, and more localized findings on imaging, and it is typically treated with antibiotics. Viral pneumonia is caused by viruses such as Influenza or SARS-CoV-2 and tends to appear more gradually. Symptoms include dry cough, fatigue, and diffused patterns on X-rays, and usually do not respond to antibiotics. Instead, viral cases are managed with supportive care and, in some cases, antivrial medication. Despite the antibiotics rarely affecting viral pnuemonia, they are often still prescribed as a precaution.")
