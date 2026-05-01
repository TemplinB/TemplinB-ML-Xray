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

# ---------- Custom Header Styling ----------




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


