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
st.markdown(
    """
    <style>
    .header-banner {
        background-color: #4B9CD3;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }

    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 32px;
    }

    .header-banner p {
        color: white;
        margin: 6px 0 0 0;
        font-size: 16px;
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
        <p>Upload an X-ray to classify as NORMAL or PNEUMONIA using a CNN model</p>
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
st.sidebar.header("Model Inputs")

uploaded_file = st.sidebar.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
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
tab1, tab2, tab3 = st.tabs(
    ["Run Model", "About the Model", "About Pneumonia"]
)

with tab1:
    st.header("Run the Pneumonia Detection Model")

    if uploaded_file is None:
        st.info("Use the sidebar to upload an X-ray image and adjust the prediction threshold.")
    else:
        try:
            
            display_image, model_input = preprocess_uploaded_image(uploaded_file)
            results = predict_image(model, model_input, threshold)

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

            st.subheader("Uploaded Image")
            st.image(
                display_image,
                caption="Uploaded chest X-ray",
                use_container_width=True
            )


        except Exception as exc:
            st.error(f"Error while processing the uploaded image: {exc}")

with tab2:
    st.header("About the Model")
    st.write("Add your model description here.")

with tab3:
    st.header("About Pneumonia")
    st.write("Add your pneumonia information here.")
