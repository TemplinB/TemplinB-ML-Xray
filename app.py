from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 64
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
MODEL_PATH = APP_DIR / "CNN.keras"
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", layout="centered")


@st.cache_resource
def load_cnn_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def preprocess_uploaded_image(uploaded_file):
    """
    Convert uploaded image to grayscale, resize to 64x64,
    normalize to [0, 1], and add channel + batch dimensions.
    """
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    model_input = np.expand_dims(normalized, axis=-1)   # (64, 64, 1)
    model_input = np.expand_dims(model_input, axis=0)   # (1, 64, 64, 1)

    return image_np, model_input


def predict_image(model, model_input: np.ndarray, threshold: float = 0.5):
    raw_pred = model.predict(model_input, verbose=0)

    # output is pneumonia probability
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


st.title("Chest X-Ray Pneumonia Model")
st.write(
    "Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA."
)

with st.expander("Important notes"):
    st.markdown(
        """
        - Images are automatically converted to grayscale and resized to 64x64
        - This is only a demo app and not for medical diagnosis
        """
    )

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at: `{MODEL_PATH}`\n\n"
    )
    st.stop()

try:
    model = load_cnn_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

threshold = st.slider(
    "Prediction threshold for pneumonia",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

if uploaded_file is not None:
    try:
        display_image, model_input = preprocess_uploaded_image(uploaded_file)
        results = predict_image(model, model_input, threshold=threshold)

        st.subheader("Uploaded Image")
        st.image(display_image, caption="Uploaded chest X-ray", use_container_width=True)

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
        st.error(f"Error while processing the uploaded image: {exc}")
