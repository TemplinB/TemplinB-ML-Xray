import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 224
MODEL_PATH = Path("CNN.keras")
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


@st.cache_resource

def load_cnn_model(model_path: str):
    """Load and cache the trained Keras model."""
    import keras
        model = keras.models.load_model("CNN.keras")
    



def preprocess_uploaded_image(uploaded_file) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an uploaded image to grayscale, resize to model input size,
    normalize to [0, 1], and return both display image and model-ready array.
    """
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized.astype("float32") / 255.0
    model_input = normalized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return image_np, model_input



def predict_image(model, model_input: np.ndarray) -> dict:
    """Run model inference and return formatted prediction results."""
    raw_pred = model.predict(model_input, verbose=0)
    pneumonia_prob = float(raw_pred[0][0])
    normal_prob = 1.0 - pneumonia_prob

    predicted_class = 1 if pneumonia_prob >= 0.5 else 0
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = pneumonia_prob if predicted_class == 1 else normal_prob

    return {
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": normal_prob,
    }


st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", layout="centered")

st.title("Chest X-Ray Pneumonia Detector")
st.write(
    "Upload a JPEG or PNG chest X-ray image to classify it as NORMAL or PNEUMONIA "
    "using your trained CNN model."
)

with st.expander("Important notes"):
    st.markdown(
        """
        - This app assumes your trained model file is named `CNN.keras`.
        - Put `app.py` and `CNN.keras` in the same folder before running the app.
        - The model was trained on **grayscale 224x224** images, so uploads are converted automatically.
        - This is a demo/inference app and should **not** be used for medical diagnosis.
        """
    )

if not MODEL_PATH.exists():
    st.error(
        "Model file `CNN.keras` was not found in the same folder as `app.py`. "
        "Move your saved model into the app folder and rerun Streamlit."
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
    help="If pneumonia probability is greater than or equal to this threshold, the app predicts PNEUMONIA.",
)

if uploaded_file is not None:
    try:
        display_image, model_input = preprocess_uploaded_image(uploaded_file)
        results = predict_image(model, model_input)

        predicted_label = (
            "PNEUMONIA" if results["pneumonia_prob"] >= threshold else "NORMAL"
        )
        confidence = (
            results["pneumonia_prob"]
            if predicted_label == "PNEUMONIA"
            else results["normal_prob"]
        )

        st.subheader("Uploaded Image")
        st.image(display_image, caption="Uploaded chest X-ray", use_container_width=True)

        st.subheader("Prediction Result")
        if predicted_label == "PNEUMONIA":
            st.error(f"Prediction: {predicted_label}")
        else:
            st.success(f"Prediction: {predicted_label}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{confidence:.2%}")
        col2.metric("Pneumonia Probability", f"{results['pneumonia_prob']:.2%}")
        col3.metric("Normal Probability", f"{results['normal_prob']:.2%}")

        st.progress(float(results["pneumonia_prob"]))
        st.caption(
            f"Pneumonia score: {results['pneumonia_prob']:.4f} | "
            f"Threshold: {threshold:.2f}"
        )

    except Exception as exc:
        st.error(f"Error while processing the uploaded image: {exc}")

