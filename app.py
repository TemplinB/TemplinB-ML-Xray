import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# Try TensorFlow/Keras first
USE_TF = False
USE_SKLEARN = False

try:
    from tensorflow.keras.models import load_model
    USE_TF = True
except Exception:
    pass

try:
    import joblib
    USE_SKLEARN = True
except Exception:
    pass


st.set_page_config(
    page_title="Pneumonia Detection Dashboard",
    page_icon="🩺",
    layout="centered"
)


# -----------------------------
# Helper functions
# -----------------------------
def load_metadata():
    """
    Optional metadata file to store image size, class names, etc.
    """
    metadata_path = "model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {
        "img_size": [224, 224],
        "class_names": ["NORMAL", "PNEUMONIA"]
    }


@st.cache_resource
def load_prediction_model():
    """
    Loads either:
    - a TensorFlow/Keras model (.keras or .h5)
    - a scikit-learn model (.pkl)
    """
    # Keras model options
    keras_paths = [
        "pneumonia_model.keras",
        "pneumonia_model.h5"
    ]

    for path in keras_paths:
        if USE_TF and os.path.exists(path):
            model = load_model(path)
            return model, "keras"

    # sklearn model option
    pkl_path = "pneumonia_model.pkl"
    if USE_SKLEARN and os.path.exists(pkl_path):
        model = joblib.load(pkl_path)
        return model, "sklearn"

    raise FileNotFoundError(
        "No saved model file found. Expected one of: "
        "'pneumonia_model.keras', 'pneumonia_model.h5', or 'pneumonia_model.pkl'."
    )


def preprocess_for_keras(image: Image.Image, img_size=(224, 224)):
    """
    Preprocess image for CNN / Keras model.
    """
    image = image.convert("RGB")
    image = image.resize(img_size)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, H, W, C)
    return image_array


def preprocess_for_sklearn(image: Image.Image, img_size=(224, 224)):
    """
    Preprocess image for traditional ML model trained on flattened pixels.
    """
    image = image.convert("L")  # grayscale
    image = image.resize(img_size)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.flatten().reshape(1, -1)
    return image_array


def predict_image(model, model_type, image, metadata):
    class_names = metadata.get("class_names", ["NORMAL", "PNEUMONIA"])
    img_size = tuple(metadata.get("img_size", [224, 224]))

    if model_type == "keras":
        processed = preprocess_for_keras(image, img_size=img_size)
        pred = model.predict(processed)

        # Handle binary output shape like [[0.87]]
        if pred.shape[-1] == 1:
            pneumonia_prob = float(pred[0][0])
            normal_prob = 1.0 - pneumonia_prob
            probs = [normal_prob, pneumonia_prob]
        else:
            probs = pred[0].tolist()

    elif model_type == "sklearn":
        processed = preprocess_for_sklearn(image, img_size=img_size)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(processed)[0].tolist()
        else:
            # fallback if only predict() exists
            pred_class = int(model.predict(processed)[0])
            probs = [0.0] * len(class_names)
            probs[pred_class] = 1.0
    else:
        raise ValueError("Unsupported model type.")

    predicted_index = int(np.argmax(probs))
    predicted_label = class_names[predicted_index]
    confidence = float(probs[predicted_index])

    return predicted_label, confidence, probs


# -----------------------------
# UI
# -----------------------------
st.title("🩺 Pneumonia Detection Dashboard")
st.write(
    "Upload a chest X-ray image and the model will predict whether it shows **Pneumonia** or **Normal**."
)

metadata = load_metadata()

try:
    model, model_type = load_prediction_model()
    st.success(f"Loaded model successfully ({model_type}).")
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Predict"):
        try:
            predicted_label, confidence, probs = predict_image(
                model, model_type, image, metadata
            )

            st.subheader("Prediction Result")

            if predicted_label.upper() == "PNEUMONIA":
                st.error(f"Prediction: **{predicted_label}**")
            else:
                st.success(f"Prediction: **{predicted_label}**")

            st.write(f"Confidence: **{confidence:.2%}**")

            st.subheader("Class Probabilities")
            class_names = metadata.get("class_names", ["NORMAL", "PNEUMONIA"])
            for cls, prob in zip(class_names, probs):
                st.write(f"**{cls}**: {prob:.2%}")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
