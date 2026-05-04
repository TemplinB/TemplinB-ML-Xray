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

MODEL_ACCURACY = {
    "Base Model": 0.8356,
    "Final Model": 0.9087
}


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

    .stTabs [data-baseweb="tab-highlight"] {
    background-color: #44C7B7;
    }

    .stTabs [data-baseweb="tab-border"] {
        background-color: var(--border);
    }

    /* Slider track filled portion */
    
    div[data-testid="stSlider"] div[role="progressbar"] {
        background-color: #44C7B7 !important;
    }

    /* Slider thumb */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #44C7B7 !important;
        border-color: #44C7B7 !important;
    }

    /* Radio button outer ring */
    div[data-testid="stRadio"] div[data-baseweb="radio"] div:first-child {
        border-color: #44C7B7 !important;
        background-color: transparent !important;
    }

    /* Radio button filled inner dot */
    div[data-testid="stRadio"] div[data-baseweb="radio"] div:first-child div {
        background-color: #44C7B7 !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_cnn_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def preprocess_uploaded_image(uploaded_file, image_size, color_mode="grayscale"):
    if color_mode == "rgb":
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        resized = cv2.resize(image_np, (image_size, image_size))
        normalized = resized.astype(np.float32) / 255.0

        model_input = np.expand_dims(normalized, axis=0)

    else:
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
    ["Base Model", "Final Model"],
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
        - MobileNetV2 Trial Model uses RGB 128x128 images
        - Switching models will re-run the prediction on the same uploaded image
        - This tool is for demonstration only and not medical diagnosis
        """
    )


# ---------- Shared Display Function ----------
def show_model_results(model_to_use, image_size, image_index_key, tab_label, color_mode):
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
            image_size,
            color_mode

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
            if st.button("<-", use_container_width=True):
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
            if st.button("->", use_container_width=True):
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
        col1.metric("Model Accuracy", f"{MODEL_ACCURACY[selected_model_name]:.2%}")
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

    if selected_model_name == "Base Model":
        active_model = model
        active_image_size = MAIN_IMAGE_SIZE
        active_model_file = "CNN.keras"
        active_color_mode = "grayscale"
    else:
        active_model = trial_model
        active_image_size = TRIAL_IMAGE_SIZE
        active_model_file = "CNN_MNV2.keras"
        active_color_mode = "rgb"

    st.caption(
        f"Current model: {selected_model_name} "
    )

    show_model_results(
        model_to_use=active_model,
        image_size=active_image_size,
        image_index_key="main_image_index",
        tab_label="Diagnosis",
        color_mode=active_color_mode
    )


with tab2:
    st.header("The Final Model")
    
    st.markdown("""
    &nbsp;&nbsp;&nbsp;&nbsp;The model that utilizes transfer learning is supported by the MobileNetV2 model under the ImageNet weights. This is a dataset comprised of 1.2 million images across 1000 categories. The benefit of transfer learning is this allows the model to focus on identifying the patterns in pneumonia without having to learn the basic lines, edges, and shapes of any object. For our model, the classification head has been stripped off the MobileNetV2 model meaning we are training the identification of pneumonia solely and not other everyday objects. The last 20 layers and weights of the MobileNetV2 model have been unfrozen to allow for a specific pneumonia model. 

    &nbsp;&nbsp;&nbsp;&nbsp;The head includes global average pooling to transform the output from MobileNetV2 into one long 1,280 value vector by taking the values from the 4x4 feature maps and averaging the values into a single number. The model then takes that long vector of single-value feature maps and inserts them into 128 neurons to find which combination of them detect pneumonia. An additional step or function of an L2 regularizer which takes place in the last dense layer that ensures each feature map value will remain small unless absolutely necessary. This forces the model to avoid overfitting on a few of the signals from the feature maps. Ultimately the model is trying to identify many paths to get to the pneumonia prediction. From there the model has a dropout level which shuts off half of the neurons meaning 64 patterns pass through to the final dense layer which produces a single value or probability that the image is pneumatic.

    &nbsp;&nbsp;&nbsp;&nbsp;After completion of the basic architecture, the model can then be compiled. In this step, there are two adjustments that affect the model. The first is the loss function which in this case is binary cross entropy which is the learning aspect of the model. After the image goes through the flow of the model and the probability is given, the model then takes the probability and compares it to the actual prediction either 0 or 1. It then takes a logarithmic function to flag major error as high values and slight error as minor. If the value is off by any amount, the model engages in backpropagation. This means the model, starting at the final dense layer, goes back through each layer in reverse and assigns some numerical value for the amount of blame the specific entry in a node had in making the erroneous prediction. Once the blame is assigned, the model uses Adam to calculate the specific amount the weights need to be adjusted to make the model more accurate in the next prediction. Adam in particular looks at history of weight changes as well before adjusting.

    &nbsp;&nbsp;&nbsp;&nbsp;For the training portion of the model, it takes 32 of the 5,216 images at a time and runs them through the model itself. After one batch of 32, the model calculates the error from the images and sends the next batch of 32. This repeats until all the images are seen from the total. This entire run is called an epoch. After this is finished, the model starts this process over with the same weights as the previous epoch and continues until all the fixed number of epochs are run. Another penalizer is the weight system which took the balance of the dataset into account for the loss calculations. Additionally, after each epoch a prediction check is made. This means using the current weights of the epoch, the test dataset is run through the model to calculate validation loss and accuracy which is seen in the model on the diagnosis tab.
    """, unsafe_allow_html=True)

    st.markdown("---")  # horizontal divider line
    
    st.subheader("References")
    
    st.markdown("""
    **Dataset**  
    Kermany, D. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell.  
    [Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

    **Model Architecture**  
    Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.  
    [MobileNetV2 Paper on ArXiv](https://arxiv.org/abs/1801.04381)

    **Pretrained Weights**  
    Deng, J. et al. (2009). *ImageNet: A large-scale hierarchical image database*. CVPR.  
    [ImageNet](https://www.image-net.org)

    **Source Code**  
    [GitHub Repository](https://github.com/TemplinB/TemplinB-ML-Xray)
    """, unsafe_allow_html=True)

with tab3:

    st.header("Pneumonia")
    
    st.markdown("""
    &nbsp;&nbsp;&nbsp;&nbsp;Pneumonia is an infection of the lungs that causes inflammation and fluid or pus to fill the air sacs (alveoli), making it harder for oxygen to pass into the bloodstream. On a chest X-ray, it typically appears as areas of increased opacity (white or cloudy patches) where air should normally look dark, often localized to a lobe. Clinically, it's generally safer to be overly cautious and treat a suspected case of pneumonia, even if it turns out not to be present, because untreated pneumonia can rapidly worsen, leading to serious complications like respiratory failure or sepsis, whereas the risks of short-term treatment such as antibiotics when indicated are usually much lower than the potential harm of missing a true infection.

    &nbsp;&nbsp;&nbsp;&nbsp;There are two types of pneumonia: bacterial and viral. Both infect the lungs but differ in cause, severity, and treatment. Bacterial pneumonia is commonly caused by organisms like Streptococcus pneumoniae and often develops suddenly with high fever, coughing, and more localized findings on imaging, and it is typically treated with antibiotics. Viral pneumonia is caused by viruses such as Influenza or SARS-CoV-2 and tends to appear more gradually. Symptoms include dry cough, fatigue, and diffused patterns on X-rays, and usually do not respond to antibiotics. Instead, viral cases are managed with supportive care and, in some cases, antiviral medication. Although antibiotics rarely affect viral pneumonia, they are often still prescribed as a precaution.
    """, unsafe_allow_html=True)
