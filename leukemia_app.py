import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
import requests

st.set_page_config(
    page_title="Leukemia Classification Dashboard",
    page_icon="🩸",
    layout="wide"
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['all', 'hem']

# S3 Configuration
MODEL_URL = "https://efficientnet-trained-model-bucket.s3.eu-north-1.amazonaws.com/efficientnet-trained.h5"
LOCAL_MODEL_PATH = "efficientnet-trained.h5"


@st.cache_resource
def load_model_from_s3(url, local_path):
    """
    Downloads the model from S3 if not present, then loads it into TensorFlow.
    Uses st.cache_resource to keep the model in memory.
    """
    # 1. Download the file if it doesn't exist locally
    if not os.path.exists(local_path):
        try:
            with st.spinner(f'Downloading model from S3... this may take a minute.'):
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise error for bad status codes

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model from S3: {e}")
            return None

    # 2. Load the model using Keras
    try:
        model = tf.keras.models.load_model(local_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from file: {e}")
        return None


def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.sidebar.title("🩸 C-NMC Classification")
st.sidebar.info("Acute Lymphoblastic Leukemia Detection System")

app_mode = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Evaluation Metrics", "Live Prediction"]
)

if app_mode == "Project Overview":
    st.title("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### About the Model
        This application utilizes a Deep Learning model based on **EfficientNetB3** to classify Leukemia from microscopic blood cell images.

        **Model Architecture:**
        * **Base:** EfficientNetB3 (Pre-trained on ImageNet)
        * **Custom Head:** * Batch Normalization
            * Dense (256 units, ReLU, L1/L2 Regularization)
            * Dropout (0.45)
            * Output Dense (2 units, Softmax)

        **Optimizer:** Adamax (LR=0.001)
        """)

    with col2:
        st.markdown("### Dataset Distribution (Test Set)")
        data = {
            'Class': ['ALL (Leukemia)', 'HEM (Normal)'],
            'Count': [1091, 509]
        }
        df_dist = pd.DataFrame(data)

        fig = px.pie(df_dist, values='Count', names='Class',
                     title='Class Distribution in Test Data',
                     color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Evaluation Metrics":
    st.title("Model Evaluation Metrics")

    st.markdown("### Training History")

    # Hardcoded metrics for visualization purposes
    epochs = list(range(1, 20))
    train_acc = [82.485, 88.756, 91.088, 92.428, 93.648, 94.184, 94.666, 95.417, 97.293,
                 97.856, 98.258, 98.459, 99.049, 98.847, 99.317, 99.558, 99.223, 99.571, 99.611]
    val_acc = [70.169, 73.671, 88.993, 86.116, 88.430, 87.805, 93.183, 91.307, 93.684,
               95.122, 94.371, 94.809, 95.810, 95.872, 95.935, 96.185, 96.248, 95.872, 96.060]

    train_loss = [5.135, 2.546, 1.479, 0.888, 0.560, 0.392, 0.296, 0.237, 0.185,
                  0.164, 0.145, 0.132, 0.118, 0.118, 0.107, 0.099, 0.104, 0.092, 0.090]
    val_loss = [3.637, 2.381, 1.149, 0.864, 0.546, 0.437, 0.296, 0.303, 0.253,
                0.223, 0.229, 0.221, 0.201, 0.194, 0.193, 0.180, 0.181, 0.183, 0.180]

    tab1, tab2, tab3 = st.tabs(["Accuracy Plot", "Loss Plot", "Confusion Matrix"])

    with tab1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Train Accuracy'))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'))
        fig_acc.update_layout(title="Accuracy over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig_acc, use_container_width=True)

    with tab2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train Loss'))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'))
        fig_loss.update_layout(title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

    with tab3:
        cm_data = [[1068, 23], [49, 460]]

        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Test Data)')
        st.pyplot(fig_cm)

        st.metric(label="Overall Test Accuracy", value="95.5%")

elif app_mode == "Live Prediction":
    st.title("Live Model Prediction")

    # Load Model from S3 using the cached function
    model = load_model_from_s3(MODEL_URL, LOCAL_MODEL_PATH)

    if model:
        st.markdown("### Upload Cell Image")
        uploaded_file = st.file_uploader("Choose a microscopic image...", type=["jpg", "png", "jpeg", "bmp"])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

            with col2:
                st.markdown("### Analysis Results")
                if st.button("Predict Classification"):
                    with st.spinner('Analyzing cell structure...'):
                        processed_img = preprocess_image(image)

                        prediction = model.predict(processed_img)

                        predicted_class_idx = np.argmax(prediction)
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        confidence = np.max(prediction) * 100

                        if predicted_class == 'all':
                            st.error(f"**Prediction: ALL (Leukemia)**")
                            st.markdown("Result: **Acute Lymphoblastic Leukemia** Detected")
                        else:
                            st.success(f"**Prediction: HEM (Normal)**")
                            st.markdown("Result: **Normal / Hemaglobin** Detected")

                        st.progress(int(confidence))
                        st.write(f"Model Confidence: **{confidence:.2f}%**")

                        with st.expander("See probability details"):
                            prob_df = pd.DataFrame(prediction, columns=[c.upper() for c in CLASS_NAMES])
                            st.dataframe(prob_df.style.format("{:.2%}"))
    else:
        st.error("Model could not be loaded. Please check the S3 URL or your internet connection.")

st.markdown("---")
st.markdown("Created based on Leukemia Classification Jupyter Notebook | Source: S3 Bucket")
