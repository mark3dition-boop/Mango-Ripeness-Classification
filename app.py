import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# ==========================
# Load Model
# ==========================
with open("aug.pkl", "rb") as f:
    model, scaler = pickle.load(f)


# ==========================
# Segmentation
# ==========================
def segment_mango(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array([10, 30, 30]),
        np.array([95, 255, 255])
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([0, 30, 30]),
        np.array([10, 255, 255])
    )

    mask = cv2.bitwise_or(mask1, mask2)

    return mask


# ==========================
# Feature Extraction
# ==========================
def extract_features(image, mask):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    mask_pixels = mask > 0

    h_mean = np.mean(hsv[:, :, 0][mask_pixels])
    s_mean = np.mean(hsv[:, :, 1][mask_pixels])
    v_mean = np.mean(hsv[:, :, 2][mask_pixels])

    return np.array([[h_mean, s_mean, v_mean]])


# ==========================
# Prediction
# ==========================
def predict(image):

    mask = segment_mango(image)

    features = extract_features(image, mask)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    return prediction, mask


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Prediksi Kematangan Mangga", layout="centered")

st.title("🥭 Prediksi Kematangan Mangga")

st.write(
    "Upload gambar mangga kemudian sistem akan memprediksi tingkat kematangannya menggunakan model SVM."
)

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    prediction, mask = predict(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        st.image(mask, caption="Segmentation Mask", use_container_width=True)

    st.success(f"### Hasil Prediksi: **{prediction}**")