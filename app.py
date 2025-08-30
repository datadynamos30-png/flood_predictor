import streamlit as st
import os
import zipfile
import gdown
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
import folium
from streamlit_folium import st_folium

# ==============================
# CONFIG
# ==============================
# Replace with your Google Drive FILE ID
MODEL_ID = "1mF8NmMClUbKXYJoDdW3OYHhwiSknI5hk"
ZIP_PATH = "flood_model.zip"
MODEL_DIR = "fine_tuned_flood_detection_model"

# ==============================
# DOWNLOAD + UNZIP MODEL
# ==============================
@st.cache_resource
def load_flood_model():
    if not os.path.exists(MODEL_DIR):
        st.write("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

        st.write("📂 Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

    st.success("✅ Model ready to use!")
    return keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")

model = load_flood_model()

# ==============================
# STREAMLIT APP UI
# ==============================
st.set_page_config(page_title="🌊 Flood Predictor", layout="wide")

st.title("🌊 Flood Predictor")
st.write("Upload an image and select a state to predict flood risk.")

# Sidebar Upload + State Select
st.sidebar.header("Upload & Select Location")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
state = st.sidebar.selectbox("Select a State", [
    "Andhra Pradesh", "Bihar", "Karnataka", "Kerala", "Maharashtra"
])

# Map
st.subheader("🗺️ Location Map")
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
folium.Marker(location=[20.5937, 78.9629], popup="India").add_to(m)
st_map = st_folium(m, width=700, height=400)

# Prediction
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model(img_array)
    try:
        pred_val = float(preds[0][0])
    except:
        pred_val = float(preds)

    st.subheader("📊 Prediction Result")
    if pred_val > 0.5:
        st.error(f"⚠️ High Flood Risk detected in {state} (Score: {pred_val:.2f})")
    else:
        st.success(f"✅ Low Flood Risk in {state} (Score: {pred_val:.2f})")

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

