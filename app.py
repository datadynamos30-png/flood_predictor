import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import folium
from streamlit_folium import st_folium

# -------------------------
# Model Download & Load
# -------------------------

MODEL_PATH = "fine_tuned_flood_detection_model"
FILE_ID = "1mF8NmMClUbKXYJoDdW3OYHhwiSknI5hk"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Load the model once
@st.cache_resource
def load_flood_model(path):
    return load_model(path)

model = load_flood_model(MODEL_PATH)

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Flood Predictor 🌊", layout="wide")
st.title("Flood Predictor")
st.write("Upload an image, and this model will predict flood risk. A map shows flood results interactively.")

# Image Upload & Prediction
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = image.load_img(uploaded, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)[0][0]
    label = "🚨 Flood Likely" if pred > 0.5 else "✅ No Flood"
    st.image(uploaded, caption=f"Prediction: {label}", use_column_width=True)

    # Map Visualization
    st.subheader("Flood Map Visualization")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="cartodbpositron")
    folium.Marker(
        location=[20.5937, 78.9629],
        popup=label,
        icon=folium.Icon(color="red" if "Flood" in label else "green")
    ).add_to(m)
    st_folium(m, width=700, height=500)
else:
    st.info("Awaiting image upload to start prediction…")

