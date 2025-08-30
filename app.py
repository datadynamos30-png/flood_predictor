import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.layers import TFSMLayer
import numpy as np
from PIL import Image
import gdown, zipfile, os, pathlib
import folium
from streamlit_folium import st_folium

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="🌊 Flood Predictor", layout="wide")

MODEL_ID = "1zoECslC9YqUW7DLDw2uFKyR3dIItjAbf"  # Google Drive zip file ID
MODEL_ZIP = "flood_model.zip"
MODEL_DIR = "flood_model"

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_flood_model():
    if not os.path.exists(MODEL_ZIP):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_ZIP, quiet=False)

    if not os.path.exists(MODEL_DIR):
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    keras_files = list(pathlib.Path(MODEL_DIR).glob("*.keras"))
    h5_files = list(pathlib.Path(MODEL_DIR).glob("*.h5"))
    saved_model_dirs = [p for p in pathlib.Path(MODEL_DIR).iterdir() if p.is_dir()]

    if keras_files:
        return load_model(str(keras_files[0]))
    elif h5_files:
        return load_model(str(h5_files[0]))
    elif saved_model_dirs:
        return TFSMLayer(str(saved_model_dirs[0]), call_endpoint="serving_default")
    else:
        raise ValueError("❌ No supported model format found in extracted zip.")

model = load_flood_model()

# ----------------------------
# APP HEADER
# ----------------------------
st.title("🌊 Flood Predictor")
st.markdown("Upload a **satellite image** or select a **state** to predict flood risk. \
            View predictions on an **interactive OSM map** 🗺️")

# ----------------------------
# INPUT OPTIONS
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_img = st.file_uploader("📤 Upload a satellite image", type=["jpg", "jpeg", "png"])

with col2:
    state = st.selectbox("🌍 Or choose a state:", 
                         ["None", "California", "Texas", "Florida", "Bihar", "Assam", "Kerala", "Odisha"])

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_flood(image: Image.Image):
    img = image.resize((224, 224))  # adjust to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model(img_array, training=False) if isinstance(model, TFSMLayer) else model.predict(img_array)
    prob = float(preds[0][0]) if preds.ndim == 2 else float(preds[0])
    return prob

# ----------------------------
# HANDLE PREDICTION
# ----------------------------
if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing flood risk..."):
        prob = predict_flood(image)

    st.subheader(f"🌊 Flood Risk Probability: **{prob:.2%}**")

elif state != "None":
    st.subheader(f"📍 Selected State: {state}")
    st.info("Flood prediction for states is placeholder — attach state-level risk data later.")

# ----------------------------
# MAP VISUALIZATION
# ----------------------------
st.subheader("🗺️ Flood Risk Map")

# Example: center India (lat=20.59, lon=78.96)
m = folium.Map(location=[20.59, 78.96], zoom_start=4, tiles="cartodb positron")

# Add a marker if state is chosen
state_coords = {
    "California": [36.7783, -119.4179],
    "Texas": [31.9686, -99.9018],
    "Florida": [27.9944, -81.7603],
    "Bihar": [25.0961, 85.3131],
    "Assam": [26.2006, 92.9376],
    "Kerala": [10.8505, 76.2711],
    "Odisha": [20.9517, 85.0985]
}

if state in state_coords:
    folium.Marker(state_coords[state], tooltip=f"Flood Risk: {state}").add_to(m)

st_map = st_folium(m, width=700, height=500)

