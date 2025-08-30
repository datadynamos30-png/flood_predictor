import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import folium
from streamlit_folium import st_folium

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "fine_tuned_flood_detection_model.h5"  # make sure this is in your repo
model = load_model(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🌊 Flood Predictor", layout="wide")

st.title("🌊 Flood Predictor")
st.write("Upload a satellite or region image to check flood prediction, and view it on the map.")

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    result = "🚨 Flood Likely" if prediction[0][0] > 0.5 else "✅ No Flood Detected"

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.subheader(f"Prediction Result: {result}")

    # ----------------------------
    # Folium Map
    # ----------------------------
    st.write("### 🌍 Map Visualization")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Centered on India
    folium.Marker(
        [20.5937, 78.9629],
        popup=result,
        icon=folium.Icon(color="red" if "Flood" in result else "green")
    ).add_to(m)

    st_folium(m, width=700, height=500)
else:
    st.info("⬆️ Please upload an image to start flood prediction.")

