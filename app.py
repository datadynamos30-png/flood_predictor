
    import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import folium
from streamlit_folium import st_folium
import geopandas as gpd

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fine_tuned_flood_detection_model.keras")

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Flood Predictor", page_icon="🌊", layout="wide")

st.title("🌊 Flood Predictor")
st.write("Upload an image and select a state to predict flood risk, visualized on a map.")

# Sidebar
state = st.sidebar.selectbox(
    "Select a State",
    ["Andhra Pradesh", "Bihar", "Kerala", "Maharashtra", "Odisha", "Uttar Pradesh"]
)
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    label = "🌊 Flooding" if preds[0][0] > 0.5 else "✅ No Flooding"

    st.subheader("Prediction Result")
    st.success(f"**{label}** (Confidence: {preds[0][0]:.2f})")

    # Map visualization
    st.subheader("📍 Location Map")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
    folium.Marker(
        location=[20.5937, 78.9629], 
        popup=f"{state} - {label}", 
        icon=folium.Icon(color="blue" if "No" in label else "red", icon="info-sign")
    ).add_to(m)
    st_folium(m, width=700, height=500)

