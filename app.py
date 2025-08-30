import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import folium
from folium.plugins import HeatMap
from PIL import Image as PILImage
from io import BytesIO
import geopandas as gpd

# Load pre-trained model
model_path = 'fine_tuned_flood_detection_model'  # Update the path if needed
model = tf.keras.models.load_model(model_path)

# Define states for the selection
states = ["Assam", "Kerala", "Bihar"]

# Function to load and preprocess the image for prediction
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] if it was normalized during training
    return img_array

# Function to make flood prediction
def predict_flood(img_array):
    prediction = model.predict(img_array)
    return prediction

# Function to display OSM layers on the map
def display_osm_map(state):
    # Use OSM to create a map and add layers for roads, hospitals, and schools
    if state == "Assam":
        coordinates = [26.1, 92.5]  # Assam center coordinates
    elif state == "Kerala":
        coordinates = [10.5, 76.8]  # Kerala center coordinates
    else:
        coordinates = [25.0, 85.0]  # Bihar center coordinates

    # Map setup
    m = folium.Map(location=coordinates, zoom_start=6)
    
    # Example OSM layers (You can fetch these from OSM APIs or your dataset)
    HeatMap([[coordinates[0], coordinates[1]]]).add_to(m)
    
    folium.Marker(coordinates, popup=f"{state} Hospital").add_to(m)
    folium.Marker([coordinates[0] + 0.1, coordinates[1] + 0.1], popup=f"{state} School").add_to(m)
    
    return m

# Streamlit user interface
def main():
    st.title("Flood Predictor")
    st.write("### Predict flood status based on uploaded image")

    # User selects state
    state = st.selectbox("Select a State", states)

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Process the uploaded image for prediction
        img_path = f"/content/{uploaded_image.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        img_array = prepare_image(img_path)
        
        # Display the image
        st.image(img_path, caption="Uploaded Image", use_column_width=True)
        
        # Predict flood status
        prediction = predict_flood(img_array)
        
        if prediction[0] > 0.5:
            st.write("🚨 Flood detected!")
        else:
            st.write("✅ No flood detected!")
        
        # Display the OSM map with flood information
        st.subheader(f"OSM Map for {state} with Flood Data")
        osm_map = display_osm_map(state)
        folium_static(osm_map)
        
        # Add animation effects
        st.write("### Flood Impact Animation")
        st.write("This animation shows the estimated impact of flood based on the infrastructure layers and flood prediction.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
