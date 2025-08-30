import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.layers import TFSMLayer
import numpy as np
import gdown, zipfile, os, pathlib

# Google Drive ZIP ID (from your link)
MODEL_ID = "1zoECslC9YqUW7DLDw2uFKyR3dIItjAbf"
MODEL_ZIP = "flood_model.zip"
MODEL_DIR = "flood_model"

@st.cache_resource
def load_flood_model():
    # Step 1: Download model zip
    if not os.path.exists(MODEL_ZIP):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_ZIP, quiet=False)

    # Step 2: Extract model if not already
    if not os.path.exists(MODEL_DIR):
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    # Step 3: Detect model format
    keras_files = list(pathlib.Path(MODEL_DIR).glob("*.keras"))
    h5_files = list(pathlib.Path(MODEL_DIR).glob("*.h5"))
    saved_model_dirs = [p for p in pathlib.Path(MODEL_DIR).iterdir() if p.is_dir()]

    if keras_files:
        st.success("Loaded .keras model ✅")
        return load_model(str(keras_files[0]))
    elif h5_files:
        st.success("Loaded .h5 model ✅")
        return load_model(str(h5_files[0]))
    elif saved_model_dirs:
        st.success("Loaded SavedModel ✅ (using TFSMLayer)")
        return TFSMLayer(str(saved_model_dirs[0]), call_endpoint="serving_default")
    else:
        raise ValueError("❌ No supported model format found in extracted zip.")

# Load model
model = load_flood_model()
