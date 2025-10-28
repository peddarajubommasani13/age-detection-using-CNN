# age_detection_ui_final.py
import streamlit as st
import numpy as np
import io
import cv2
import os
import psutil
import platform
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd

# ================================================================
# 1️⃣ Load Model
# ================================================================
@st.cache_resource
def load_age_model():
    model = load_model("age_model.h5")
    return model

model = load_age_model()

# ================================================================
# 2️⃣ Utility Functions
# ================================================================
def preprocess_image(image):
    """Convert image to grayscale, resize, normalize for CNN input."""
    image = np.array(image.convert("L"))
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = image.reshape(1, 48, 48, 1)
    return image


@st.cache_data
def predict_age(image_bytes):
    """Predict age and 5-year range."""
    img = Image.open(io.BytesIO(image_bytes))
    img_prep = preprocess_image(img)
    pred_age = model.predict(img_prep, verbose=0)
    age = int(np.round(pred_age[0][0]))

    lower = max(0, age - 2)
    upper = age + 2
    return age, lower, upper


def log_prediction(file_name, predicted_age, age_range):
    """Save prediction info into CSV (optional)."""
    log_file = "prediction_history.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date/Time", "File Name", "Predicted Age", "Age Range"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            file_name,
            predicted_age,
            age_range
        ])


# ================================================================
# 3️⃣ Streamlit UI
# ================================================================
st.set_page_config(page_title="Age Detection", page_icon="🧠", layout="centered")
st.title("🧠 Age Detection using CNN")
st.write("Upload a photo or use your webcam to detect **age** using a CNN model.")

option = st.radio("Choose Input Mode:", ["📁 Upload Image", "📸 Use Camera"])

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Predicting..."):
            age, lower, upper = predict_age(image_bytes)

        st.success("✅ Prediction Complete!")
        st.write(f"**Predicted Age Range:** {lower} - {upper} years")

        # Save log entry
        log_prediction(uploaded_file.name, age, f"{lower}-{upper}")

elif option == "📸 Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image_bytes = camera_image.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Captured Image", use_container_width=True)

        with st.spinner("Predicting..."):
            age, lower, upper = predict_age(image_bytes)

        st.success("✅ Prediction Complete!")
        st.write(f"**Predicted Age Range:** {lower} - {upper} years")

        # Save log entry
        log_prediction("camera_image.jpg", age, f"{lower}-{upper}")


# ================================================================
# 4️⃣ Admin Dashboard (Module 6)
# ================================================================
def admin_dashboard():
    """
    Admin Dashboard – Simplified version without login.
    Shows system performance, TensorFlow info, and log access.
    """
    st.title("🧠 System Dashboard (Admin Panel)")
    st.write("Monitor performance, manage upload limits, and access history.")

    # --- System Information ---
    st.subheader("🖥️ System Information")
    st.text(f"OS: {platform.system()} {platform.release()}")
    st.text(f"Processor: {platform.processor()}")
    st.text(f"TensorFlow version: {tf.__version__}")

    # --- Resource Monitoring ---
    st.subheader("📊 System Performance")
    memory = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)
    st.progress(cpu_usage / 100)  # fixed range issue
    st.text(f"CPU Usage: {cpu_usage}%")
    st.text(
        f"Available Memory: {round(memory.available / (1024**3), 2)} GB / "
        f"Total: {round(memory.total / (1024**3), 2)} GB"
    )

    # --- Model Info ---
    st.subheader("🧩 Model Information")
    try:
        total_params = model.count_params()
        st.text(f"Total Trainable Parameters: {total_params:,}")
    except Exception as e:
        st.warning(f"Unable to load model details: {e}")

    # --- Upload Size Control ---
    st.subheader("⚙️ Upload Controls")
    max_upload_size = st.slider("Maximum upload size (MB):", 1, 20, 5)
    st.write(f"✅ Max upload size set to: {max_upload_size} MB")

    # --- History File ---
    st.subheader("🧾 Prediction History")
    log_file = "prediction_history.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download History File",
            data=open(log_file, "rb").read(),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        if st.button("🧹 Clear History"):
            os.remove(log_file)
            st.success("History cleared successfully!")
    else:
        st.info("No prediction history available yet.")


# ================================================================
# 5️⃣ Footer and Admin Access
# ================================================================
st.markdown("---")
if st.checkbox("🧠 Open Admin Dashboard"):
    admin_dashboard()

st.markdown("---")
st.markdown("👨‍💻 Developed by *Bommasani Peddaraju & Sreedhar* | Mini Project 2025")
