# ================================================================
# age_detection_ui_final.py
# Multi-User Login + Sign-Up System + Age Detection Dashboard
# ================================================================
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
import csv

# ================================================================
# 1️⃣ File Paths
# ================================================================
BASE_DIR = r"C:\Users\Bommasani peddaraju\OneDrive\Desktop\age_detection"
USERS_FILE = os.path.join(BASE_DIR, "users.csv")
MODEL_PATH = os.path.join(BASE_DIR, "age_model.h5")

# ================================================================
# 2️⃣ Load Model
# ================================================================
@st.cache_resource
def load_age_model():
    model = load_model(MODEL_PATH)
    return model

model = load_age_model()

# ================================================================
# 3️⃣ Utility Functions
# ================================================================
def preprocess_image(image):
    image = np.array(image.convert("L"))
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = image.reshape(1, 48, 48, 1)
    return image


@st.cache_data
def predict_age(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img_prep = preprocess_image(img)
    pred_age = model.predict(img_prep, verbose=0)
    age = int(np.round(pred_age[0][0]))
    lower = max(0, age - 2)
    upper = age + 2
    return age, lower, upper


def log_prediction(file_name, predicted_age, age_range):
    log_file = os.path.join(BASE_DIR, "prediction_history.csv")
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
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
# 4️⃣ User Management (Sign Up + Login)
# ================================================================
def ensure_users_file():
    """Create users.csv if it doesn't exist."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password", "name"])

def load_users():
    ensure_users_file()
    return pd.read_csv(USERS_FILE)

def check_login(username, password, users_df):
    user = users_df[(users_df["username"] == username) & (users_df["password"] == password)]
    if not user.empty:
        return user.iloc[0]["name"]
    return None

def register_user(name, username, password):
    ensure_users_file()
    users_df = pd.read_csv(USERS_FILE)
    if username in users_df["username"].values:
        return False  # username already exists
    with open(USERS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password, name])
    return True

# ================================================================
# 5️⃣ Age Detection UI
# ================================================================
def age_detection_ui(user_name):
    st.sidebar.success(f"Logged in as: {user_name}")
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
            log_prediction("camera_image.jpg", age, f"{lower}-{upper}")

# ================================================================
# 6️⃣ Admin Dashboard
# ================================================================
def admin_dashboard():
    st.title("🧠 System Dashboard (Admin Panel)")
    st.write("Monitor performance, manage upload limits, and access history.")

    st.subheader("🖥️ System Information")
    st.text(f"OS: {platform.system()} {platform.release()}")
    st.text(f"Processor: {platform.processor()}")
    st.text(f"TensorFlow version: {tf.__version__}")

    st.subheader("📊 System Performance")
    memory = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)
    st.progress(cpu_usage / 100)
    st.text(f"CPU Usage: {cpu_usage}%")
    st.text(
        f"Available Memory: {round(memory.available / (1024**3), 2)} GB / "
        f"Total: {round(memory.total / (1024**3), 2)} GB"
    )

    st.subheader("🧾 Prediction History")
    log_file = os.path.join(BASE_DIR, "prediction_history.csv")
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
# 7️⃣ Main App Logic
# ================================================================
def main():
    st.set_page_config(page_title="Age Detection Dashboard", page_icon="🧠", layout="centered")
    ensure_users_file()
    users_df = load_users()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_name = ""

    if not st.session_state.logged_in:
        mode = st.radio("Select Mode:", ["🔐 Login", "🆕 Sign Up"])

        if mode == "🔐 Login":
            st.title("Login to Dashboard")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                name = check_login(username, password, users_df)
                if name:
                    st.session_state.logged_in = True
                    st.session_state.user_name = name
                    st.success(f"Welcome, {name}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        elif mode == "🆕 Sign Up":
            st.title("Create New Account")
            name = st.text_input("Full Name")
            username = st.text_input("Choose Username")
            password = st.text_input("Choose Password", type="password")

            if st.button("Register"):
                if name and username and password:
                    success = register_user(name, username, password)
                    if success:
                        st.success("✅ Account created successfully! You can now log in.")
                    else:
                        st.warning("⚠️ Username already exists. Please choose another.")
                else:
                    st.error("❌ All fields are required.")
    else:
        age_detection_ui(st.session_state.user_name)
        st.markdown("---")
        if st.checkbox("🧠 Open Admin Dashboard"):
            admin_dashboard()

        st.sidebar.button("Logout", on_click=lambda: logout())

        st.markdown("---")
        st.markdown("👨‍💻 Developed by *Bommasani Peddaraju & Sreedhar* | Mini Project 2025")

def logout():
    st.session_state.logged_in = False
    st.session_state.user_name = ""
    st.rerun()

if __name__ == "__main__":
    main()
