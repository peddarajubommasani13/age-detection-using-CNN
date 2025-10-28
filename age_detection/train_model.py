# train_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Load dataset
# ---------------------------
print("🔹 Loading dataset...")
df = pd.read_csv("age_data.csv")

# Convert pixels string to numpy array
def str_to_array(pixels):
    return np.array(pixels.split(), dtype="float32").reshape(48, 48, 1) / 255.0

df["pixels"] = df["pixels"].apply(str_to_array)

# Features and labels
X = np.stack(df["pixels"].values)
y = df["age"].values.astype("float32")

print(f"Dataset loaded. {X.shape[0]} samples.")

# ---------------------------
# Build CNN model
# ---------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')  # output is age
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
model.summary()

# ---------------------------
# Train model
# ---------------------------
print("🔹 Training model...")
history = model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# ---------------------------
# Save model
# ---------------------------
model.save("age_model.h5")
print("✅ Model saved as age_model.h5")
