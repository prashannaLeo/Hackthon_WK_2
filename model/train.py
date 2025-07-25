import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_PATH = "data/eye_metrics_dataset.csv"
MODEL_PATH = "blink_disease.h5"
SCALER_PATH = "scaler.pkl"

# Step 1: Load dataset
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Step 2: Check for nulls
print("🔍 Checking for missing values...")
if df.isnull().sum().any():
    print(df.isnull().sum())
    print("⚠️ Dataset contains missing values. Please clean before training.")
    exit()

# Step 3: Define features
features = ["blink_rate", "avg_blink_duration", "gaze_stability", "screen_distance"]
X = df[features]

# Step 4: Simulate dummy labels temporarily
labels = np.random.choice(["normal", "eye_strain", "dry_eye"], size=len(X))
le = LabelEncoder()
y = le.fit_transform(labels)

# Step 5: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to {SCALER_PATH}")

# Step 6: Train/test split
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Build TensorFlow model
print("🧠 Building TensorFlow model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 8: Train model
print("🚀 Training model...")
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=1)

# Step 9: Evaluate
print("\n📈 Evaluation:")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

# Step 10: Save model
model.save(MODEL_PATH)
print(f"✅ TensorFlow model saved to {MODEL_PATH}")