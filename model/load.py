
# load_model.py

import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import time
import pickle
from datetime import datetime
import pandas as pd

# Optional: Install tabulate for table formatting (pip install tabulate)
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from eye_tracking.tracker import get_realtime_eye_metrics, get_eye_color
except ImportError as e:
    print(f"Error importing from Eye_tracking.tracker: {e}")
    sys.exit(1)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "blink_disease.h5")
print(f"Attempting to load model from: {model_path}")

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")
print(f"Attempting to load scaler from: {scaler_path}")

if not os.path.exists(scaler_path):
    print(f"Error: Scaler file '{scaler_path}' not found. Please run train.py to generate it.")
    sys.exit(1)

try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading scaler: {e}")
    sys.exit(1)

# Simplified function to capture one set of metrics
def get_single_eye_metrics(video_source=0, capture_delay=15):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    if face_cascade.empty() or eye_cascade.empty():
        print("Error: Could not load cascade classifiers")
        return None

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    start_time = time.time()
    blink_count = 0
    blink_durations = []
    prev_eye_count = 0
    last_blink_time = start_time
    blink_start_time = None
    eye_centers = []
    left_eye_color = "Unknown"
    right_eye_color = "Unknown"
    capture_done = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            eyes = []
            face_width = 0
            fx = 0
            fw = 0
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Blue face rectangle
                face_width = fw
                roi_gray = gray[fy:fy+fh//2, fx:fx+fw]
                roi_color = frame[fy:fy+fh//2, fx:fx+fw]
                detected_eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.01, minNeighbors=10, minSize=(30, 30))
                for (ex, ey, ew, eh) in detected_eyes:
                    eyes.append((fx + ex, fy + ey, ew, eh))

            eyes = sorted(eyes, key=lambda x: x[0])
            print(f"Detected eyes: {len(eyes)}")  # Debug: Check eye detection

            # Draw eye rectangles and labels
            for i, (x, y, w, h) in enumerate(eyes[:2]):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green eye rectangle
                center = (x + w/2, y + h/2)
                cv2.line(frame, (int(center[0])-10, int(center[1])), (int(center[0])+10, int(center[1])), (0, 0, 255), 1)
                cv2.line(frame, (int(center[0]), int(center[1])-10), (int(center[0]), int(center[1])+10), (0, 0, 255), 1)
                label = "Right Eye" if i == 1 else "Left Eye"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            current_time = time.time()

            # Capture eye images for color detection
            if not capture_done and len(eyes) > 0 and current_time - start_time >= capture_delay:
                timestamp = int(time.time())
                os.makedirs('captured_eyes', exist_ok=True)
                for i, (x, y, w, h) in enumerate(eyes[:2]):
                    eye_img = clean_frame[y:y+h, x:x+w]
                    eye_name = 'right_eye' if i == 1 else 'left_eye'
                    eye_color = get_eye_color(eye_img)
                    if i == 0:
                        left_eye_color = eye_color
                    else:
                        right_eye_color = eye_color
                    filename = f"captured_eyes/{eye_name}_{timestamp}.png"
                    cv2.imwrite(filename, eye_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    print(f"Saved {filename} (Detected color: {eye_color})")
                capture_done = True

            # Get metrics
            metrics, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers = get_realtime_eye_metrics(
                eyes, face_width, current_time, start_time, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers, left_eye_color, right_eye_color
            )

            # Display metrics
            cv2.putText(frame, f"Eyes Detected: {len(eyes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink Rate: {metrics['blink_rate']:.1f}/min", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Avg Blink Duration: {metrics['avg_blink_duration']:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze Stability: {metrics['gaze_stability']:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Screen Distance: {metrics['screen_distance']:.1f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Left Eye Color: {metrics['left_eye_color']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye Color: {metrics['right_eye_color']}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time Remaining: {max(0, capture_delay - (current_time - start_time)):.1f}s", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if current_time - start_time >= capture_delay:
                cap.release()
                cv2.destroyAllWindows()
                return metrics

            cv2.imshow('Eye Tracking for Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return None

# Get real-time eye metrics
metrics = get_single_eye_metrics(capture_delay=15)
if metrics is None:
    print("Error: Failed to obtain eye metrics")
    sys.exit(1)

# Format metrics output
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\nEye Tracking Metrics ({timestamp}):")
if tabulate:
    table = [
        ["Blink Rate", f"{metrics['blink_rate']:.1f}", "blinks/min"],
        ["Avg Blink Duration", f"{metrics['avg_blink_duration']:.1f}", "ms"],
        ["Gaze Stability", f"{metrics['gaze_stability']:.1f}", "%"],
        ["Screen Distance", f"{metrics['screen_distance']:.1f}", "cm"],
        ["Left Eye Color", metrics['left_eye_color'], ""],
        ["Right Eye Color", metrics['right_eye_color'], ""]
    ]
    print(tabulate(table, headers=["Metric", "Value", "Unit"], tablefmt="fancy_grid"))
else:
    print("=" * 50)
    print(f"{'Metric':<25} {'Value':<10} {'Unit':<10}")
    print("=" * 50)
    print(f"{'Blink Rate':<25} {metrics['blink_rate']:<10.1f} blinks/min")
    print(f"{'Avg Blink Duration':<25} {metrics['avg_blink_duration']:<10.1f} ms")
    print(f"{'Gaze Stability':<25} {metrics['gaze_stability']:<10.1f} %")
    print(f"{'Screen Distance':<25} {metrics['screen_distance']:<10.1f} cm")
    print(f"{'Left Eye Color':<25} {metrics['left_eye_color']:<10}")
    print(f"{'Right Eye Color':<25} {metrics['right_eye_color']:<10}")
    print("=" * 50)

# Save metrics to a log file
os.makedirs('logs', exist_ok=True)
log_file = f"logs/eye_metrics_{int(time.time())}.txt"
with open(log_file, 'w') as f:
    f.write(f"Eye Tracking Metrics ({timestamp}):\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Metric':<25} {'Value':<10} {'Unit':<10}\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Blink Rate':<25} {metrics['blink_rate']:<10.1f} blinks/min\n")
    f.write(f"{'Avg Blink Duration':<25} {metrics['avg_blink_duration']:<10.1f} ms\n")
    f.write(f"{'Gaze Stability':<25} {metrics['gaze_stability']:<10.1f} %\n")
    f.write(f"{'Screen Distance':<25} {metrics['screen_distance']:<10.1f} cm\n")
    f.write(f"{'Left Eye Color':<25} {metrics['left_eye_color']:<10}\n")
    f.write(f"{'Right Eye Color':<25} {metrics['right_eye_color']:<10}\n")
    f.write("=" * 50 + "\n")
print(f"Metrics saved to {log_file}")

# Extract and reshape values with feature names
feature_names = ["blink_rate", "avg_blink_duration", "gaze_stability", "screen_distance"]
input_data = pd.DataFrame([[metrics["blink_rate"],
                            metrics["avg_blink_duration"],
                            metrics["gaze_stability"],
                            metrics["screen_distance"]]],
                          columns=feature_names)

# Apply preprocessing
input_scaled = scaler.transform(input_data)  # Use saved scaler with feature names

# Predict
try:
    prediction = model.predict(input_scaled)
    predicted_class = prediction.argmax(axis=1)[0]
    label_map = {0: "dry_eye", 1: "eye_strain", 2: "normal"}
    print(f"\nPrediction Result ({timestamp}):")
    print("=" * 50)
    print(f"🧠 Predicted Eye Condition: {label_map[predicted_class]}")
    print("=" * 50)
    # Append prediction to log file
    with open(log_file, 'a') as f:
        f.write(f"\nPrediction Result ({timestamp}):\n")
        f.write("=" * 50 + "\n")
        f.write(f"🧠 Predicted Eye Condition: {label_map[predicted_class]}\n")
        f.write("=" * 50 + "\n")
except Exception as e:
    print(f"Error during prediction: {e}")
