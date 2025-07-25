import cv2
import os
import time
import numpy as np

def get_eye_color(eye_img):
    """Analyze the eye image to determine the iris color."""
    hsv = cv2.cvtColor(eye_img, cv2.COLOR_BGR2HSV)
    h, w = eye_img.shape[:2]
    iris_region = hsv[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    avg_hsv = np.mean(iris_region, axis=(0, 1))
    hue = avg_hsv[0]
    if 0 <= hue < 30 or 150 <= hue < 180:
        return "Brown"
    elif 30 <= hue < 90:
        return "Green"
    elif 90 <= hue < 150:
        return "Blue"
    else:
        return "Unknown"

def estimate_screen_distance(face_width_pixels, reference_width_pixels=200, reference_distance_cm=50):
    """Estimate screen distance based on face width in pixels."""
    return (reference_width_pixels / face_width_pixels) * reference_distance_cm if face_width_pixels > 0 else 0

def get_realtime_eye_metrics(eyes, face_width, current_time, start_time, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers, left_eye_color, right_eye_color):
    """Calculate real-time eye-tracking metrics."""
    # Blink detection and duration
    if len(eyes) == 0 and prev_eye_count > 0 and current_time - last_blink_time > 0.7:  # Increased threshold
        blink_count += 1
        last_blink_time = current_time
        blink_start_time = current_time
    elif len(eyes) > 0 and blink_start_time is not None:
        blink_duration = (current_time - blink_start_time) * 1000  # Convert to ms
        if 50 < blink_duration < 1000:
            blink_durations.append(blink_duration)
        blink_start_time = None
    prev_eye_count = len(eyes)
    
    # Gaze stability
    gaze_stability = 0
    if len(eyes) > 0:
        for (x, y, w, h) in eyes[:2]:
            center = (x + w/2, y + h/2)
            eye_centers.append(center)
        if len(eye_centers) > 30:
            eye_centers = eye_centers[-30:]
            centers_array = np.array(eye_centers)
            std_dev = np.std(centers_array, axis=0)
            gaze_stability = (std_dev[0] + std_dev[1]) / face_width * 100 if face_width > 0 else 0
    
    # Calculate metrics
    blink_rate = (blink_count / (current_time - start_time)) * 60 if current_time > start_time else 0
    avg_blink_duration = np.mean(blink_durations) if blink_durations else 0
    
    return {
        "blink_rate": blink_rate,
        "avg_blink_duration": avg_blink_duration,
        "gaze_stability": gaze_stability,
        "screen_distance": estimate_screen_distance(face_width) if face_width > 0 else 0,
        "left_eye_color": left_eye_color,
        "right_eye_color": right_eye_color
    }, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers

def track_eye(video_source=0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    if face_cascade.empty() or eye_cascade.empty():
        print("Error: Could not load cascade classifiers")
        return
    
    os.makedirs('captured_eyes', exist_ok=True)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    capture_count = 0
    tracking_active = True
    start_time = time.time()
    capture_done = False
    capture_delay = 5
    blink_count = 0
    blink_durations = []
    prev_eye_count = 0
    last_blink_time = start_time
    blink_start_time = None
    left_eye_color = "Unknown"
    right_eye_color = "Unknown"
    eye_centers = []
    
    cv2.namedWindow('Eye Tracking System', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            
            if tracking_active:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
                
                eyes = []
                face_width = 0
                fx = 0
                fw = 0
                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                    face_width = fw
                    roi_gray = gray[fy:fy+fh//2, fx:fx+fw]
                    roi_color = frame[fy:fy+fh//2, fx:fx+fw]
                    detected_eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.01, minNeighbors=10, minSize=(30, 30))
                    for (ex, ey, ew, eh) in detected_eyes:
                        eyes.append((fx + ex, fy + ey, ew, eh))
                
                eyes = sorted(eyes, key=lambda x: x[0])
                
                current_time = time.time()
                metrics, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers = get_realtime_eye_metrics(
                    eyes, face_width, current_time, start_time, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers, left_eye_color, right_eye_color
                )
                
                for i, (x, y, w, h) in enumerate(eyes[:2]):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    center = (x + w/2, y + h/2)
                    cv2.line(frame, (int(center[0])-10, int(center[1])), (int(center[0])+10, int(center[1])), (0, 0, 255), 1)
                    cv2.line(frame, (int(center[0]), int(center[1])-10), (int(center[0]), int(center[1])+10), (0, 0, 255), 1)
                    label = "Right Eye" if i == 1 else "Left Eye"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(frame, f"Eyes: {len(eyes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Blink Rate: {metrics['blink_rate']:.1f}/min", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Avg Blink Duration: {metrics['avg_blink_duration']:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze Stability: {metrics['gaze_stability']:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Screen Distance: {metrics['screen_distance']:.1f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Eye Color: {metrics['left_eye_color']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Eye Color: {metrics['right_eye_color']}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "'q': Quit", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if not capture_done:
                    time_remaining = max(0, capture_delay - (current_time - start_time))
                    cv2.putText(frame, f"Time Remaining: {time_remaining:.1f}s", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if not capture_done and current_time - start_time >= capture_delay and len(eyes) > 0:
                    timestamp = int(time.time())
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
                    capture_count += len(eyes[:2])
                    if len(eyes) == 1:
                        x, y, w, h = eyes[0]
                        eye_img = clean_frame[y:y+h, x:x+w]
                        face_center_x = fx + fw / 2 if len(faces) > 0 else frame.shape[1] / 2
                        eye_name = 'right_eye' if x + w/2 > face_center_x else 'left_eye'
                        eye_color = get_eye_color(eye_img)
                        if eye_name == 'left_eye':
                            left_eye_color = eye_color
                        else:
                            right_eye_color = eye_color
                        filename = f"captured_eyes/{eye_name}_{timestamp}.png"
                        cv2.imwrite(filename, eye_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                        print(f"Saved {filename} (Detected color: {eye_color})")
                        capture_count += 1
                    capture_done = True
                
                cv2.imshow('Eye Tracking System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        final_metrics = get_realtime_eye_metrics(
            eyes, face_width, time.time(), start_time, blink_count, blink_durations, prev_eye_count, last_blink_time, blink_start_time, eye_centers, left_eye_color, right_eye_color
        )[0]
        print(f"Session ended. Captured {capture_count} eye images.")
        print(f"Final Metrics:")
        print(f" - Blink Rate: {final_metrics['blink_rate']:.1f} blinks/min")
        print(f" - Avg Blink Duration: {final_metrics['avg_blink_duration']:.1f} ms")
        print(f" - Gaze Stability: {final_metrics['gaze_stability']:.1f}%")
        print(f" - Screen Distance: {final_metrics['screen_distance']:.1f} cm")
        print(f" - Left Eye Color: {final_metrics['left_eye_color']}")
        print(f" - Right Eye Color: {final_metrics['right_eye_color']}")

if __name__ == "__main__":
    print("Eye Tracking System")
    print("Instructions:")
    print(" - 'q': Quit program")
    print(" - Single capture of each eye after 5 seconds with color detection")
    print(" - Countdown timer and metrics displayed on preview")
    track_eye()