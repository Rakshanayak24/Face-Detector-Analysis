import cv2
import os
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Automatically pick video
# -------------------------------
video_dir = "videos"
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

if not video_files:
    print("❌ No video found in 'videos/' folder.")
    exit()

video_path = os.path.join(video_dir, video_files[0])
print(f"🎥 Using video: {video_path}")

# -------------------------------
# Step 2: Initialize face detector
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("❌ Error loading Haar cascade.")
    exit()

# -------------------------------
# Step 3: Process video
# -------------------------------
cap = cv2.VideoCapture(video_path)
confidences = []
frame_count = 0

if not cap.isOpened():
    print("❌ Error opening video file.")
    exit()

print("📈 Collecting confidence scores for ~300 frames...")

while frame_count < 300:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Relaxed parameters for higher sensitivity
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    # Approximate "confidence" = relative face size
    for (x, y, w, h) in faces:
        confidence = (w * h) / (frame.shape[0] * frame.shape[1])
        confidences.append(confidence)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()

# -------------------------------
# Step 4: Plot results
# -------------------------------
if len(confidences) == 0:
    print("⚠️ No faces detected, nothing to plot.")
else:
    plt.figure(figsize=(8, 4))
    plt.hist(confidences, bins=20, color='royalblue', alpha=0.7)
    plt.title("Distribution of Detected Face Confidences")
    plt.xlabel("Relative Face Size (proxy for confidence)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

print(f"✅ Processed {frame_count} frames, detected {len(confidences)} faces.")

