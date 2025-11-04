import cv2
import numpy as np
import time

# -------------------------------
# Load both models
# -------------------------------
yunet = cv2.FaceDetectorYN.create(
    model="models/face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

ssd_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# -------------------------------
# Pick a sample video automatically
# -------------------------------
video_path = "videos/video_benchmark.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Cannot open {video_path}")

# -------------------------------
# Show a few frames only
# -------------------------------
print("🎬 Displaying detections side-by-side for 5 seconds...")
start_time = time.time()
frame_count = 0

while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    yunet.setInputSize((w, h))
    _, faces_yunet = yunet.detect(frame)

    # Draw YuNet detections (Green)
    if faces_yunet is not None:
        for (x, y, w, h) in faces_yunet[:, :4]:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Run SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    # Draw SSD detections (Blue)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("YuNet (Green) vs Caffe SSD (Blue)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ Compared {frame_count} frames successfully.")
