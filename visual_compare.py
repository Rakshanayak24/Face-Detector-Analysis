import cv2
import sys
import time

# -------------------------------
# Load Models
# -------------------------------
yunet = cv2.FaceDetectorYN.create(
    model="models/face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# -------------------------------
# Video Setup
# -------------------------------
video_path = "videos/video_benchmark.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error opening video file.")
    sys.exit(1)

print("🎥 Running automatic comparison for ~10 seconds...\n")

# -------------------------------
# Frame Loop (auto-close)
# -------------------------------
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

    # Stop automatically after ~10 seconds (~300 frames at 30 fps)
    if frame_count > 300:
        print("\n🕒 Auto-closing after 10 seconds of processing...")
        break

    h, w, _ = frame.shape

    # ----- YuNet (green boxes)
    yunet.setInputSize((w, h))
    _, faces_yunet = yunet.detect(frame)
    if faces_yunet is not None:
        for face in faces_yunet:
            box = face[0:4].astype(int)
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, "YuNet", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ----- Caffe SSD (red boxes)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Caffe", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display both detections
    cv2.imshow("Face Detection Comparison", frame)

    # Wait briefly for UI refresh (no key handling)
    cv2.waitKey(10)

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
end_time = time.time()

print(f"\n✅ Processed {frame_count} frames in {end_time - start_time:.2f} s.")
print("✅ Visualization window closed automatically.\n")
