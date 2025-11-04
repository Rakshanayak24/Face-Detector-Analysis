import cv2
import numpy as np

# -------------------------------
# Helper: Compute Intersection over Union (IoU)
# -------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# -------------------------------
# Load models
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
# Load video
# -------------------------------
video_path = "videos/video_benchmark.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Cannot open {video_path}")

ious = []
frame_count = 0

print("📊 Calculating IoU between YuNet and SSD...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    yunet.setInputSize((w, h))
    _, faces_yunet = yunet.detect(frame)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    boxes_ssd = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes_ssd.append(box.astype(int))

    if faces_yunet is not None and len(boxes_ssd) > 0:
        for (x, y, fw, fh) in faces_yunet[:, :4]:
            box_yunet = [x, y, x + fw, y + fh]
            for box_ssd in boxes_ssd:
                iou = compute_iou(box_yunet, box_ssd)
                if iou > 0.1:  # small overlap threshold to count
                    ious.append(iou)

    frame_count += 1
    if frame_count >= 200:
        break

cap.release()

if ious:
    mean_iou = np.mean(ious)
    print(f"✅ Compared {frame_count} frames.")
    print(f"📈 Average IoU (overlap agreement): {mean_iou:.3f}")
else:
    print("⚠️ No overlapping detections found — try another video with clear faces.")
