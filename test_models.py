import cv2

# Test YuNet
print("🔹 Testing YuNet model...")
yunet = cv2.FaceDetectorYN.create(
    model="models/face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320)
)
print("✅ YuNet loaded successfully.")

# Test Caffe SSD
print("\n🔹 Testing Caffe SSD model...")
ssd_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)
print("✅ Caffe SSD loaded successfully.")
