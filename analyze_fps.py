import cv2
import time
import os
import json

# -------------------------------
# Helper Function to Measure FPS
# -------------------------------
def measure_fps(video_path):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return 0.0

    print(f"\n🎥 Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠️ Could not open {video_path}")
        return 0.0

    total_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = "models/haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            print(f"❌ Missing model file: {cascade_path}")
            break

        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"⚠️ Failed to load cascade at {cascade_path}")
            break

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        total_frames += 1

        if total_frames % 50 == 0:
            print(f"   ⏳ Processed {total_frames} frames...")

    end_time = time.time()
    cap.release()

    fps = total_frames / (end_time - start_time + 1e-6)
    print(f"✅ Done. FPS: {fps:.2f}")
    return fps


# -------------------------------
# Main Experiment
# -------------------------------
def main():
    print("\n🚀 Running FPS Analysis...\n")

    videos = {
        "Benchmark": "videos/video_benchmark.mp4",
        "StressTest": "videos/video_stress_test.mp4"
    }

    results = {}
    for name, path in videos.items():
        results[name] = measure_fps(path)

    os.makedirs("results", exist_ok=True)
    with open("results/fps_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n💾 FPS Results saved to results/fps_results.json")
    print("📊 Summary:")
    for k, v in results.items():
        print(f"   • {k}: {v:.2f} FPS")


if __name__ == "__main__":
    main()





