import cv2
import os
import json

# ✅ Fix: use OpenCV's built-in path to haarcascade file
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def analyze_video(video_path):
    # Load the face detector
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        print(f"❌ Error: Could not load cascade file at {CASCADE_PATH}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return None

    total_frames = 0
    failed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If no faces found, count as failed frame
        if len(faces) == 0:
            failed_frames += 1

        total_frames += 1

        # Automatically stop after 300 frames (for faster testing)
        if total_frames >= 300:
            break

    cap.release()

    failure_rate = (failed_frames / total_frames) * 100 if total_frames > 0 else 0
    return {"total": total_frames, "failed": failed_frames, "failure_rate": failure_rate}


def main():
    print("\n🚀 Running Visual Failure Analysis for all videos...\n")

    videos = {
        "Benchmark": "videos/video_benchmark.mp4",
        "StressTest": "videos/video_stress_test.mp4"
    }

    results = {}

    for name, path in videos.items():
        res = analyze_video(path)
        if res is not None:
            print(f"✅ {name} — Failure Rate: {res['failure_rate']:.2f}% ({res['failed']} failed of {res['total']} frames)")
            results[name] = res

    if results:
        with open("failure_report.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\n📊 Report saved as failure_report.json")
    else:
        print("\n⚠️ No videos analyzed successfully.")


if __name__ == "__main__":
    main()

