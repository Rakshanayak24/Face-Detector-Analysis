import cv2

# Paths to videos
videos = [
    "videos/video_benchmark.mp4",
    "videos/video_stress_test.mp4"
]

for path in videos:
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {path}")
        continue
    
    # Get basic info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🎥 {path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {frame_count}")

    # Show a sample frame
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Sample Frame - {path}", frame)
        cv2.waitKey(2000)  # Display for 2 seconds
        cv2.destroyAllWindows()
    else:
        print(f"⚠️ Could not read frame from {path}")

    cap.release()

cv2.destroyAllWindows()
