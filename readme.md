Overview:
1. Assets to Acquire
You must acquire the following assets. This part of the challenge tests your resourcefulness.
A. The Test Videos (Must be 720p)
File 1: video_benchmark.mp4
Resolution: 720p (1280x720).
Content: A "standard" but challenging video. Search royalty-free sites (like Pexels, Pixabay) for terms like "busy cafe," "walking in the city," or "office meeting."
Goal: This video should have a good mix of faces at different angles and distances, plus background "noise" (posters, art, patterns) that could trigger False Positives.
File 2: video_stress_test.mp4
Resolution: 720p (1280x720).
Content: A "difficult" video designed to find the model's breaking point. Search for terms like "people at night," "low light club," or "blurry street footage."
Goal: This video should test robustness against noise, blur, and extreme lighting conditions.
B. The Model Options (Choose ANY 2)
Model 1: The "Classic" (Haar Cascade)
File: haarcascade_frontalface_default.xml
Link: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 
Model 2: The "Classic DNN" (Caffe SSD)
File 1 (.prototxt): deploy.prototxt.txt
Link 1: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt 
File 2 (.caffemodel): res10_300x300_ssd_iter_140000.caffemodel
Link 2: https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel 
Model 3: The "Modern DNN" (YuNet)
File: face_detection_yunet_2023mar.onnx
Link: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx 
(Note: All models are CPU-friendly. You will need a local Python environment with libraries like opencv-python and numpy to complete the task.)
--------------------------------------------------------------------------------------------


# 1️⃣ Clone or download the repository
git clone https://github.com/<your-username>/face_detector_analysis.git
cd face_detector_analysis
# 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate     # (Windows)
# or source venv/bin/activate  (Mac/Linux)

# 3️⃣ Install dependencies
pip install opencv-python numpy reportlab matplotlib

# 4️⃣ Run the analysis
python analyze_fps.py
python visual_failure_analysis.py
python generate_report.py
python final_report.py
