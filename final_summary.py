import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1️⃣ Enter your collected results manually here
# (use your actual observed FPS and IoU values)
# -------------------------------
results = {
    "YuNet": {
        "Benchmark_FPS": 1.37,
        "Stress_FPS": 1.25,
        "IoU_Avg": 0.60,   # Replace with your IoU value from iou_analysis.py
        "Confidence_Mode": 0.85  # Approx median high-confidence detection (if available)
    },
    "Caffe_SSD": {
        "Benchmark_FPS": 7.91,
        "Stress_FPS": 6.90,
        "IoU_Avg": 0.45,
        "Confidence_Mode": 0.75
    }
}

# -------------------------------
# 2️⃣ Print Summary Table
# -------------------------------
print("\n📊 FINAL COMPARISON SUMMARY")
print("---------------------------------------------------------")
print(f"{'Model':<12}{'Bench FPS':>12}{'Stress FPS':>12}{'IoU':>10}{'Conf':>10}")
print("---------------------------------------------------------")
for model, data in results.items():
    print(f"{model:<12}{data['Benchmark_FPS']:>12.2f}{data['Stress_FPS']:>12.2f}{data['IoU_Avg']:>10.2f}{data['Confidence_Mode']:>10.2f}")
print("---------------------------------------------------------")

# -------------------------------
# 3️⃣ Visualization — FPS Bar Chart
# -------------------------------
labels = list(results.keys())
benchmark_fps = [results[m]["Benchmark_FPS"] for m in labels]
stress_fps = [results[m]["Stress_FPS"] for m in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, benchmark_fps, width, label="Benchmark")
plt.bar(x + width/2, stress_fps, width, label="Stress Test")
plt.ylabel("FPS")
plt.title("Average FPS Comparison")
plt.xticks(x, labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# 4️⃣ Visualization — IoU Comparison
# -------------------------------
ious = [results[m]["IoU_Avg"] for m in labels]

plt.figure(figsize=(6, 4))
plt.bar(labels, ious, color=['green', 'orange'])
plt.ylabel("Average IoU")
plt.title("Bounding Box Agreement (Higher = Better)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# 5️⃣ Confidence Comparison
# -------------------------------
conf_modes = [results[m]["Confidence_Mode"] for m in labels]
plt.figure(figsize=(6, 4))
plt.bar(labels, conf_modes, color=['purple', 'gray'])
plt.ylabel("Typical Confidence Score")
plt.title("Confidence Distribution Peak (Benchmark Video)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
