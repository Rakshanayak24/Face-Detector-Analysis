import json
import os
from datetime import datetime

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} not found. Skipping...")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Load results ---
failure_data = load_json("failure_results.json")
success_data = load_json("success_results.json")

# --- Extract safely ---
def extract_metrics(data, label):
    if not data or label not in data:
        return {"total": 0, "failed": 0, "failure_rate": 0.0}
    return data[label]

benchmark = extract_metrics(failure_data, "Benchmark")
stress = extract_metrics(failure_data, "StressTest")

# --- Generate summary ---
text_summary = f"""
📊 FACE DETECTOR ANALYSIS REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

⚙️ BENCHMARK RESULTS:
    • Total frames: {benchmark['total']}
    • Failed detections: {benchmark['failed']}
    • Failure rate: {benchmark['failure_rate']:.2f}%

🧪 STRESS TEST RESULTS:
    • Total frames: {stress['total']}
    • Failed detections: {stress['failed']}
    • Failure rate: {stress['failure_rate']:.2f}%

🕒 OVERALL SUMMARY:
    • Combined total frames: {benchmark['total'] + stress['total']}
    • Combined failures: {benchmark['failed'] + stress['failed']}
    • Overall failure rate: {((benchmark['failed'] + stress['failed']) / (benchmark['total'] + stress['total']))*100 if (benchmark['total'] + stress['total'])>0 else 0:.2f}%

✅ Report successfully generated.
"""

print(text_summary)

# --- Save report ---
with open("report.txt", "w", encoding="utf-8") as f:
    f.write(text_summary)
print("[INFO] report.txt saved successfully (UTF-8).")




