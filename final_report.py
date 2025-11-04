# final_report.py
# --------------------------------------------------------
# 📊 FACE DETECTOR FINAL REPORT GENERATOR
# --------------------------------------------------------
# Combines FPS results + failure analysis into one PDF file
# --------------------------------------------------------

import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ----------- Load Data Safely -----------
def load_json(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] {filename} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"[WARN] Error decoding {filename}.")
        return {}

fps_data = load_json("fps_results.json")
failure_data = load_json("failure_results.json")

# ----------- Prepare Report -----------
doc = SimpleDocTemplate("Face_Detector_Report.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>📊 FACE DETECTOR ANALYSIS REPORT</b>", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph("Generated as part of the Face Detector Performance Evaluation Project.", styles["Normal"]))
story.append(Spacer(1, 20))

# ----------- FPS Section -----------
if fps_data:
    story.append(Paragraph("<b>⚙️ FPS ANALYSIS RESULTS</b>", styles["Heading2"]))
    table_data = [["Video", "FPS"]]
    for name, value in fps_data.items():
        table_data.append([name, f"{value:.2f}"])
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
else:
    story.append(Paragraph("⚠️ No FPS data found.", styles["Normal"]))
    story.append(Spacer(1, 20))

# ----------- Failure Section -----------
if failure_data:
    story.append(Paragraph("<b>🧪 FAILURE ANALYSIS RESULTS</b>", styles["Heading2"]))
    table_data = [["Video", "Total Frames", "Failed Frames", "Failure Rate (%)"]]
    for name, stats in failure_data.items():
        table_data.append([name, stats["total"], stats["failed"], f"{stats['failure_rate']:.2f}"])
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
else:
    story.append(Paragraph("⚠️ No failure analysis data found.", styles["Normal"]))
    story.append(Spacer(1, 20))

# ----------- Summary Section -----------
if failure_data:
    total_frames = sum(v["total"] for v in failure_data.values())
    total_failed = sum(v["failed"] for v in failure_data.values())
    overall_rate = (total_failed / total_frames) * 100 if total_frames > 0 else 0

    story.append(Paragraph("<b>🕒 OVERALL SUMMARY</b>", styles["Heading2"]))
    story.append(Paragraph(
        f"• Combined Total Frames: {total_frames}<br/>"
        f"• Combined Failed Detections: {total_failed}<br/>"
        f"• Overall Failure Rate: {overall_rate:.2f}%<br/>",
        styles["Normal"]
    ))

story.append(Spacer(1, 30))
story.append(Paragraph("<b>✅ Report successfully generated.</b>", styles["Normal"]))

# ----------- Build PDF -----------
doc.build(story)
print("📄 Face_Detector_Report.pdf has been created successfully!")
