from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()

def add_slide(title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.text = bullets[0] if bullets else ""
    for b in bullets[1:]:
        p = tf.add_paragraph()
        p.text = b
        p.level = 0

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "AgriShield AI"
slide.placeholders[1].text = "Smarter Detection. Precise Spraying. Sustainable Farming."

add_slide("Problem", [
    "Pesticide overuse wastes money and harms environment.",
    "Farmers spray entire fields even when only parts are infected.",
    "Need targeted spraying based on disease detection.",
])

add_slide("Solution", [
    "Smart Spray Planner: detect disease on leaves, map whole field, spray only where needed.",
    "AI model (MobileNetV2) classifies tomato leaf diseases.",
    "Field mode splits photo into grid, decides Spray/Skip/Watch/Retake.",
    "Simulation compares Mode A (spray all) vs Mode B (targeted): litres, time, savings.",
    "Live IoT dashboard (AgriShield) monitors sensors and triggers spraying.",
])

add_slide("Tech Stack", [
    "Python, TensorFlow/Keras (MobileNetV2), OpenCV.",
    "Streamlit (web app), Qdrant (hybrid RAG: dense + BM25).",
    "scikit-learn, matplotlib, seaborn, pandas.",
    "Kaggle datasets (PlantVillage, PlantDoc).",
    "GitHub for version control.",
])

add_slide("AI Model", [
    "MobileNetV2 transfer learning, 10 tomato disease classes.",
    "Validation accuracy: 0.913 (91.3%).",
    "Low confidence -> retake photo (never guess).",
    "Grad-CAM for explainability.",
    "Severity estimation (colour-based, Horsfall-Barratt simplified).",
])

add_slide("Field Mode", [
    "Split field photo into 3x3 (configurable) grid.",
    "Vegetation mask (ExG) to skip soil/no-crop tiles.",
    "Sub-crop voting: affected_score from model probability.",
    "Decision: Spray / Skip / Watch / Retake (blur/no leaf).",
    "Strictness: Cautious / Balanced / Fewer checks.",
    "Overlay labels short: 'A2 Spray'.",
])

add_slide("Spray Plan & Simulation", [
    "Upload field photo -> auto spray map + recommendation.",
    "Per-section spray amount (litres).",
    "Mode A vs Mode B: litres, seconds, % saved.",
    "Route steps with nozzle ON/OFF and priority.",
    "Weather gate (GO/WAIT) from extension rules.",
    "Download spray_plan.csv and map PNG.",
])

add_slide("Live IoT Dashboard (AgriShield)", [
    "Real-time sensor monitoring (temp, humidity, soil, tank, battery).",
    "AI prediction integration: disease, confidence, healthy/infected.",
    "Smart selective spraying: auto toggle, manual button, pump status.",
    "Analytics: healthy vs infected pie, usage comparison, savings.",
    "Notifications and activity log.",
])

add_slide("Results", [
    "Validation accuracy: 91.3%.",
    "Field accuracy (PlantDoc): 22.3% (domain gap).",
    "Mode A: 4.5 L / 225 s; Mode B: 2.5 L / 195 s; saved 44.4%.",
    "RAG hit rates: BM25 1.0, dense 0.733, hybrid 0.867.",
    "Held-out field accuracy after fine-tune: 0.2335.",
])

add_slide("Limitations & Future", [
    "Field accuracy lower than lab (domain shift).",
    "Severity colour-based; best on plain backgrounds.",
    "Weather rules from US/Canadian guidance; label overrides.",
    "Future: more field data, better domain adaptation, real IoT sensors.",
])

prs.save("results/AgriShield_AI_Presentation.pptx")
print("saved")
