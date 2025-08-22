# Object Detection & Tracking — YOLOv8 + Deep SORT (Web App)

Web-based real-time **object detection** and **multi-object tracking** using **YOLOv8** and **Deep SORT**. The app runs in your browser with an interactive UI (Streamlit).

---

## ✨ Features
- YOLOv8 pretrained models (n/s/m/l)
- Deep SORT tracker with configurable settings
- Confidence/IoU sliders
- Class filter
- Webcam or video file input
- Option to save annotated output video

---

## 🚀 Setup & Run
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# OR source .venv/bin/activate (Linux/Mac)

pip install -r requirements.txt

streamlit run app.py
