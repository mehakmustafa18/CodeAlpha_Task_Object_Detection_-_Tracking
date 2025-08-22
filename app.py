# Streamlit web app (UI + backend)
import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Utility: pick device (GPU if available)
# -----------------------------
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0  # GPU index 0 for Ultralytics
        return "cpu"
    except Exception:
        return "cpu"

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_bbox_with_label(img, x1, y1, x2, y2, label):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 0), -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="YOLOv8 + Deep SORT Tracking", layout="wide")
st.title("🧠 YOLOv8 + Deep SORT — Real-time Object Detection & Tracking")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=0,
    )
    conf_thres = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
    iou_thres = st.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05)
    max_age = st.slider("Deep SORT: max_age", 5, 120, 30, 5)
    n_init = st.slider("Deep SORT: n_init", 1, 10, 3, 1)
    nn_budget = st.slider("Deep SORT: nn_budget", 10, 256, 100, 10)
    save_video = st.checkbox("Save annotated output video")

    uploaded_file = st.file_uploader(
        "Upload video (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"]
    )
    run_btn = st.toggle("▶️ Start / Stop", value=False)

# Load model once and cache
@st.cache_resource(show_spinner=True)
def load_model(name):
    model = YOLO(name)
    names = model.model.names if hasattr(model.model, "names") else {}
    return model, names

model, class_names = load_model(model_name)

# Optional class filter UI
with st.sidebar:
    st.subheader("Class Filter (optional)")
    class_options = [f"{i}: {n}" for i, n in class_names.items()]
    selected = st.multiselect("Track only these classes", class_options, default=[])
    selected_ids = set()
    for s in selected:
        try:
            idx = int(s.split(":")[0])
            selected_ids.add(idx)
        except Exception:
            pass

# Prepare Deep SORT
tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

# Prepare video source
temp_video_path = None
if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    tfile.flush()
    temp_video_path = tfile.name
    source = temp_video_path
else:
    source = None

# Main display area
video_placeholder = st.empty()
status_placeholder = st.empty()

writer = None
out_path = None

# Run loop
if run_btn and source is not None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open video source.")
    else:
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            out_dir = Path("runs/webapp")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / f"output_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        device = get_device()
        frame_count = 0
        t0 = time.time()

        while run_btn:
            ok, frame = cap.read()
            if not ok:
                status_placeholder.warning("End of stream or cannot read frame.")
                break

            results = model.predict(
                source=frame,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
                device=device,
            )
            result = results[0]

            dets = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    cls_id = int(k)
                    if selected_ids and cls_id not in selected_ids:
                        continue
                    # ✅ Proper Deep SORT format
                    dets.append([[x1, y1, x2, y2], float(c), cls_id])

            tracks = tracker.update_tracks(dets, frame=frame)

            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                l, t_, r, b = t.to_ltrb()
                label = f"ID {tid}"
                if hasattr(t, "det_class") and t.det_class is not None:
                    cls_id = int(t.det_class)
                    cls_name = class_names.get(cls_id, str(cls_id))
                    label += f" {cls_name}"
                draw_bbox_with_label(frame, l, t_, r, b, label)

            frame_count += 1
            elapsed = max(1e-6, time.time() - t0)
            fps_live = frame_count / elapsed
            cv2.putText(
                frame,
                f"FPS: {fps_live:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB")

            if writer is not None:
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()
            st.success(f"Saved: {out_path}")

if temp_video_path and Path(temp_video_path).exists():
    try:
        os.unlink(temp_video_path)
    except Exception:
        pass

st.caption("Built with Ultralytics YOLOv8 + Deep SORT | Streamlit UI")
