#for IR

!pip install -q ultralytics
# --- 1. Installation and Checks ---
print("--- Installing YOLOv8 (Ultralytics) and Supervision ---")
!pip install -q ultralytics supervision[assets]
print("\n--- Ultralytics and Supervision installed successfully. ---")
!pip install -q motmetrics

import os
import cv2
import numpy as np
import ultralytics
import supervision as sv
from IPython import display
from collections import defaultdict, deque
import glob
from tqdm import tqdm
from ultralytics import YOLO
import motmetrics as mm


# --- 1. Installation and Checks ---
print("--- Installing YOLOv8 (Ultralytics) and Supervision ---")
!pip install -q ultralytics
!pip install supervision[assets]==0.24.0
display.clear_output()

print("\n--- Ultralytics and Supervision installed successfully. ---")
ultralytics.checks()
print(f"supervision.__version__: {sv.__version__}")

import time

# -----------------------------------------------------------
# 1. INPUT: Path to folder containing extracted frames (.png/.jpg)
# -----------------------------------------------------------
FRAME_DIR_PATH = "/kaggle/input/drone-zenodo-partial-v1/ExtractedFrames_IR/IR_BIRD_001_LABELS"
GT_LABEL_PATH = "/kaggle/working/yolo_labels/IR_BIRD_001"  # Ground-truth labels folder (YOLO format)
OUTPUT_VIDEO_PATH = "/kaggle/working/output_tracked_IR_BIRD_001.mp4"

# Sorted list of all frames
frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR_PATH, "*.png")))

if len(frame_paths) == 0:
    raise FileNotFoundError("No frames found in the folder! Check FRAME_DIR_PATH.")

# -----------------------------------------------------------
# 2. Setup Model, Tracker, Annotators
# -----------------------------------------------------------
model_weights_path = "/kaggle/input/yolo11n-icip2023dataset/yolo11n_raihan_best.pt"
model = YOLO(model_weights_path)
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['bird', 'drone']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

trace_annotator = sv.TraceAnnotator(trace_length=9999, color=sv.Color.BLUE)
track_area_history = defaultdict(lambda: deque(maxlen=5))

# Video writer setup from first frame size
first_frame = cv2.imread(frame_paths[0])
h, w = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30, (w, h))

# For metrics
acc = mm.MOTAccumulator(auto_id=True)

# -----------------------------------------------------------
# 3. Tracking + Annotating
# -----------------------------------------------------------
for idx, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    start = time.time()
    # run inference + tracking on 1 frame
    results = model(frame, verbose=False, conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)
    end = time.time()

    labels, directions = [], []
    current_gt_ids, current_gt_boxes = [], []
    current_pred_ids, current_pred_boxes = [], []

    if len(detections) > 0:
        for box, tracker_id, class_id, confidence in zip(
                detections.xyxy, detections.tracker_id, detections.class_id, detections.confidence
        ):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Label text (class + conf)
            label_text = f"{CLASS_NAMES_DICT[class_id]} {confidence:.2f}"

            direction = ""
            if tracker_id is not None:
                track_area_history[tracker_id].append((x2 - x1) * (y2 - y1))
                if len(track_area_history[tracker_id]) == track_area_history[tracker_id].maxlen:
                    avg_prev_area = np.mean(list(track_area_history[tracker_id])[:-1])
                    cur_area = track_area_history[tracker_id][-1]
                    if cur_area > avg_prev_area * 1.08:
                        direction = "Approaching"
                    elif cur_area < avg_prev_area * 0.92:
                        direction = "Receding"
                    else:
                        direction = "Constant"

                label_text = f"#{tracker_id} {label_text}"
                current_pred_ids.append(tracker_id)
                current_pred_boxes.append([cx, cy])

            labels.append(label_text)
            directions.append(direction)

    # --- Load ground-truth labels ---
    label_path = os.path.join(GT_LABEL_PATH, os.path.basename(frame_path).replace('.png', '.txt'))
    if os.path.exists(label_path):
        img_h, img_w = frame.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                cls, cx, cy, w_, h_ = map(float, line.strip().split())
                x1_gt = (cx - w_ / 2) * img_w
                y1_gt = (cy - h_ / 2) * img_h
                x2_gt = (cx + w_ / 2) * img_w
                y2_gt = (cy + h_ / 2) * img_h
                gt_cx, gt_cy = (x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2
                current_gt_ids.append(len(current_gt_ids))
                current_gt_boxes.append([gt_cx, gt_cy])

    # --- Update MOTA ---

    # --- Update MOTA + MOTP (Option 1: normalized by object size) ---
    if current_gt_ids and current_pred_ids:
        dist_matrix = np.full((len(current_gt_ids), len(current_pred_ids)), np.nan)

        # We assume you have ground truth boxes info: (cx, cy, width, height)
        for i, (gt_cx, gt_cy) in enumerate(current_gt_boxes):
            for j, (pred_cx, pred_cy) in enumerate(current_pred_boxes):
                # Find the matching ground truth size (approximate)
                # You need ground truth box width/height here
                # If you don't have, estimate using avg box size
                if "current_gt_boxes_info" in locals():
                    gt_w, gt_h = current_gt_boxes_info[i][0], current_gt_boxes_info[i][1]
                else:
                    # Fallback: assume average object size ~50px
                    gt_w, gt_h = 50, 50

                dist = np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2)
                gt_diag = np.sqrt(gt_w ** 2 + gt_h ** 2)

                if gt_diag > 0:
                    dist_matrix[i, j] = dist / gt_diag

        acc.update(current_gt_ids, current_pred_ids, dist_matrix)

    # --- Annotate trace (tracked paths) ---
    frame = trace_annotator.annotate(scene=frame, detections=detections)

    # --- Draw bounding boxes, class & direction ---
    for i, box in enumerate(detections.xyxy.astype(int)):
        x1, y1, x2, y2 = box
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label above
        cv2.putText(frame, labels[i], (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw direction below
        if directions[i]:
            color = (0, 255, 0) if "Approaching" in directions[i] else \
                (0, 0, 255) if "Receding" in directions[i] else (0, 255, 255)
            cv2.putText(frame, directions[i], (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    video_writer.write(frame)

video_writer.release()

fps = 1 / (end - start)
print(f"FPS running inference and tracking on a single video: {fps}")

import motmetrics as mm
import pandas as pd
import matplotlib.pyplot as plt

# Compute summary as before
metrics = mm.metrics.create()

summary = metrics.compute(
    acc,
    metrics=[
        "num_frames", "mota", "motp", "idf1", "idp", "idr",
        "mostly_tracked", "mostly_lost", "num_switches",
        "num_fragmentations", "precision", "recall"
    ],
    name='IR_TRACKING'
)

print("\n--- Tracking Evaluation Metrics (MOTChallenge Format) ---")
print(summary.to_string())

# ✅ Save metrics to Excel
excel_path = "/kaggle/working/tracking_metrics_summary.xlsx"
summary.to_excel(excel_path)
print(f"✅ Metrics summary saved as Excel: {excel_path}")
