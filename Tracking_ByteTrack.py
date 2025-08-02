# --- Setup ---
!pip
install - q
deep_sort_realtime

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from ultralytics import YOLO
import motmetrics as mm
import pandas as pd
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Paths ---
FRAME_ROOT = "/kaggle/input/drone-zenodo-partial-v1/ExtractedFrames_IR"
GT_ROOT = "/kaggle/working/yolo_labels"
OUTPUT_DIR = "/kaggle/working/tracked_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Setup ---
model_weights_path = "/kaggle/input/yolo11n-icip2023dataset/yolo11n_raihan_best.pt"
model = YOLO(model_weights_path)
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['bird', 'drone']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# --- Initialize Metrics Accumulator ---
acc = mm.MOTAccumulator(auto_id=True)

# --- Track all videos ---
video_folders = sorted(os.listdir(FRAME_ROOT))

for folder in video_folders:
    print(f"\n📽️ Processing video: {folder}")
    FRAME_DIR_PATH = os.path.join(FRAME_ROOT, folder)
    GT_LABEL_PATH = os.path.join(GT_ROOT, folder.replace('_LABELS', ''))
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f"tracked_{folder}.mp4")

    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR_PATH, "*.png")))
    if len(frame_paths) == 0:
        print(f"⚠️ No frames in {folder}. Skipping.")
        continue

    # --- Initialize DeepSORT ---
    deepsort_tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.4,
        nn_budget=None,
        override_track_class=None
    )
    trace_annotator = sv.TraceAnnotator(trace_length=9999, color=sv.Color.BLUE)
    track_area_history = defaultdict(lambda: deque(maxlen=5))

    # --- Video Writer ---
    first_frame = cv2.imread(frame_paths[0])
    h, w = first_frame.shape[:2]
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for idx, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        results = model(frame, verbose=False, conf=0.5)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

        # Convert detections to DeepSORT format
        detection_list = []
        for xyxy, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = xyxy
            class_name = CLASS_NAMES_DICT[class_id]
            detection_list.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

        tracks = deepsort_tracker.update_tracks(detection_list, frame=frame)

        labels, directions = [], []
        current_gt_ids, current_gt_boxes = [], []
        current_pred_ids, current_pred_boxes = [], []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            class_name = track.get_det_class()
            conf = track.get_det_conf()
            if class_name is None or conf is None:
                continue

            class_id = {v: k for k, v in CLASS_NAMES_DICT.items()}.get(class_name, -1)
            if class_id not in SELECTED_CLASS_IDS:
                continue

            label = f"{class_name} {conf:.2f}"

            direction = ""
            track_area_history[track_id].append((x2 - x1) * (y2 - y1))
            if len(track_area_history[track_id]) == track_area_history[track_id].maxlen:
                avg_area = np.mean(list(track_area_history[track_id])[:-1])
                cur_area = track_area_history[track_id][-1]
                if cur_area > avg_area * 1.08:
                    direction = "Approaching"
                elif cur_area < avg_area * 0.92:
                    direction = "Receding"
                else:
                    direction = "Constant"

            label = f"#{track_id} {label}"
            current_pred_ids.append(track_id)
            current_pred_boxes.append([cx, cy])
            labels.append(label)
            directions.append(direction)

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if directions[-1]:
                color = (0, 255, 0) if "Approaching" in directions[-1] else \
                    (0, 0, 255) if "Receding" in directions[-1] else (0, 255, 255)
                cv2.putText(frame, directions[-1], (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Load GT
        label_path = os.path.join(GT_LABEL_PATH, os.path.basename(frame_path).replace('.png', '.txt'))
        if os.path.exists(label_path):
            img_h, img_w = frame.shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w_, h_ = map(float, parts)
                    if int(cls) not in SELECTED_CLASS_IDS:
                        continue
                    x1_gt = (cx - w_ / 2) * img_w
                    y1_gt = (cy - h_ / 2) * img_h
                    x2_gt = (cx + w_ / 2) * img_w
                    y2_gt = (cy + h_ / 2) * img_h
                    gt_cx, gt_cy = (x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2
                    current_gt_ids.append(len(current_gt_ids))
                    current_gt_boxes.append([gt_cx, gt_cy])

        # Update MOTAccumulator
        if current_gt_ids and current_pred_ids:
            dist_matrix = np.full((len(current_gt_ids), len(current_pred_ids)), np.nan)
            for i, (gt_cx, gt_cy) in enumerate(current_gt_boxes):
                for j, (pred_cx, pred_cy) in enumerate(current_pred_boxes):
                    dist = np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2)
                    dist_matrix[i, j] = dist / 50  # Normalize by object size
            acc.update(current_gt_ids, current_pred_ids, dist_matrix)

        video_writer.write(frame)

    video_writer.release()

# --- Compute Final Metrics ---
metrics = mm.metrics.create()
summary = metrics.compute(
    acc,
    metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr",
             "mostly_tracked", "mostly_lost", "num_switches",
             "num_fragmentations", "precision", "recall"],
    name='ALL_VIDEOS'
)

print("\n✅ Combined Tracking Metrics (All Videos):")
print(summary.to_string())

# Save Excel
excel_path = "/kaggle/working/tracking_metrics_summary_all.xlsx"
summary.to_excel(excel_path)
print(f"\n📊 Saved combined metrics summary: {excel_path}")


import os
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

# Paths
image_root = '/kaggle/input/drone-zenodo-partial-v1/ExtractedFrames_IR'
label_root = '/kaggle/working/yolo_labels'
output_root = '/kaggle/working/hota_predictions'  # TrackEval-compatible
os.makedirs(output_root, exist_ok=True)

# DeepSORT tracker
tracker = DeepSort(max_age=30)

# For each video folder
video_list = sorted(os.listdir(label_root))

for video_name in video_list:
    print(f"🔁 Processing video: {video_name}")
    label_folder = os.path.join(label_root, video_name)
    frame_folder = os.path.join(image_root, video_name + "_LABELS")
    output_file = os.path.join(output_root, video_name + ".txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Track ID output
    frame_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])
    results = []

    for frame_file in tqdm(frame_files):
        frame_id = int(frame_file.replace('frame_', '').replace('.txt', ''))
        label_path = os.path.join(label_folder, frame_file)
        image_path = os.path.join(frame_folder, frame_file.replace('.txt', '.png'))

        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
        except:
            print(f"[!] Skipping frame {frame_file} due to image read error.")
            continue

        detections = []
        with open(label_path, 'r') as f:
            for line in f:
                cls, x_c, y_c, w, h = map(float, line.strip().split())
                x = (x_c - w / 2) * width
                y = (y_c - h / 2) * height
                w_pix = w * width
                h_pix = h * height
                conf = 1.0  # YOLO label files don’t store confidence
                detections.append(([x, y, w_pix, h_pix], conf, cls))

        # Run DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            cls = int(track.det_class)
            results.append(f"{frame_id} {track_id} {l:.2f} {t:.2f} {w:.2f} {h:.2f} 1 -1 -1 {cls}")

    # Save TrackEval-compatible prediction file
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))

print("✅ All tracking results saved in TrackEval format.")
