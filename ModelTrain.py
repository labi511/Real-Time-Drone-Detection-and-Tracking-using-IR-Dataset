# Setup
model_weights_path = "/kaggle/input/best-retrain-ir/best_retrain_ir.pt"
model = YOLO(model_weights_path)
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['bird', 'drone']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

import os
from PIL import Image

# Input paths
label_root = '/kaggle/input/drone-zenodo-labels-v1/all_detection_txt'
image_root = '/kaggle/input/drone-zenodo-partial-v1/ExtractedFrames_IR'

# Output path for YOLO-formatted labels
output_root = '/kaggle/working/yolo_labels'
os.makedirs(output_root, exist_ok=True)

# Define class mapping: original -> new
class_map = {
    0: 2,  # airplane -> 2
    1: 1,  # drone -> 1
    2: 0,  # bird -> 0
    3: 3   # helicopter -> 3
}

# List all video folders from labels
subfolders = sorted(os.listdir(label_root))

for video_name in subfolders:
    label_folder = os.path.join(label_root, video_name)
    image_folder = os.path.join(image_root, video_name + '_LABELS')
    output_folder = os.path.join(output_root, video_name)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(label_folder):
        if not file.endswith('.txt'):
            continue

        label_path = os.path.join(label_folder, file)
        image_name = file.replace('.txt', '.png')  # Assuming image is .png
        image_path = os.path.join(image_folder, image_name)
        output_path = os.path.join(output_folder, file)

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"[ERROR] Could not open {image_path}: {e}")
            continue

        new_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 6:
                    continue
                x, y, w, h, conf, cls = map(float, parts)

                original_cls = int(cls)
                mapped_cls = class_map.get(original_cls, -1)
                if mapped_cls == -1:
                    # Skip classes not in the map
                    continue

                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                new_lines.append(f"{mapped_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save YOLO-format label
        with open(output_path, 'w') as f:
            f.write("\n".join(new_lines))

print("✅ All labels converted to YOLO format with new class mapping.")
