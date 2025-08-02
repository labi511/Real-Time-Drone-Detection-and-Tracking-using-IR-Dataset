import os
import cv2
import random
import matplotlib.pyplot as plt

# Path to the directory containing videos
video_dir = '/kaggle/input/video-data/DroneDetectionThesis-Drone-detection-dataset-e7a6eaf/Data/Video_IR'

# List all video files and filter out those with 'DRONE' or 'BIRD' in the name
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]

# Find a drone video and a bird video
drone_video = next((f for f in video_files if 'drone' in f.lower()), None)
bird_video = next((f for f in video_files if 'bird' in f.lower()), None)

print("Drone video:", drone_video)
print("Bird video:", bird_video)


def extract_random_frames(video_path, num_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print(f"Video has only {total_frames} frames; extracting all.")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = random.sample(range(total_frames), num_frames)
        frame_indices.sort()

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def show_frames(frames, title):
    plt.figure(figsize=(15, 5))
    for i, frame in enumerate(frames):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame)
        plt.title(f"{title} Frame {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Extract and display 5 random frames from each video
if drone_video:
    drone_frames = extract_random_frames(os.path.join(video_dir, drone_video))
    show_frames(drone_frames, "Drone")

if bird_video:
    bird_frames = extract_random_frames(os.path.join(video_dir, bird_video))
    show_frames(bird_frames, "Bird")
