import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO("yolov8n-pose.pt")  # облегчённая модель для скорости

VIDEO_DIR = "dataset"
SAVE_DIR = "keypoints_data"
os.makedirs(SAVE_DIR, exist_ok=True)

max_persons = 5
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 240

for label in os.listdir(VIDEO_DIR):
    label_path = os.path.join(VIDEO_DIR, label)
    save_label_path = os.path.join(SAVE_DIR, label)
    os.makedirs(save_label_path, exist_ok=True)

    for video_file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(label_path, video_file)
        cap = cv2.VideoCapture(video_path)

        keypoints_sequence = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Ресайз для ускорения
            frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            results = model(frame_resized)

            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                persons = results[0].keypoints.data.cpu().numpy()
                if persons.size == 0:
                    persons = persons.reshape((0, 17, 3))
            else:
                persons = np.zeros((0, 17, 3), dtype=np.float32)

            num_persons = persons.shape[0]

            if num_persons >= max_persons:
                keypoints_frame = persons[:max_persons]
            else:
                padding = np.zeros((max_persons - num_persons, 17, 3), dtype=np.float32)
                keypoints_frame = np.concatenate([persons, padding], axis=0)

            keypoints_sequence.append(keypoints_frame)

        cap.release()

        keypoints_array = np.array(keypoints_sequence)
        save_path = os.path.join(save_label_path, video_file.replace(".mp4", ".npy"))
        np.save(save_path, keypoints_array)

print("✅ Все keypoints успешно извлечены и сохранены.")
