import os
import cv2

input_dir = r"C:\Users\user\Documents\action_detection\dataset\normal"
output_dir = r"C:\Users\user\Documents\action_detection\dataset\normal_cropped"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".mp4"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cropped_width = width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (cropped_width, height))

    print(f"[INFO] Processing: {filename}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        right_half = frame[:, width // 2:] 
        out.write(right_half)

    cap.release()
    out.release()
    print(f"[✓] Saved cropped video to: {output_path}")

print("[DONE] All videos processed.")
