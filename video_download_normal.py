import os
import requests
from tqdm import tqdm

save_dir = r"C:\Users\user\Documents\action_detection\dataset\normal"
os.makedirs(save_dir, exist_ok=True)

base_url = "https://fenix.ur.edu.pl/mkepski/ds/data"

print("[INFO] Starting download of normal videos...")

for i in range(1, 41):  # From adl-01 to adl-40
    file_name = f"adl-{i:02d}-cam0.mp4"
    video_url = f"{base_url}/{file_name}"
    save_path = os.path.join(save_dir, file_name)

    if os.path.exists(save_path):
        print(f"[✓] Already exists: {file_name}")
        continue

    try:
        print(f"[↓] Downloading: {file_name}")
        response = requests.get(video_url, stream=True)

        if response.status_code != 200:
            print(f"[✗] Failed to download {file_name} (HTTP {response.status_code})")
            continue

        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"[✔] Saved: {file_name}")

    except Exception as e:
        print(f"[ERROR] Exception while downloading {file_name}: {e}")

print("[DONE] All normal videos downloaded.")
