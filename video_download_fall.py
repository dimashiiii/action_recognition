import os
import requests

from tqdm import tqdm

# Папка для сохранения
save_dir = r"C:\Users\user\Documents\action_detection\dataset\fall"
os.makedirs(save_dir, exist_ok=True)

# Базовый URL
base_url = "https://fenix.ur.edu.pl/mkepski/ds/data"

print("[INFO] Начинаю загрузку видео...")

for i in range(1, 31):  # fall-01 to fall-30
    for cam in [0, 1]:  # cam0 и cam1
        file_name = f"fall-{i:02d}-cam{cam}.mp4"
        url = f"{base_url}/{file_name}"
        save_path = os.path.join(save_dir, file_name)

        print(f"[INFO] Проверяю: {file_name}")

        if os.path.exists(save_path):
            print(f"[✓] Уже загружено: {file_name}")
            continue

        try:
            print(f"[↓] Загружаю с {url}")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                print(f"[✗] Не удалось загрузить {file_name} (HTTP {response.status_code})")
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

            print(f"[✔] Сохранено в: {save_path}")

        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке {file_name}: {e}")

print("[DONE] Скрипт завершён.")
