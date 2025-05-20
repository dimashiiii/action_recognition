import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
DATA_DIR = 'keypoints_data'  # папка с fight/, fall/, normal/
OUTPUT_TRAIN = 'train_data.pkl'
OUTPUT_VAL = 'val_data.pkl'

LABEL_NAMES = ['fight', 'fall', 'normal']
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_NAMES)}

C, T, V, M = 2, 30, 17, 5  # Channels, Frames, Joints, Persons

# === LOAD AND FORMAT KEYPOINTS ===
data_list = []
label_list = []

for label_name in LABEL_NAMES:
    folder_path = os.path.join(DATA_DIR, label_name)
    label = LABEL_MAP[label_name]
    
    for file in os.listdir(folder_path):
        if not file.endswith('.npy'):
            continue
        file_path = os.path.join(folder_path, file)
        keypoints = np.load(file_path)  # shape: (T, 5, 17, 3)

        # Проверка на соответствие ожидаемой формы
        if keypoints.shape[1] != M or keypoints.shape[2] != V:
            print(f"⚠️ Warning: skipping {file} due to unexpected shape {keypoints.shape}")
            continue

        # Обрезка или дополнение по кадрам
        if keypoints.shape[0] < T:
            padding = np.zeros((T - keypoints.shape[0], M, V, 3))
            keypoints = np.vstack((keypoints, padding))
        else:
            keypoints = keypoints[:T]

        # Перевод в ST-GCN формат: (C, T, V, M)
        skeleton = np.zeros((C, T, V, M), dtype=np.float32)
        for m in range(M):
            skeleton[0, :, :, m] = keypoints[:, m, :, 0]  # x
            skeleton[1, :, :, m] = keypoints[:, m, :, 1]  # y

        data_list.append(skeleton)
        label_list.append(label)

print(f"✅ Total samples loaded: {len(data_list)}")

# === SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(
    data_list, label_list, test_size=0.2, stratify=label_list, random_state=42
)

# === SAVE FUNCTION ===
def save_to_pkl(X, y, output_file):
    data_array = np.stack(X)
    label_array = np.array(y)
    output = {
        'data': data_array,               # shape: (N, C, T, V, M)
        'label': label_array,             # shape: (N,)
        'label_name': LABEL_NAMES         # class names
    }
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)
    print(f"✅ Saved {len(X)} samples to {output_file}")

# === SAVE FILES ===
save_to_pkl(X_train, y_train, OUTPUT_TRAIN)
save_to_pkl(X_val, y_val, OUTPUT_VAL)
