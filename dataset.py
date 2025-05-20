import os
import numpy as np
import torch
from torch.utils.data import Dataset

class KeypointsDataset(Dataset):
    def __init__(self, root_dir, max_frames=60, transform=None):
        """
        root_dir: путь к директории с подпапками (fall, normal, fight)
        max_frames: максимальное количество кадров в последовательности
        transform: необязательные преобразования
        """
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.transform = transform

        self.data = []
        self.labels = []

        label_map = {'normal': 0, 'fall': 1, 'fight': 2}  # добавляй больше классов по мере необходимости

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            for file in os.listdir(label_path):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(label_path, file))
                    self.labels.append(label_map[label_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoints = np.load(self.data[idx])  # shape = (T, 5, 17, 3)

        # Усечение или паддинг до max_frames
        T = keypoints.shape[0]
        if T >= self.max_frames:
            keypoints = keypoints[:self.max_frames]
        else:
            pad_shape = (self.max_frames - T, *keypoints.shape[1:])
            padding = np.zeros(pad_shape, dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=0)

        # Преобразование в тензор
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # shape = (max_frames, 5, 17, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            keypoints = self.transform(keypoints)

        return keypoints, label
    
