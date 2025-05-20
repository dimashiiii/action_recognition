import torch
from torch.utils.data import DataLoader
from dataset import KeypointsDataset  

def main():
    dataset = KeypointsDataset(root_dir="keypoints_data", max_frames=60)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"🔍 Количество сэмплов в датасете: {len(dataset)}")

    for i, (keypoints, labels) in enumerate(dataloader):
        print(f"\n🔢 Батч {i + 1}")
        print(f"Keypoints shape: {keypoints.shape}")  # (B, 60, 5, 17, 3)
        print(f"Labels: {labels}")                   # (B,) тензор из 0/1/2
        print(f"Unique labels in this batch: {labels.unique().tolist()}")

        if i == 1:  # протестируем только 2 батча
            break

    # Пример: проверить один сэмпл напрямую
    sample_kps, sample_label = dataset[0]
    print(f"\n🧪 Один сэмпл:")
    print(f"Keypoints shape: {sample_kps.shape}")  # (60, 5, 17, 3)
    print(f"Label: {sample_label.item()}")

if __name__ == "__main__":
    main()
