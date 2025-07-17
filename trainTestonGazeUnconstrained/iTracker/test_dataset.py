from itracker_dataset import ITrackerDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置你的数据路径，这里举例
root_dir = "/home/lunet/cowz2/Documents/GazeEstimation/Dataset/GazeUnconstrained/processed_dataset"  # 修改成你自己的路径

# 加载数据集
dataset = ITrackerDataset(root_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 取一条数据看看
sample = next(iter(loader))

print("Face shape:", sample["image_face"].shape)
print("Left eye shape:", sample["image_left"].shape)
print("Right eye shape:", sample["image_right"].shape)
print("Face grid shape:", sample["face_grid"].shape)
print("Gaze label:", sample["gaze"])

# 显示图像（可选）
face = sample["image_face"][0].permute(1, 2, 0).numpy()
left_eye = sample["image_left"][0].permute(1, 2, 0).numpy()
right_eye = sample["image_right"][0].permute(1, 2, 0).numpy()

plt.subplot(1, 3, 1)
plt.imshow(face)
plt.title("Face")

plt.subplot(1, 3, 2)
plt.imshow(left_eye)
plt.title("Left Eye")

plt.subplot(1, 3, 3)
plt.imshow(right_eye)
plt.title("Right Eye")

plt.show()
