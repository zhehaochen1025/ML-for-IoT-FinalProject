# plot_data_distribution.py
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==== 配置你的 data 目录和各个 client 的 npz 文件名 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLIENT_FILES = {
    "byh": "byh_imu_75feat_minmax_norm.npz",
    "czh": "czh_imu_75feat_minmax_norm.npz",
    "fje": "fje_imu_75feat_minmax_norm.npz",
    # 你有几个 client 就写几个
}

class_counts_per_client = {}
all_classes = set()

for client, fname in CLIENT_FILES.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在，请检查文件名")

    data = np.load(path)
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    cnt = Counter(y_all.tolist())
    class_counts_per_client[client] = cnt
    all_classes.update(cnt.keys())

all_classes = sorted(list(all_classes))
print("Classes:", all_classes)

clients = list(class_counts_per_client.keys())
num_clients = len(clients)
num_classes = len(all_classes)

x = np.arange(num_clients)
width = 0.12  # 每个类别 bar 的宽度

fig, ax = plt.subplots(figsize=(8, 4))

for i, c in enumerate(all_classes):
    heights = []
    for client in clients:
        heights.append(class_counts_per_client[client].get(c, 0))
    ax.bar(x + i * width, heights, width=width, label=f"Class {c}")

ax.set_xticks(x + width * (num_classes - 1) / 2)
ax.set_xticklabels(clients)
ax.set_ylabel("Number of samples")
ax.set_title("Label distribution per client (data kept local)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = os.path.join(BASE_DIR, "../Slides/figures/fig_data_dist_per_client.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("已保存图像:", out_path)
