import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt    # 新增：用于可视化

# ============================================
# 1. 读取 .npz 数据（里面已经是 min-max 归一化过的特征）
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLIENT_FILES = {
    "byh": "byh_imu_75feat_minmax_norm.npz",
    "czh": "czh_imu_75feat_minmax_norm.npz",
    "fje": "fje_imu_75feat_minmax_norm.npz",
}

EPOCHS = 50
BATCH_SIZE = 1
LR = 0.0015

# ==== 1. 读入所有 client 的数据并拼在一起 ====
X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
X_test_list,  y_test_list  = [], []

for client, fname in CLIENT_FILES.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在")

    data = np.load(path)
    X_train_list.append(data["X_train"])
    y_train_list.append(data["y_train"])
    X_val_list.append(data["X_val"])
    y_val_list.append(data["y_val"])
    X_test_list.append(data["X_test"])
    y_test_list.append(data["y_test"])

X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
y_train = np.concatenate(y_train_list, axis=0).astype(np.int64)
X_val   = np.concatenate(X_val_list,   axis=0).astype(np.float32)
y_val   = np.concatenate(y_val_list,   axis=0).astype(np.int64)
X_test  = np.concatenate(X_test_list,  axis=0).astype(np.float32)
y_test  = np.concatenate(y_test_list,  axis=0).astype(np.int64)

# 这些是基于 RAW 特征算出来的 min/max，Arduino 用；这里我们只打印看看，不再做二次归一化
feature_min = data["feature_min"]
feature_max = data["feature_max"]

# 类型转换
X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)
X_test  = X_test.astype(np.float32)

y_train = y_train.astype(np.int64)
y_val   = y_val.astype(np.int64)
y_test  = y_test.astype(np.int64)

print("X_train shape:", X_train.shape)
print("X_val   shape:", X_val.shape)
print("X_test  shape:", X_test.shape)

# 简单 sanity check：现在的特征大概在 0~1 之间
print("X_train range:", X_train.min(), "->", X_train.max())
print("X_val   range:", X_val.min(),   "->", X_val.max())
print("X_test  range:", X_test.min(),  "->", X_test.max())

# 类别数 & 输入维度
classes_cnt = int(len(np.unique(y_train)))
first_layer_input_cnt = X_train.shape[1]  # 应该是 75
print("first_layer_input_cnt =", first_layer_input_cnt)
print("classes_cnt            =", classes_cnt)

# ============================================
# 2. 转成 PyTorch Tensor + DataLoader（不再额外归一化）
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)

X_val_t   = torch.from_numpy(X_val)
y_val_t   = torch.from_numpy(y_val)

X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

batch_size = 1
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ============================================
# 3. 定义与 Arduino 一样结构的 MLP：75 → 64 → classes_cnt
# ============================================
class IMUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # 输出 logits，CrossEntropyLoss 里自带 softmax
        return x

model = IMUNet(first_layer_input_cnt, 64, classes_cnt).to(device)
print(model)

# ============================================
# 4. 超参数（尽量贴近 Arduino：SGD + 小学习率）
# ============================================
LEARNING_RATE = 0.0015
EPOCH = 50

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# ============================================
# 5. 评估函数
# ============================================
def eval_loader(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, total_correct / total

# ============================================
# 6. 训练循环（只用一次归一化后的特征） + 记录曲线
# ============================================
train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

for epoch in range(1, EPOCH + 1):
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_train_correct += (preds == yb).sum().item()
        total_train += xb.size(0)

    train_loss = total_train_loss / total_train
    train_acc  = total_train_correct / total_train

    val_loss, val_acc = eval_loader(val_loader)

    # 记录到列表里，用于后面画图
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch:03d} | "
          f"Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
          f"Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# ============================================
# 7. 在 test 集上评估
# ============================================
test_loss, test_acc = eval_loader(test_loader)
print("\nFinal Test:")
print(f"  test_loss = {test_loss:.4f}")
print(f"  test_acc  = {test_acc:.3f}")

# ============================================
# 8. 画训练过程曲线并保存
# ============================================
epochs = np.arange(1, EPOCH + 1)


# 图 2：上下两个子图（loss / acc 分开）
fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig.suptitle("Training / Validation Curves")

ax_loss.plot(epochs, train_losses, label="Train Loss")
ax_loss.plot(epochs, val_losses,   label="Val Loss", linestyle="--")
ax_loss.set_ylabel("Loss")
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)

ax_acc.plot(epochs, train_accs, label="Train Acc")
ax_acc.plot(epochs, val_accs,   label="Val Acc", linestyle="--")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.legend()
ax_acc.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_path2 = os.path.join(BASE_DIR, "../Slides/figures/training_loss_acc.png")
plt.savefig(out_path2, dpi=300)
plt.close()
print("Saved curve figure:", out_path2)
