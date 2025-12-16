# central_training_pc.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==== 配置 ====
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

print("Central X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

classes_cnt = int(len(np.unique(y_train)))
input_dim   = X_train.shape[1]
print("input_dim   =", input_dim)
print("classes_cnt =", classes_cnt)

# ==== 2. DataLoader ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ==== 3. 定义网络 ====
class IMUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = IMUNet(input_dim, 64, classes_cnt).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# ==== 4. 评估函数 ====
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

# ==== 5. 训练，记录 history ====
train_acc_hist = []
val_acc_hist   = []
val_loss_hist  = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)

    train_loss = total_loss / total
    train_acc = total_correct / total

    val_loss, val_acc = eval_loader(val_loader)

    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)
    val_loss_hist.append(val_loss)

    print(f"[Central] Epoch {epoch:03d} | "
          f"Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
          f"Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# test
test_loss, test_acc = eval_loader(test_loader)
print("\n[Central] Final Test: loss = %.4f, acc = %.3f" % (test_loss, test_acc))

# 保存 history
out_path = os.path.join(BASE_DIR, "data/central_history.npz")
np.savez(out_path,
         train_acc=np.array(train_acc_hist),
         val_acc=np.array(val_acc_hist),
         val_loss=np.array(val_loss_hist))
print("Central history saved to:", out_path)
