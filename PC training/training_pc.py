import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# 1. 读取 .npz 数据
# ============================================
npz_path = "jiuenfeng_imu_75feat_minmax_norm.npz"  # TODO: 改成你自己的 .npz 文件名
data = np.load(npz_path)

print("NPZ keys:", data.files)

X_train = data["X_train"]   # shape (N_train, 75)
y_train = data["y_train"]   # shape (N_train,)
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

feature_min = data["feature_min"]  # shape (75,)
feature_max = data["feature_max"]  # shape (75,)

X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)
X_test  = X_test.astype(np.float32)

y_train = y_train.astype(np.int64)
y_val   = y_val.astype(np.int64)
y_test  = y_test.astype(np.int64)

print("X_train shape:", X_train.shape)
print("X_val   shape:", X_val.shape)
print("X_test  shape:", X_test.shape)

# 类别数 & 输入维度
classes_cnt = int(len(np.unique(y_train)))
first_layer_input_cnt = X_train.shape[1]  # 应该是 75
print("first_layer_input_cnt =", first_layer_input_cnt)
print("classes_cnt            =", classes_cnt)

# ============================================
# 2. 与 Arduino 相同规则做 Min-Max 归一化
#    (x - min) / (max - min)，常量特征置 0
# ============================================
feature_min = feature_min.astype(np.float32)
feature_max = feature_max.astype(np.float32)

eps = 1e-6
range_vec = feature_max - feature_min
mask_zero = np.abs(range_vec) < eps
range_vec[mask_zero] = 1.0  # 防止除 0

def normalize(X):
    Xn = (X - feature_min) / range_vec
    Xn[:, mask_zero] = 0.0
    return Xn

X_train_n = normalize(X_train)
X_val_n   = normalize(X_val)
X_test_n  = normalize(X_test)

# ============================================
# 3. 转成 PyTorch Tensor + DataLoader
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train_t = torch.from_numpy(X_train_n)
y_train_t = torch.from_numpy(y_train)

X_val_t   = torch.from_numpy(X_val_n)
y_val_t   = torch.from_numpy(y_val)

X_test_t  = torch.from_numpy(X_test_n)
y_test_t  = torch.from_numpy(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ============================================
# 4. 定义与 Arduino 一样结构的 MLP：75 → 64 → classes_cnt
# ============================================
class IMUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (N, in_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # 输出 logits，CrossEntropyLoss 里自带 softmax
        return x

model = IMUNet(first_layer_input_cnt, 64, classes_cnt).to(device)
print(model)

# 换成和 Arduino 接近的超参
LEARNING_RATE = 0.0015
EPOCH = 1000

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  # 也可以换成 Adam

# ============================================
# 5. 训练 & 验证
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

    print(f"Epoch {epoch:02d} | "
          f"Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
          f"Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# ============================================
# 6. 在 test 集上评估
# ============================================
test_loss, test_acc = eval_loader(test_loader)
print("\nFinal Test:")
print(f"  test_loss = {test_loss:.4f}")
print(f"  test_acc  = {test_acc:.3f}")
