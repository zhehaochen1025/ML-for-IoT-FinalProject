import os
import numpy as np

# ============================================
# 1. 读取 .npz 数据（已 min-max 归一化）
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(BASE_DIR, "data", "czh_imu_75feat_minmax_norm.npz")
data = np.load(npz_path)

print("NPZ keys:", data.files)

X_train = data["X_train"]   # 已经归一化过
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# RAW 特征的 min/max，只给板子用，这里不再归一化
feature_min = data["feature_min"]
feature_max = data["feature_max"]

X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)
X_test  = X_test.astype(np.float32)

y_train = y_train.astype(np.int64)
y_val   = y_val.astype(np.int64)
y_test  = y_test.astype(np.int64)

print("X_train shape:", X_train.shape)
print("X_val   shape:", X_val.shape)
print("X_test  shape:", X_test.shape)
print("X_train range:", X_train.min(), "->", X_train.max())

classes_cnt = int(len(np.unique(y_train)))
input_dim   = X_train.shape[1]   # 75
print("input_dim   =", input_dim)
print("classes_cnt =", classes_cnt)

# ============================================
# 2. 一些工具函数
# ============================================
def one_hot(labels, num_classes):
    """将 int 标签转成 one-hot，shape: (N,) -> (N, C)"""
    N = labels.shape[0]
    y_oh = np.zeros((N, num_classes), dtype=np.float32)
    y_oh[np.arange(N), labels] = 1.0
    return y_oh

def softmax(logits):
    """数值稳定版 softmax，按最后一维"""
    # logits: (N, C)
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    return exp / (sum_exp + 1e-12)

def cross_entropy_loss(probs, y_onehot):
    """
    probs: (N, C) softmax 概率
    y_onehot: (N, C)
    """
    eps = 1e-12
    log_p = np.log(probs + eps)
    loss = -np.sum(y_onehot * log_p, axis=1)  # (N,)
    return np.mean(loss)

def accuracy(probs, y_true):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)

# 把标签转成 one-hot（后面计算 loss 用）
y_train_oh = one_hot(y_train, classes_cnt)
y_val_oh   = one_hot(y_val,   classes_cnt)
y_test_oh  = one_hot(y_test,  classes_cnt)

# ============================================
# 3. 定义 MLP 参数：75 → 64 → classes_cnt
# ============================================
rng = np.random.default_rng(seed=0)

hidden_dim = 64

# He 初始化 / Xavier 初始化都可以，这里简单一点：
W1 = rng.normal(loc=0.0, scale=1.0 / np.sqrt(input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
b1 = np.zeros((hidden_dim,), dtype=np.float32)

W2 = rng.normal(loc=0.0, scale=1.0 / np.sqrt(hidden_dim), size=(hidden_dim, classes_cnt)).astype(np.float32)
b2 = np.zeros((classes_cnt,), dtype=np.float32)

# ============================================
# 4. 训练超参数
# ============================================
LEARNING_RATE = 0.0015
EPOCHS = 50
BATCH_SIZE = 1

N_train = X_train.shape[0]

print("Start training (NumPy MLP)")

for epoch in range(1, EPOCHS + 1):
    # ---- 4.1 打乱训练样本顺序 ----
    indices = rng.permutation(N_train)
    X_train_shuffled = X_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]
    y_train_shuffled = y_train[indices]

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # ---- 4.2 mini-batch 训练 ----
    for start in range(0, N_train, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N_train)
        xb = X_train_shuffled[start:end]        # (B, 75)
        yb_oh = y_train_oh_shuffled[start:end]  # (B, C)
        yb = y_train_shuffled[start:end]        # (B,)

        B = xb.shape[0]  # 实际 batch 大小

        # ---------- forward ----------
        # layer1: z1 = xW1 + b1, h1 = ReLU(z1)
        z1 = xb @ W1 + b1          # (B, 64)
        h1 = np.maximum(z1, 0.0)   # ReLU

        # layer2: z2 = h1W2 + b2, softmax
        z2 = h1 @ W2 + b2          # (B, C)
        probs = softmax(z2)        # (B, C)

        # loss & acc
        loss = cross_entropy_loss(probs, yb_oh)  # 标量
        total_loss += loss * B
        total_correct += np.sum(np.argmax(probs, axis=1) == yb)
        total_seen += B

        # ---------- backward ----------
        # dL/dz2 = (probs - y_onehot) / B
        dz2 = (probs - yb_oh) / B               # (B, C)

        # dW2 = h1^T @ dz2, db2 = sum(dz2)
        dW2 = h1.T @ dz2                        # (64, C)
        db2 = np.sum(dz2, axis=0)               # (C,)

        # dz1 = dz2 @ W2^T
        dz1 = dz2 @ W2.T                        # (B, 64)
        # ReLU 导数
        dz1[z1 <= 0] = 0.0

        # dW1 = x^T @ dz1, db1 = sum(dz1)
        dW1 = xb.T @ dz1                         # (75, 64)
        db1 = np.sum(dz1, axis=0)               # (64,)

        # ---------- SGD 更新 ----------
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2
        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1

    train_loss = total_loss / total_seen
    train_acc = total_correct / total_seen

    # ============================================
    # 5. 在 val 集上评估
    # ============================================
    # forward 计算 val
    z1_val = X_val @ W1 + b1
    h1_val = np.maximum(z1_val, 0.0)
    z2_val = h1_val @ W2 + b2
    probs_val = softmax(z2_val)

    val_loss = cross_entropy_loss(probs_val, y_val_oh)
    val_acc = accuracy(probs_val, y_val)

    print(f"Epoch {epoch:03d} | "
          f"Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
          f"Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# ============================================
# 6. Test 集评估
# ============================================
z1_test = X_test @ W1 + b1
h1_test = np.maximum(z1_test, 0.0)
z2_test = h1_test @ W2 + b2
probs_test = softmax(z2_test)

test_loss = cross_entropy_loss(probs_test, y_test_oh)
test_acc = accuracy(probs_test, y_test)

print("\nFinal Test:")
print(f"  test_loss = {test_loss:.4f}")
print(f"  test_acc  = {test_acc:.3f}")
