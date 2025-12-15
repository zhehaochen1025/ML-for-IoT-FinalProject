import numpy as np

# ============================================
# 1. 读取 .npz 数据
# ============================================
npz_path = "jiuenfeng_imu_75feat_minmax_norm.npz"  # TODO: 换成你自己的 .npz 文件名
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

# 类别数
classes_cnt = int(len(np.unique(y_train)))
first_layer_input_cnt = X_train.shape[1]  # 应该是 75
print("first_layer_input_cnt =", first_layer_input_cnt)
print("classes_cnt            =", classes_cnt)

# ============================================
# 2. 与 Arduino 相同的 Min-Max 归一化
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
# 3. 定义 MLP，结构与 Arduino 相同：75 → 64 → classes_cnt
# ============================================

LEARNING_RATE = 0.05    # 和 Arduino 版一样
EPOCH = 1000                 # 训练轮数
batch_size = 32            # 你可以改小/改大

D_in = first_layer_input_cnt
H = 64
D_out = classes_cnt

rng = np.random.default_rng(seed=0)

# Xavier/He 风格初始化
W1 = (rng.random((D_in, H)).astype(np.float32) - 0.5) * 2 / np.sqrt(D_in)
b1 = np.zeros((1, H), dtype=np.float32)

W2 = (rng.random((H, D_out)).astype(np.float32) - 0.5) * 2 / np.sqrt(H)
b2 = np.zeros((1, D_out), dtype=np.float32)

def relu(x):
    return np.maximum(x, 0.0)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(logits):
    # 数值稳定
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(X):
    """
    X: (N, D_in)
    return:
      z1, h1, logits, y_pred
    """
    z1 = X @ W1 + b1   # (N, H)
    h1 = relu(z1)      # (N, H)
    logits = h1 @ W2 + b2  # (N, D_out)
    y_pred = softmax(logits)
    return z1, h1, logits, y_pred

def cross_entropy(y_pred, y_true):
    """
    y_pred: (N, C) softmax prob
    y_true: (N,) int labels
    """
    N = y_pred.shape[0]
    # 取出正确类的概率
    p = y_pred[np.arange(N), y_true]
    # 避免 log(0)
    p = np.clip(p, 1e-9, 1.0)
    return -np.mean(np.log(p))

def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(pred_labels == y_true)

# ============================================
# 4. 训练循环
# ============================================

N_train = X_train_n.shape[0]
indices = np.arange(N_train)

for epoch in range(1, EPOCH + 1):
    # 打乱
    rng.shuffle(indices)
    X_train_n = X_train_n[indices]
    y_train   = y_train[indices]

    # mini-batch 训练
    for start in range(0, N_train, batch_size):
        end = start + batch_size
        xb = X_train_n[start:end]
        yb = y_train[start:end]
        if xb.shape[0] == 0:
            continue

        # forward
        z1, h1, logits, y_pred = forward(xb)

        # loss (不用反传时也可以算)
        # loss = cross_entropy(y_pred, yb)

        # backward (交叉熵 + softmax)
        N = xb.shape[0]
        dlogits = y_pred.copy()
        dlogits[np.arange(N), yb] -= 1.0   # dL/dlogits = (p - y_onehot)
        dlogits /= N

        # W2, b2 梯度
        dW2 = h1.T @ dlogits            # (H, N)@(N, C) = (H, C)
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        # 传播到隐藏层
        dh1 = dlogits @ W2.T           # (N, C)@(C, H) = (N, H)
        dz1 = dh1 * relu_grad(z1)      # (N, H)

        # W1, b1 梯度
        dW1 = xb.T @ dz1               # (D_in, N)@(N, H) = (D_in, H)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 参数更新
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2
        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1

    # 每个 epoch 结束评估一次 train / val
    _, _, _, y_train_pred = forward(X_train_n)
    train_loss = cross_entropy(y_train_pred, y_train)
    train_acc  = accuracy(y_train_pred, y_train)

    _, _, _, y_val_pred = forward(X_val_n)
    val_loss = cross_entropy(y_val_pred, y_val)
    val_acc  = accuracy(y_val_pred, y_val)

    print(f"Epoch {epoch:02d} | "
          f"Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
          f"Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# ============================================
# 5. 最终在 test 集上评估
# ============================================

_, _, _, y_test_pred = forward(X_test_n)
test_loss = cross_entropy(y_test_pred, y_test)
test_acc  = accuracy(y_test_pred, y_test)

print("\nFinal Test:")
print(f"  test_loss = {test_loss:.4f}")
print(f"  test_acc  = {test_acc:.3f}")
