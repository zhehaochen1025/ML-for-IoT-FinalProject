# fed_training_mpi.py
import os
import numpy as np
from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ===========================
# 0. MPI & 全局配置
# ===========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 这里列出每个客户端的数据文件（都在 DATA_DIR 下面）
# 确保 len(CLIENT_FILES) = size - 1
CLIENT_FILES = [
    "czh_imu_75feat_minmax_norm.npz",
    "fje_imu_75feat_minmax_norm.npz",
    #"byh_imu_75feat_minmax_norm.npz",
    # 如果有更多客户端，就继续加:
    # "client3_imu_75feat_minmax_norm.npz",
    # ...
]

NUM_CLIENTS = len(CLIENT_FILES)

if rank == 0:
    print(f"[Server] MPI size = {size}, NUM_CLIENTS = {NUM_CLIENTS}")
if size != NUM_CLIENTS + 1:
    if rank == 0:
        print(f"❌ 进程数不匹配！需要 size = NUM_CLIENTS + 1 = {NUM_CLIENTS + 1}，"
              f"但现在 size = {size}")
        print("请用: mpiexec -n <1+客户端数> python fed_training_mpi.py")
    MPI.Finalize()
    raise SystemExit

# 联邦超参数
NUM_ROUNDS   = 50      # 联邦通信轮数
LOCAL_EPOCHS = 1       # 每轮每个客户端本地训练 epoch 数
BATCH_SIZE   = 1       # 和板子上的实现保持一致
LR           = 0.0015  # 学习率

device = torch.device("cpu")  # 建议 MPI 下统一用 CPU，避免 GPU 资源冲突


# ===========================
# 1. 工具函数
# ===========================
def load_npz_dataset(path):
    data = np.load(path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.int64)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int64)

    # feature_min / feature_max 是给板子用的，这里不再归一化
    feature_min = data["feature_min"]
    feature_max = data["feature_max"]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_min, feature_max


class IMUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # logits
        return x


def get_state_dict(model: nn.Module):
    # 拷贝一份 CPU 上的参数（避免引用问题）
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict(model: nn.Module, state_dict):
    model.load_state_dict(state_dict)


def average_state_dicts(state_dict_list):
    """FedAvg: 对多个客户端的 state_dict 做简单平均"""
    avg_state = {}
    keys = state_dict_list[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k] for sd in state_dict_list], dim=0)  # (num_clients, ...)
        avg_state[k] = stacked.mean(dim=0)
    return avg_state


def build_dataloader(X, y, batch_size, shuffle):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def eval_model(model, loader):
    criterion = nn.CrossEntropyLoss()
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


# ===========================
# 2. 每个进程加载自己需要的数据
# ===========================
if rank == 0:
    # Server: 用所有客户端的验证集 / 测试集做评估
    all_val_X_list = []
    all_val_y_list = []
    all_test_X_list = []
    all_test_y_list = []

    input_dim = None
    num_classes = None

    for fname in CLIENT_FILES:
        path = os.path.join(DATA_DIR, fname)
        Xtr, ytr, Xv, yv, Xt, yt, fmin, fmax = load_npz_dataset(path)

        if input_dim is None:
            input_dim = Xtr.shape[1]
            num_classes = int(len(np.unique(ytr)))
        else:
            # 简单 sanity check
            assert Xtr.shape[1] == input_dim, "各客户端 input_dim 不一致！"
            assert len(np.unique(ytr)) == num_classes, "各客户端类别数不一致！"

        all_val_X_list.append(Xv)
        all_val_y_list.append(yv)
        all_test_X_list.append(Xt)
        all_test_y_list.append(yt)

    X_val_global = np.concatenate(all_val_X_list, axis=0)
    y_val_global = np.concatenate(all_val_y_list, axis=0)

    X_test_global = np.concatenate(all_test_X_list, axis=0)
    y_test_global = np.concatenate(all_test_y_list, axis=0)

    val_loader = build_dataloader(X_val_global, y_val_global,
                                  batch_size=64, shuffle=False)
    test_loader = build_dataloader(X_test_global, y_test_global,
                                   batch_size=64, shuffle=False)

    print(f"[Server] Global val size = {len(X_val_global)}, "
          f"Global test size = {len(X_test_global)}")
    print(f"[Server] input_dim = {input_dim}, num_classes = {num_classes}")

else:
    # Client: 每个 rank 对应一个文件
    client_idx = rank - 1  # 0..NUM_CLIENTS-1
    fname = CLIENT_FILES[client_idx]
    path = os.path.join(DATA_DIR, fname)
    X_train, y_train, X_val_local, y_val_local, X_test_local, y_test_local, fmin, fmax = load_npz_dataset(path)

    input_dim = X_train.shape[1]
    num_classes = int(len(np.unique(y_train)))

    train_loader = build_dataloader(X_train, y_train,
                                    batch_size=BATCH_SIZE, shuffle=True)

    print(f"[Client {rank}] file = {fname}, "
          f"train_size = {len(X_train)}, val_size = {len(X_val_local)}, "
          f"test_size = {len(X_test_local)}")

# 把 input_dim 和 num_classes 从一个进程广播到所有
input_dim = comm.bcast(input_dim if rank == 0 else None, root=0)
num_classes = comm.bcast(num_classes if rank == 0 else None, root=0)

# ===========================
# 3. 每个进程创建自己的模型 & optimizer
# ===========================
model = IMUNet(input_dim, 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

if rank == 0:
    print("[Server] 模型结构:")
    print(model)

# ===========================
# 4. 联邦训练主循环 (FedAvg)
# ===========================
history_val_acc = []
history_val_loss = []

for rnd in range(1, NUM_ROUNDS + 1):
    if rank == 0:
        print(f"\n========== Round {rnd}/{NUM_ROUNDS} ==========")

    # ---- 4.1 广播当前全局模型权重 ----
    if rank == 0:
        global_state = get_state_dict(model)
    else:
        global_state = None

    global_state = comm.bcast(global_state, root=0)
    load_state_dict(model, global_state)

    # ---- 4.2 各客户端本地训练 ----
    if rank != 0:
        model.train()
        for local_ep in range(LOCAL_EPOCHS):
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

            avg_loss = total_loss / total
            acc = total_correct / total
        print(f"[Client {rank}] Round {rnd}: "
              f"local_train_loss={avg_loss:.4f}, acc={acc:.3f}")

        local_state = get_state_dict(model)
    else:
        local_state = None

    # ---- 4.3 server 收集所有客户端的权重并做 FedAvg ----
    all_states = comm.gather(local_state, root=0)

    if rank == 0:
        # all_states[0] 是 None，其它是客户端的 state_dict
        client_states = [st for st in all_states[1:] if st is not None]
        assert len(client_states) == NUM_CLIENTS

        new_global_state = average_state_dicts(client_states)
        load_state_dict(model, new_global_state)

        # ---- 4.4 在全局验证集上评估 ----
        val_loss, val_acc = eval_model(model, val_loader)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        print(f"[Server] After Round {rnd}: "
              f"Global Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# ===========================
# 5. 训练结束：server 上保存曲线 & 测试性能
# ===========================
if rank == 0:
    # 最终在全局 test 上评估
    test_loss, test_acc = eval_model(model, test_loader)
    print("\n===== Final Global Test (after FedAvg training) =====")
    print(f"  test_loss = {test_loss:.4f}")
    print(f"  test_acc  = {test_acc:.3f}")

    # 保存联邦训练曲线，供 plot_training_curves.py 使用
    fed_history_path = os.path.join(BASE_DIR, "data/fed_history.npz")
    np.savez(
        fed_history_path,
        val_acc=np.array(history_val_acc, dtype=np.float32),
        val_loss=np.array(history_val_loss, dtype=np.float32),
        num_rounds=np.array(NUM_ROUNDS, dtype=np.int32),
        num_clients=np.array(NUM_CLIENTS, dtype=np.int32),
    )
    print(f"\n已保存联邦训练历史到: {fed_history_path}")
