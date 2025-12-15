from mpi4py import MPI
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# MPI 初始化
# ============================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size <= 1:
    raise RuntimeError("需要至少 2 个进程：rank 0 做 server，其余 rank 做 clients。")

# ============================================
# 1. 读取多个 .npz，每个文件看作一个“原始客户端数据源”
#    - rank 0 负责列出文件名并广播
#    - rank 1..(size-1) 各自接收并加载自己负责的那部分数据
# ============================================

if rank == 0:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    npz_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".npz")
    )
    if not npz_files:
        raise RuntimeError(f"在 {DATA_DIR} 没有找到 .npz 文件。")

    print("发现的 npz 文件（将给不同 worker 使用）:")
    for i, f in enumerate(npz_files):
        print(f"  [{i}] {f}")
else:
    DATA_DIR = None
    npz_files = None

# 广播 DATA_DIR 和 npz_files 给所有进程
DATA_DIR = comm.bcast(DATA_DIR, root=0)
npz_files = comm.bcast(npz_files, root=0)

num_sources = len(npz_files)
num_workers = size - 1  # rank 1..size-1 是 worker

if rank == 0:
    print(f"\n总共有 {num_sources} 个 .npz 数据源，"
          f"{num_workers} 个 worker 进程用于训练。")

# ============================================
# 2. 每个 worker 决定自己负责哪些 .npz（按取模分配）
#    例如：有 3 个 worker, 5 个 npz:
#    - worker(rank=1) 负责 index: 0, 3 ...
#    - worker(rank=2) 负责 index: 1, 4 ...
#    - worker(rank=3) 负责 index: 2 ...
# ============================================

def load_npz_for_indices(indices):
    """把若干 npz 的 train/val/test 拼接在一起。"""
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []

    for idx in indices:
        fname = npz_files[idx]
        path = os.path.join(DATA_DIR, fname)
        data = np.load(path)

        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        X_val   = data["X_val"].astype(np.float32)
        y_val   = data["y_val"].astype(np.int64)
        X_test  = data["X_test"].astype(np.float32)
        y_test  = data["y_test"].astype(np.int64)

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    if not X_train_list:
        # 这个 worker 没有被分配数据
        return None

    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val_all = np.concatenate(y_val_list, axis=0)
    X_test_all = np.concatenate(X_test_list, axis=0)
    y_test_all = np.concatenate(y_test_list, axis=0)

    return (X_train_all, y_train_all,
            X_val_all,   y_val_all,
            X_test_all,  y_test_all)

# rank 0 只做“服务器”，不持有本地训练数据
if rank == 0:
    local_data = None
else:
    # worker 编号从 0..num_workers-1，对应 rank-1
    worker_id = rank - 1
    my_indices = [i for i in range(num_sources) if i % num_workers == worker_id]
    if my_indices:
        print(f"[Rank {rank}] 负责 npz index: {my_indices}")
    else:
        print(f"[Rank {rank}] 没有被分配任何 npz (可能 npz 数量 < worker 数量)")

    local_data = load_npz_for_indices(my_indices)

# ============================================
# 3. rank 0 需要一个“全局验证 / 测试集”
#    简单做法：rank 0 自己加载所有 npz 拼接。
# ============================================

if rank == 0:
    # 重新加载所有 npz，在 server 上做全局 eval
    val_X_list, val_y_list = [], []
    test_X_list, test_y_list = [], []

    input_dim = None
    classes_cnt = None

    for fname in npz_files:
        path = os.path.join(DATA_DIR, fname)
        data = np.load(path)

        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        X_val   = data["X_val"].astype(np.float32)
        y_val   = data["y_val"].astype(np.int64)
        X_test  = data["X_test"].astype(np.float32)
        y_test  = data["y_test"].astype(np.int64)

        if input_dim is None:
            input_dim = X_train.shape[1]
        else:
            assert X_train.shape[1] == input_dim

        if classes_cnt is None:
            classes_cnt = int(len(np.unique(y_train)))
        else:
            assert classes_cnt == int(len(np.unique(y_train)))

        val_X_list.append(X_val)
        val_y_list.append(y_val)
        test_X_list.append(X_test)
        test_y_list.append(y_test)

    X_val_global = np.concatenate(val_X_list, axis=0)
    y_val_global = np.concatenate(val_y_list, axis=0)
    X_test_global = np.concatenate(test_X_list, axis=0)
    y_test_global = np.concatenate(test_y_list, axis=0)

    print("\n[Server] Global Val:", X_val_global.shape)
    print("[Server] Global Test:", X_test_global.shape)
    print("[Server] input_dim   =", input_dim)
    print("[Server] classes_cnt =", classes_cnt)

else:
    # 其它 rank 不需要这两个，稍后从 server 广播
    X_val_global = None
    y_val_global = None
    X_test_global = None
    y_test_global = None
    input_dim = None
    classes_cnt = None

# 广播 input_dim / classes_cnt
input_dim = comm.bcast(input_dim, root=0)
classes_cnt = comm.bcast(classes_cnt, root=0)

# ============================================
# 4. 定义模型和评估函数
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if rank == 0:
    print("\nUsing device:", device)

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

def eval_loader(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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

# server 构造 global val/test loader
if rank == 0:
    val_ds_global = TensorDataset(
        torch.from_numpy(X_val_global),
        torch.from_numpy(y_val_global)
    )
    test_ds_global = TensorDataset(
        torch.from_numpy(X_test_global),
        torch.from_numpy(y_test_global)
    )
    val_loader_global = DataLoader(val_ds_global, batch_size=64, shuffle=False)
    test_loader_global = DataLoader(test_ds_global, batch_size=64, shuffle=False)

# ============================================
# 5. 联邦学习超参数
# ============================================

NUM_ROUNDS   = 50      # 联邦通信轮数
LOCAL_EPOCHS = 1       # 每轮中，每个 worker 本地训练 epoch 数
BATCH_SIZE   = 1
LR           = 0.0015

if rank == 0:
    print(f"\nStart Federated Training with MPI: {num_workers} workers")
    print(f"NUM_ROUNDS={NUM_ROUNDS}, LOCAL_EPOCHS={LOCAL_EPOCHS}, "
          f"LR={LR}, BATCH_SIZE={BATCH_SIZE}\n")

# 初始化全局模型（所有 rank 都要有同构模型）
global_model = IMUNet(input_dim, 64, classes_cnt).to(device)

criterion = nn.CrossEntropyLoss()

# ============================================
# 6. FedAvg + MPI 训练循环
# ============================================

for rnd in range(1, NUM_ROUNDS + 1):

    if rank == 0:
        print(f"\n===== Round {rnd}/{NUM_ROUNDS} =====")

    # ---- step 1: server 广播当前全局权重 ----
    # 将 state_dict 转成 CPU tensor，方便 pickling
    if rank == 0:
        global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    else:
        global_state = None

    global_state = comm.bcast(global_state, root=0)

    # 每个 rank 用收到的全局权重更新自己的模型
    global_model.load_state_dict(global_state)

    # ---- step 2: worker 本地训练，上传本地权重和样本数 ----
    if rank == 0:
        # server 不训练，只负责聚合
        local_result = None
    else:
        if local_data is None:
            # 这个 worker 没有数据
            local_result = None
        else:
            X_train_all, y_train_all, _, _, _, _ = local_data

            train_ds = TensorDataset(
                torch.from_numpy(X_train_all),
                torch.from_numpy(y_train_all)
            )
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True
            )

            # 本地模型：从全局权重开始
            local_model = IMUNet(input_dim, 64, classes_cnt).to(device)
            local_model.load_state_dict(global_state)
            optimizer = optim.SGD(local_model.parameters(), lr=LR)

            local_model.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    logits = local_model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

            # 把本地权重 + 样本数打包
            local_state = {k: v.cpu() for k, v in local_model.state_dict().items()}
            local_num_samples = len(train_ds)
            local_result = (local_state, local_num_samples)

            print(f"[Rank {rank}] local training done, samples = {local_num_samples}")

    # 所有 rank 把 local_result gather 到 server
    all_results = comm.gather(local_result, root=0)

    # ---- step 3: server 做 FedAvg 聚合并更新 global_model ----
    if rank == 0:
        # all_results 长度 = size，index 0 对应 server 自己，是 None
        agg_state = None
        total_samples = 0.0

        for r in range(1, size):
            item = all_results[r]
            if item is None:
                continue
            state_r, n_r = item
            total_samples += n_r
            if agg_state is None:
                agg_state = {}
                for k in state_r.keys():
                    agg_state[k] = state_r[k] * n_r
            else:
                for k in state_r.keys():
                    agg_state[k] += state_r[k] * n_r

        if agg_state is None:
            print("[Server] 没有收到任何客户端参数，可能所有 worker 都没有数据。")
        else:
            for k in agg_state.keys():
                agg_state[k] /= total_samples

            global_model.load_state_dict(agg_state)

            # 在全局验证集上评估
            val_loss, val_acc = eval_loader(global_model, val_loader_global, device)
            print(f"[Server] After Round {rnd}: "
                  f"Global Val loss={val_loss:.4f}, acc={val_acc:.3f}")

# ============================================
# 7. 最终 server 在全局测试集上评估
# ============================================

if rank == 0:
    test_loss, test_acc = eval_loader(global_model, test_loader_global, device)
    print("\n===== Final Global Test (Server) =====")
    print(f"Test loss = {test_loss:.4f}")
    print(f"Test acc  = {test_acc:.3f}")
