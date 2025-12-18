import re
import matplotlib.pyplot as plt
import os

# 同样，你可以把日志保存在 log.txt 里读取
# with open('log.txt', 'r', encoding='utf-8') as f:
#     log_data = f.read()

script_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接成文件的完整路径
file_path = os.path.join(script_dir, 'log_worker.txt')

# 使用完整路径打开文件
with open(file_path, 'r', encoding='utf-8') as f:
    log_data = f.read()

def parse_fl_log(text):
    data = []
    
    # 使用正则表达式分割每个 block (按 Epoch count 分割)
    # 也可以按 STARTING ITERATION 分割，看你的 log 哪个更完整
    blocks = re.split(r"(?=Epoch count \(training count\):)", text)

    for block in blocks:
        if "Epoch count" not in block:
            continue
            
        entry = {}
        
        # 1. 提取 Epoch
        epoch_match = re.search(r"Epoch count \(training count\): (\d+)", block)
        if not epoch_match: continue
        entry['epoch'] = int(epoch_match.group(1))

        # 2. 提取 Before Aggregation 数据
        before_match = re.search(r"Accuracy before aggregation:(.*?)((?=Accuracy after aggregation:)|(?=Epoch count)|$)", block, re.DOTALL)
        if before_match:
            content = before_match.group(1)
            entry['train_before'] = float(re.search(r"Training Accuracy: ([\d\.]+)", content).group(1))
            entry['val_before']   = float(re.search(r"Validation Accuracy: ([\d\.]+)", content).group(1))
            entry['test_before']  = float(re.search(r"Test Accuracy: ([\d\.]+)", content).group(1))
        
        # 3. 提取 After Aggregation 数据 (可能有些 iteration 还没有跑完或者没有这部分)
        after_match = re.search(r"Accuracy after aggregation:(.*?)((?=Epoch count)|$)", block, re.DOTALL)
        if after_match:
            content = after_match.group(1)
            # 使用 try-except 防止部分数据缺失导致报错
            try:
                entry['train_after'] = float(re.search(r"Training Accuracy: ([\d\.]+)", content).group(1))
                entry['val_after']   = float(re.search(r"Validation Accuracy: ([\d\.]+)", content).group(1))
                entry['test_after']  = float(re.search(r"Test Accuracy: ([\d\.]+)", content).group(1))
            except AttributeError:
                pass 

        data.append(entry)
    
    return data

def plot_curves(data):
    save_path = 'distributed_dnn_on_device/distributed_test/output/'
    epochs = [d['epoch'] for d in data]
    
    # 提取各列数据，如果某一行没有该数据，则填充 None (matplotlib 会自动跳过 None 的点)
    train_before = [d.get('train_before') for d in data]
    val_before   = [d.get('val_before') for d in data]
    test_before  = [d.get('test_before') for d in data]
    
    train_after  = [d.get('train_after') for d in data]
    val_after    = [d.get('val_after') for d in data]
    test_after   = [d.get('test_after') for d in data]

    plt.figure(figsize=(12, 7))

    # 绘制 "Before Aggregation" (实线)
    plt.plot(epochs, train_before, 'o-', label='Train (Before Agg)', color='tab:blue', linewidth=2)
    plt.plot(epochs, val_before,   's-', label='Val (Before Agg)',   color='tab:orange', linewidth=2)
    plt.plot(epochs, test_before,  '^-', label='Test (Before Agg)',  color='tab:green', linewidth=2)

    # 绘制 "After Aggregation" (虚线) - 过滤掉全是 None 的情况
    if any(x is not None for x in train_after):
        plt.plot(epochs, train_after, 'o--', label='Train (After Agg)', color='tab:blue', alpha=0.5)
        plt.plot(epochs, val_after,   's--', label='Val (After Agg)',   color='tab:orange', alpha=0.5)
        plt.plot(epochs, test_after,  '^--', label='Test (After Agg)',  color='tab:green', alpha=0.5)

    plt.title('Federated Training Metrics: Before vs After Aggregation', fontsize=16)
    plt.xlabel('Epoch / Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path + 'plot_fd_worker.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parsed_data = parse_fl_log(log_data)
    plot_curves(parsed_data)