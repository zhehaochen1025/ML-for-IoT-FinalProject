import re
import matplotlib.pyplot as plt
import os

# 方法1：直接把日志粘贴在这里（适合少量数据）
# raw_data = """
# Epoch count (training count): 46
# Accuracy after local training:
# Training Accuracy: 0.84
# Validation Accuracy: 0.82
# Test Accuracy: 0.72
# Epoch count (training count): 47
# Accuracy after local training:
# Training Accuracy: 0.89
# Validation Accuracy: 0.87
# Test Accuracy: 0.92
# Epoch count (training count): 48
# Accuracy after local training:
# Training Accuracy: 0.80
# Validation Accuracy: 0.76
# Test Accuracy: 0.79
# Epoch count (training count): 49
# Accuracy after local training:
# Training Accuracy: 0.87
# Validation Accuracy: 0.92
# Test Accuracy: 0.77
# Epoch count (training count): 50
# Accuracy after local training:
# Training Accuracy: 0.81
# Validation Accuracy: 0.82
# Test Accuracy: 0.81
# """

# 方法2：如果数据在文件里（例如 log.txt），取消下面两行的注释，注释掉上面的 raw_data
# 获取当前脚本所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接成文件的完整路径
file_path = os.path.join(script_dir, 'log_fje.txt')

# 使用完整路径打开文件
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = f.read()

def parse_and_plot(text):
    # 使用正则表达式提取数值
    # 提取 Epoch
    epochs = [int(x) for x in re.findall(r"Epoch count \(training count\): (\d+)", text)]
    
    # 提取各项 Accuracy
    train_acc = [float(x) for x in re.findall(r"Training Accuracy: ([\d\.]+)", text)]
    val_acc = [float(x) for x in re.findall(r"Validation Accuracy: ([\d\.]+)", text)]
    test_acc = [float(x) for x in re.findall(r"Test Accuracy: ([\d\.]+)", text)]

    # 检查数据长度是否一致，避免绘图报错
    if not (len(epochs) == len(train_acc) == len(val_acc) == len(test_acc)):
        print(f"警告: 数据解析数量不一致!")
        print(f"Epochs: {len(epochs)}, Train: {len(train_acc)}, Val: {len(val_acc)}, Test: {len(test_acc)}")
        # 截取最小长度以保证能画图
        min_len = min(len(epochs), len(train_acc), len(val_acc), len(test_acc))
        epochs = epochs[:min_len]
        train_acc = train_acc[:min_len]
        val_acc = val_acc[:min_len]
        test_acc = test_acc[:min_len]

    # 开始绘图
    plt.figure(figsize=(10, 6)) # 设置图片大小
    
    plt.plot(epochs, train_acc, marker='o', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, marker='s', label='Validation Accuracy', linewidth=2)
    plt.plot(epochs, test_acc, marker='^', label='Test Accuracy', linewidth=2)

    plt.title('Training Performance Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7) # 添加网格
    plt.legend(fontsize=12) # 显示图例
    
    # 自动调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parse_and_plot(raw_data)