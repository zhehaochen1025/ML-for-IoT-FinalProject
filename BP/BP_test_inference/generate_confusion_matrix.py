#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混淆矩阵生成脚本

功能：
1. 从串口输出或文本文件读取推理结果
2. 解析真实标签和预测标签
3. 生成混淆矩阵并可视化

使用方法：
1. 运行 BP_test_inference.ino 获取推理结果
2. 将串口输出保存到文件，或直接粘贴到 result.txt
3. 运行此脚本：python generate_confusion_matrix.py

或者：
1. 直接运行脚本，然后粘贴串口输出内容，按 Ctrl+D (Linux/Mac) 或 Ctrl+Z (Windows) 结束输入
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 类别名称（与Arduino代码中的映射一致）
class_names = ["circle", "other", "peak", "wave"]
num_classes = len(class_names)


def parse_results(input_data):
    """
    解析推理结果
    
    参数:
        input_data: 字符串，包含串口输出的文本
        
    返回:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
    """
    true_labels = []
    pred_labels = []
    
    # 匹配格式：索引,真实标签,预测标签,真实名称,预测名称
    # 例如：0,0,0,circle,circle
    pattern = r'^\d+,(\d+),(\d+),'
    
    for line in input_data.split('\n'):
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or '准确率' in line or '推理' in line or '开始' in line or '加载' in line or '权重' in line or '测试集' in line:
            continue
            
        match = re.match(pattern, line)
        if match:
            true_label = int(match.group(1))
            pred_label = int(match.group(2))
            true_labels.append(true_label)
            pred_labels.append(pred_label)
    
    return true_labels, pred_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_mbyh_dbyh.png'):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        class_names: 类别名称列表
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=f'Confusion Matrix (Accuracy: {accuracy:.2%})',
           ylabel='Truth',
           xlabel='Prediction(model_byh + data_byh)')
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存到: {save_path}")
    
    # 显示图片
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """
    打印分类报告
    """
    print("\n" + "="*60)
    print("分类报告")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("="*60)


def main():
    """主函数"""
    print("混淆矩阵生成工具")
    print("="*60)
    
    # 尝试从文件读取
    input_data = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                input_data = f.read()
            print(f"从文件读取: {file_path}")
        except FileNotFoundError:
            print(f"错误：文件 {file_path} 不存在")
            sys.exit(1)
    else:
        # 从标准输入读取
        print("请粘贴串口输出内容（按 Ctrl+D (Mac/Linux) 或 Ctrl+Z (Windows) 结束输入）：")
        print("-"*60)
        try:
            input_data = sys.stdin.read()
        except EOFError:
            print("\n错误：没有读取到输入数据")
            sys.exit(1)
    
    # 解析结果
    print("\n正在解析推理结果...")
    true_labels, pred_labels = parse_results(input_data)
    
    if len(true_labels) == 0:
        print("错误：未找到任何有效的推理结果！")
        print("请确保输入格式正确：")
        print("  样本索引,真实标签,预测标签,真实名称,预测名称")
        print("  例如：0,0,0,circle,circle")
        sys.exit(1)
    
    print(f"成功解析 {len(true_labels)} 个样本")
    
    # 计算并打印统计信息
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n测试集准确率: {accuracy:.2%} ({sum(t == p for t, p in zip(true_labels, pred_labels))}/{len(true_labels)})")
    
    # 打印混淆矩阵（文本格式）
    print("\n混淆矩阵（文本格式）：")
    print("-"*60)
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    
    # 打印表头
    header = "Truth\\Prediction" + "".join([f"{name:>10}" for name in class_names])
    print(header)
    print("-"*len(header))
    
    # 打印每一行
    for i in range(num_classes):
        row = f"{class_names[i]:>10}" + "".join([f"{cm[i][j]:>10}" for j in range(num_classes)])
        print(row)
    
    # 打印分类报告
    print_classification_report(true_labels, pred_labels, class_names)
    
    # 绘制混淆矩阵
    try:
        print("\n正在生成混淆矩阵图表...")
        plot_confusion_matrix(true_labels, pred_labels, class_names)
    except Exception as e:
        print(f"警告：无法生成图表 ({e})")
        print("可能原因：matplotlib 未正确安装或显示环境不可用")
        print("混淆矩阵已以文本格式显示在上面")


if __name__ == "__main__":
    main()

