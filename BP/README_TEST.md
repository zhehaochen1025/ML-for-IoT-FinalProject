# 测试集推理和混淆矩阵生成指南

本目录包含用于测试集推理和评估的工具。

## 文件说明

- `BP_test_inference.ino`: 在测试集上运行推理的 Arduino 程序
- `generate_confusion_matrix.py`: 生成混淆矩阵的 Python 脚本
- `BP.ino`: 训练程序（需要先运行这个获取模型权重）

## 使用步骤

### 第一步：训练模型并获取权重

1. 运行 `BP.ino` 训练模型
2. 训练完成后，从串口输出中复制权重数组（格式如：`const float trained_weights[] = {...}`）
3. 保存权重数组供下一步使用

### 第二步：配置测试推理程序

1. 打开 `BP_test_inference.ino`
2. 将第一步复制的权重数组粘贴到 `trained_weights` 数组中（替换注释部分）
3. 上传到 Arduino

### 第三步：运行测试推理

1. 打开串口监视器（波特率：9600）
2. 程序会自动运行测试集推理
3. 输出格式示例：
   ```
   0,0,0,circle,circle
   1,0,1,circle,other
   2,0,0,circle,circle
   ...
   ```
   格式：`样本索引,真实标签,预测标签,真实标签名称,预测标签名称`

### 第四步：生成混淆矩阵

有两种方式运行 Python 脚本：

#### 方式1：从文件读取（推荐）

1. 将串口输出保存到文件（例如：`result.txt`）
2. 运行脚本：
   ```bash
   python3 generate_confusion_matrix.py result.txt
   ```

#### 方式2：从标准输入读取

1. 运行脚本：
   ```bash
   python3 generate_confusion_matrix.py
   ```
2. 粘贴串口输出内容
3. 按 `Ctrl+D` (Mac/Linux) 或 `Ctrl+Z` (Windows) 结束输入

## 输出说明

Python 脚本会生成：

1. **文本混淆矩阵**：在终端中显示的文本格式混淆矩阵
2. **分类报告**：包含精确率、召回率、F1分数等指标
3. **可视化混淆矩阵**：保存为 `confusion_matrix.png` 图片文件

## 类别映射

- 0: circle（圆形手势）
- 1: other（其他/噪声）
- 2: peak（峰值手势）
- 3: wave（波浪手势）

## 依赖库

Python 脚本需要以下库：
```bash
pip install numpy matplotlib scikit-learn
```

## 注意事项

- 确保 `BP_test_inference.ino` 中的网络结构（`NN_def`）与训练时一致
- 权重数组必须完整，否则推理会失败
- 如果权重数组为空，程序会停止并提示错误



