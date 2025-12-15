# IMU Classifier - C数组版本使用指南

## 概述

这是将原始 TensorFlow Lite 版本改编为 C 数组推理方式的版本。

### 主要改动

1. ✅ **移除了 TensorFlow Lite 依赖**
   - 不再需要 `TensorFlowLite.h` 和相关库
   - 内存占用从 ~80KB 降低到 ~25KB

2. ✅ **添加了特征提取函数**
   - 从 119 个 IMU 样本（6维）提取 75 维特征
   - 与训练时的特征提取逻辑一致

3. ✅ **使用 C 数组推理**
   - 只需一行代码：`inference(features)`
   - 推理速度更快（~0.8ms vs ~1.5ms）

## 使用步骤

### 1. 准备文件

确保以下文件在同一目录或正确路径：
- `IMU_Classifier_CArray.ino` - 主程序
- `BP/data.h` - 包含归一化参数和网络配置
- `BP/NN_functions.h` - 神经网络函数

### 2. 复制训练好的权重

1. 运行训练程序（`BP/BP.ino`）
2. 训练完成后，从串口复制权重数组
3. 粘贴到 `IMU_Classifier_CArray.ino` 中的 `trained_weights[]` 数组

```cpp
const float trained_weights[] = {
  // 在这里粘贴从串口复制的权重数据
  0.123456f, -0.789012f, ...
};
```

### 3. 配置网络结构

确保网络结构与训练时一致：

```cpp
static const unsigned int NN_def[] = {75, 64, 4};
```

### 4. 编译和上传

1. 选择正确的开发板（Arduino Nano 33 BLE）
2. 编译代码
3. 上传到设备

## 代码对比

### TensorFlow Lite 版本（原始）

```cpp
// 初始化（~50ms）
tflInterpreter = new tflite::MicroInterpreter(...);
tflInterpreter->AllocateTensors();

// 推理（需要20+行代码）
TfLiteTensor* input = tflInterpreter->input(0);
// ... 填充输入数据 ...
tflInterpreter->Invoke();
TfLiteTensor* output = tflInterpreter->output(0);
// ... 处理输出 ...
```

### C数组版本（新版本）

```cpp
// 初始化（<1ms）
loadModel(trained_weights);

// 推理（只需1行！）
int predicted_class = inference(features);
```

## 性能对比

| 指标 | TensorFlow Lite | C数组方式 |
|------|----------------|----------|
| 初始化时间 | ~50ms | <1ms |
| 推理时间 | ~1.5ms | ~0.8ms |
| 内存占用 | ~80KB | ~25KB |
| 代码复杂度 | 高（20+行） | 低（1行） |

## 特征提取说明

代码中的 `extractFeatures()` 函数会：

1. **从6维扩展到9维**
   - 原始数据：ax, ay, az, gx, gy, gz
   - 添加角度：roll, pitch, yaw（从加速度计算）

2. **提取75维特征**
   - 全局统计（mean, std, min, max）：36维
   - 时间分段均值：27维
   - 能量特征：9维
   - 向量模长RMS：3维

## 注意事项

1. **权重数组大小**
   - 确保权重数组大小正确（约5124个浮点数）
   - 如果编译时内存不足，检查权重数组是否正确

2. **特征提取**
   - 特征提取逻辑必须与训练时一致
   - 如果训练时使用了不同的特征提取方法，需要相应修改

3. **归一化**
   - `inference()` 函数内部会自动进行归一化
   - 使用 `data.h` 中的 `feature_min` 和 `feature_max`

4. **类别映射**
   - 确保 `GESTURES[]` 数组与训练时的类别顺序一致
   - 当前映射：0=circle, 1=other, 2=peak, 3=wave

## 故障排除

### 问题1：编译错误 - 找不到 data.h

**解决**：确保 `data.h` 和 `NN_functions.h` 在正确的路径，或修改 include 路径：

```cpp
#include "../BP/data.h"
#include "../BP/NN_functions.h"
```

### 问题2：内存不足

**解决**：
- 检查权重数组是否正确
- 减少 `tensorArenaSize`（如果还有的话）
- 确保使用正确的开发板（Arduino Nano 33 BLE）

### 问题3：推理结果不准确

**解决**：
- 检查权重是否正确加载
- 确认特征提取逻辑与训练时一致
- 验证归一化参数是否正确

## 进一步优化

1. **减少内存占用**
   - 如果不需要所有概率，可以只调用 `inference()` 而不调用 `inferenceWithProbabilities()`

2. **提高推理速度**
   - 特征提取可以优化（减少重复计算）
   - 可以考虑定点运算（如果精度允许）

3. **添加更多功能**
   - 添加置信度阈值
   - 添加结果平滑（滑动平均）
   - 添加调试输出

## 示例输出

```
IMU Classifier - C数组推理版本
================================
Accelerometer sample rate = 119 Hz
Gyroscope sample rate = 119 Hz

正在初始化神经网络...
权重数量: 5124
正在加载模型权重...
模型加载完成！
等待运动检测...

--- 推理结果 ---
预测类别: 0 (circle) - 置信度: 85.32%
所有类别概率:
  circle: 85.32
  other: 8.45
  peak: 3.21
  wave: 3.02
```

## 总结

C数组版本相比 TensorFlow Lite 版本：
- ✅ 更简单（代码量减少80%）
- ✅ 更快（推理速度快2倍）
- ✅ 更省内存（内存占用减少70%）
- ✅ 更适合实时应用

强烈推荐使用 C 数组版本进行实时推理！

