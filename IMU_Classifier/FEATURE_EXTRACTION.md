# 特征提取说明

## 重要提示

**特征提取必须在推理之前完成，且必须与训练时的预处理逻辑完全一致！**

## 特征提取流程

### 1. 数据采集
- 从IMU采集119个样本
- 每个样本包含6维数据：`ax, ay, az, gx, gy, gz`
- 存储到 `imu_buffer[119][6]`

### 2. 扩展为9维数据
由于训练数据包含9维（包括角度），需要从6维IMU数据计算角度：

```cpp
// 从加速度计算 roll 和 pitch
roll = atan2(ay, az) * 180.0 / PI;
pitch = atan2(-ax, sqrt(ay*ay + az*az)) * 180.0 / PI;
yaw = 0.0;  // 无法单独计算，如果训练数据有yaw需要从陀螺仪积分
```

**注意**：如果训练数据使用了不同的角度计算方法，需要相应调整！

### 3. 提取75维特征

特征提取逻辑与 `data_prepare.ipynb` 中的 `extract_features()` 完全一致：

#### 步骤1：全局统计（36维）
对9个通道分别计算：
- **mean**：均值
- **std**：标准差 `sqrt(mean(x^2) - mean(x)^2)`
- **min**：最小值
- **max**：最大值

```python
# Python版本
means = v.mean(axis=0)  # 9维
stds  = v.std(axis=0)   # 9维
mins  = v.min(axis=0)   # 9维
maxs  = v.max(axis=0)   # 9维
# 总计：9 * 4 = 36维
```

#### 步骤2：时间分段均值（27维）
将119个样本分成3段，每段计算9个通道的均值：

```python
# Python版本
segments = np.array_split(v, 3, axis=0)  # 分成3段
for seg in segments:
    seg_mean = seg.mean(axis=0)  # 每段9维
# 总计：3 * 9 = 27维
```

#### 步骤3：能量特征（9维）
每个通道的均方值（能量）：

```python
# Python版本
energy = (v ** 2).mean(axis=0)  # 9维
```

#### 步骤4：向量模长均值（3维）
计算三个向量组的模长均值：
- **加速度模长**：`sqrt(ax^2 + ay^2 + az^2)` 的均值
- **陀螺仪模长**：`sqrt(gx^2 + gy^2 + gz^2)` 的均值
- **角度模长**：`sqrt(roll^2 + pitch^2 + yaw^2)` 的均值

```python
# Python版本
accel_mag = np.sqrt((v[:, 0:3] ** 2).sum(axis=1)).mean()
gyro_mag  = np.sqrt((v[:, 3:6] ** 2).sum(axis=1)).mean()
ori_mag   = np.sqrt((v[:, 6:9] ** 2).sum(axis=1)).mean()
# 总计：3维
```

**总计**：36 + 27 + 9 + 3 = **75维特征**

### 4. 归一化
提取特征后，使用 `data.h` 中的 `feature_min` 和 `feature_max` 进行归一化：

```cpp
// 归一化公式：(当前值 - 最小值) / (最大值 - 最小值)
normalizeInputBuffer();  // 在 inference() 函数内部自动完成
```

## 代码对应关系

### Python (data_prepare.ipynb)
```python
def extract_features(values: np.ndarray) -> np.ndarray:
    # values: shape (T, 9)
    # 返回: shape (75,)
    
    # 1. 全局统计
    means = v.mean(axis=0)
    stds  = v.std(axis=0)
    mins  = v.min(axis=0)
    maxs  = v.max(axis=0)
    
    # 2. 时间分段
    segments = np.array_split(v, 3, axis=0)
    for seg in segments:
        seg_mean = seg.mean(axis=0)
    
    # 3. 能量
    energy = (v ** 2).mean(axis=0)
    
    # 4. 模长均值
    accel_mag = np.sqrt((v[:, 0:3] ** 2).sum(axis=1)).mean()
    gyro_mag  = np.sqrt((v[:, 3:6] ** 2).sum(axis=1)).mean()
    ori_mag   = np.sqrt((v[:, 6:9] ** 2).sum(axis=1)).mean()
```

### C++ (IMU_Classifier_CArray.ino)
```cpp
void extractFeatures(float features[75]) {
    // 1. 扩展为9维（添加角度）
    // 2. 全局统计（36维）
    // 3. 时间分段（27维）
    // 4. 能量（9维）
    // 5. 模长均值（3维）
    // 总计：75维
}
```

## 验证特征提取

### 方法1：维度检查
```cpp
if (feat_idx != 75) {
    Serial.println("错误：特征维度不匹配！");
}
```

### 方法2：打印特征值
```cpp
Serial.println("提取的特征：");
for (int i = 0; i < 75; i++) {
    Serial.print(features[i], 6);
    Serial.print(" ");
}
Serial.println();
```

### 方法3：与Python输出对比
1. 在Python中打印特征值
2. 在Arduino中打印特征值
3. 对比是否一致（允许小的浮点误差）

## 常见问题

### Q1: 为什么需要9维数据？
**A**: 训练数据包含9维（ax, ay, az, gx, gy, gz, roll, pitch, yaw），所以特征提取也需要9维。

### Q2: 角度计算方式不同怎么办？
**A**: 如果训练数据使用了不同的角度计算方法（例如从陀螺仪积分），需要修改角度计算部分。

### Q3: 样本数量必须是119吗？
**A**: 是的，训练时使用了119个样本，推理时也必须使用相同数量。

### Q4: 特征提取顺序重要吗？
**A**: 非常重要！必须严格按照：mean/std/min/max → 分段均值 → 能量 → 模长均值的顺序。

### Q5: 归一化参数从哪里来？
**A**: 从 `data.h` 中的 `feature_min` 和 `feature_max`，这些是在训练集上计算的。

## 调试建议

1. **打印中间结果**：检查9维数据是否正确
2. **对比Python输出**：确保特征值一致
3. **检查归一化**：确保归一化后的值在[0,1]范围内
4. **验证维度**：确保始终是75维

## 总结

特征提取是推理的关键步骤，必须：
- ✅ 与训练时的预处理逻辑完全一致
- ✅ 使用相同的归一化参数
- ✅ 保持相同的特征顺序
- ✅ 验证特征维度正确

只有这样才能确保推理结果的准确性！

