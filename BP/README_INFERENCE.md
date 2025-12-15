# 模型保存和推理使用指南

## 一、模型保存格式

训练完成后，模型会以以下格式保存到串口：

### 1. 模型元数据（JSON格式）
```json
MODEL_METADATA:{
  "layers": [75, 64, 4],
  "learning_rate": 0.0015,
  "epochs": 50,
  "weights_count": 4860
}
```

### 2. 权重数组（C数组格式）
```c
const float trained_weights[] = {
  0.123456f, -0.789012f, 0.345678f, ...
};
```

## 二、推理使用方法

### 方法1：在同一项目中直接使用（训练后立即推理）

训练完成后，权重已经在内存中，可以直接使用 `forwardProp()` 进行推理：

```cpp
// 设置输入数据
for (int i = 0; i < 75; i++) {
  input[i] = your_feature_data[i];
}
normalizeInputBuffer();  // 归一化

// 执行推理
forwardProp();

// 获取预测结果
int predicted_class = 0;
for (int i = 1; i < classes_cnt; i++) {
  if (y[predicted_class] < y[i]) {
    predicted_class = i;
  }
}
```

### 方法2：加载保存的权重进行推理

1. **复制权重数据**
   - 从串口监视器复制保存的权重数组
   - 粘贴到 `inference_example.h` 中的 `trained_weights` 数组

2. **加载模型**
   ```cpp
   #include "inference_example.h"
   
   void setup() {
     // ... 其他初始化代码 ...
     
     // 加载保存的权重
     loadModel(trained_weights);
   }
   ```

3. **执行推理**
   ```cpp
   void loop() {
     // 准备输入数据（75维特征向量）
     float input_features[75] = {
       // 你的特征数据
     };
     
     // 方法1：只获取预测类别
     int predicted_class = inference(input_features);
     Serial.print("预测类别: ");
     Serial.println(predicted_class);
     
     // 方法2：获取预测类别和所有类别的概率
     float probabilities[4];
     int predicted = inferenceWithProbabilities(input_features, probabilities);
     
     Serial.println("各类别概率:");
     for (int i = 0; i < 4; i++) {
       Serial.print("  类别 ");
       Serial.print(i);
       Serial.print(": ");
       Serial.println(probabilities[i], 6);
     }
   }
   ```

## 三、权重格式说明

### 权重存储顺序
权重按照以下顺序存储在一维数组中：
1. 第一层到第二层的所有权重（75 × 64 = 4800个）
2. 第二层的所有偏置（64个）
3. 第二层到第三层的所有权重（64 × 4 = 256个）
4. 第三层的所有偏置（4个）

总计：4800 + 64 + 256 + 4 = 5124个浮点数

### 权重加载过程
`loadModel()` 函数会：
1. 将权重数组复制到 `WeightBiasPtr`
2. 使用 `packUnpackVector(UNPACK)` 将权重解包到网络结构中
3. 网络结构：`L[1].Neu[j].W[k]` 和 `L[1].Neu[j].B`

## 四、与 TensorFlow Lite 的对比

### 当前实现（自定义神经网络）
- **格式**：纯C数组（float数组）
- **优点**：
  - 轻量级，无需额外库
  - 可以直接在Arduino上训练和推理
  - 内存占用小
- **缺点**：
  - 需要手动管理权重
  - 不支持量化
  - 不能直接使用TensorFlow生态工具

### TensorFlow Lite
- **格式**：`.tflite` 文件（FlatBuffer格式）
- **优点**：
  - 标准化格式
  - 支持量化（INT8）
  - 可以使用TensorFlow工具链
  - 更好的优化
- **缺点**：
  - 需要TensorFlow Lite库（占用更多内存）
  - 不能直接在Arduino上训练

## 五、转换为 TensorFlow Lite（可选）

如果需要将训练好的权重转换为 TensorFlow Lite 格式，可以使用 Python 脚本：

```python
import tensorflow as tf
import numpy as np

# 1. 加载保存的权重
weights = np.array([...])  # 从串口复制的权重数组

# 2. 重建模型结构
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(75,)),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 3. 手动设置权重（需要按照存储顺序重新组织）
# ... 权重重组代码 ...

# 4. 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 5. 保存
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 六、注意事项

1. **输入归一化**：推理前必须对输入数据进行归一化（使用 `normalizeInputBuffer()`）
2. **特征提取**：输入必须是75维特征向量，需要先进行特征提取
3. **内存管理**：确保有足够的内存存储权重数组
4. **精度**：使用 `float` 类型，精度为32位浮点数

## 七、完整推理示例

参考 `inference_example.h` 文件中的完整示例代码。

