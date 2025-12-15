#!/usr/bin/env python3
"""
将 Arduino 训练保存的权重转换为 TensorFlow Lite 格式

使用方法：
1. 从串口复制保存的权重数组（const float trained_weights[] = {...}）
2. 将权重数组保存到 weights.txt 文件中（只保留数字部分）
3. 运行此脚本：python convert_to_tflite.py
"""

import numpy as np
import tensorflow as tf

# 模型结构配置（需要与训练时一致）
INPUT_SIZE = 75
HIDDEN_SIZE = 64
OUTPUT_SIZE = 4

def load_weights_from_file(filename='weights.txt'):
    """
    从文件加载权重数组
    文件格式：每行一个浮点数，或逗号分隔的浮点数
    """
    weights = []
    with open(filename, 'r') as f:
        for line in f:
            # 移除空白字符和 'f' 后缀
            line = line.strip().rstrip('f').rstrip()
            if not line:
                continue
            # 处理逗号分隔的值
            for val in line.split(','):
                val = val.strip().rstrip('f').rstrip()
                if val:
                    try:
                        weights.append(float(val))
                    except ValueError:
                        continue
    return np.array(weights, dtype=np.float32)

def reorganize_weights(flat_weights, input_size, hidden_size, output_size):
    """
    将一维权重数组重组为 TensorFlow/Keras 模型所需的格式
    
    存储顺序（Arduino）：
    1. Layer 1->2 weights: input_size * hidden_size
    2. Layer 2 biases: hidden_size
    3. Layer 2->3 weights: hidden_size * output_size
    4. Layer 3 biases: output_size
    """
    idx = 0
    
    # Layer 1->2 weights
    w1_size = input_size * hidden_size
    w1 = flat_weights[idx:idx + w1_size].reshape(input_size, hidden_size)
    idx += w1_size
    
    # Layer 2 biases
    b1_size = hidden_size
    b1 = flat_weights[idx:idx + b1_size]
    idx += b1_size
    
    # Layer 2->3 weights
    w2_size = hidden_size * output_size
    w2 = flat_weights[idx:idx + w2_size].reshape(hidden_size, output_size)
    idx += w2_size
    
    # Layer 3 biases
    b2_size = output_size
    b2 = flat_weights[idx:idx + b2_size]
    idx += b2_size
    
    return [w1, b1, w2, b2]

def create_and_convert_model(weights_file='weights.txt', output_file='model.tflite'):
    """
    创建 Keras 模型，加载权重，并转换为 TFLite
    """
    print("正在加载权重...")
    flat_weights = load_weights_from_file(weights_file)
    print(f"加载了 {len(flat_weights)} 个权重值")
    
    print("正在重组权重...")
    weights_list = reorganize_weights(flat_weights, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    print("正在创建模型...")
    # 创建模型结构
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            HIDDEN_SIZE,
            activation='relu',
            input_shape=(INPUT_SIZE,),
            name='dense_1'
        ),
        tf.keras.layers.Dense(
            OUTPUT_SIZE,
            activation='softmax',
            name='dense_2'
        )
    ])
    
    # 设置权重
    print("正在设置权重...")
    model.layers[0].set_weights([weights_list[0], weights_list[1]])
    model.layers[1].set_weights([weights_list[2], weights_list[3]])
    
    # 编译模型（TFLite转换需要）
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print("正在转换为 TFLite...")
    # 转换为 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 可选：量化（减小模型大小）
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # 保存
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    
    print(f"模型已保存到: {output_file}")
    print(f"模型大小: {len(tflite_model) / 1024:.2f} KB")
    
    # 测试模型
    print("\n测试模型...")
    test_input = np.random.randn(1, INPUT_SIZE).astype(np.float32)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出概率: {output[0]}")
    print(f"预测类别: {np.argmax(output[0])}")

if __name__ == '__main__':
    import sys
    
    weights_file = 'weights.txt'
    output_file = 'model.tflite'
    
    if len(sys.argv) > 1:
        weights_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("=" * 50)
    print("Arduino 权重转 TensorFlow Lite 转换工具")
    print("=" * 50)
    print(f"权重文件: {weights_file}")
    print(f"输出文件: {output_file}")
    print(f"模型结构: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print("=" * 50)
    
    try:
        create_and_convert_model(weights_file, output_file)
        print("\n转换完成！")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

