/*
 * 推理示例文件
 * 
 * 使用方法：
 * 1. 训练完成后，从串口复制保存的权重数组
 * 2. 将权重数组粘贴到下面的 trained_weights 数组中
 * 3. 在 setup() 中调用 loadModel(trained_weights) 加载权重
 * 4. 使用 inference() 或 inferenceWithProbabilities() 进行推理
 */

#ifndef INFERENCE_EXAMPLE_H
#define INFERENCE_EXAMPLE_H

#include "NN_functions.h"

// 将训练后保存的权重数组粘贴到这里
// 格式：从串口输出的 const float trained_weights[] = {...} 复制过来
const float trained_weights[] = {
  // 在这里粘贴从串口复制的权重数据
  // 例如：0.123456f, -0.789012f, ...
};

// 推理示例函数
void inferenceExample() {
  // 1. 加载保存的模型权重
  loadModel(trained_weights);
  
  // 2. 准备输入数据（75维特征向量）
  // 注意：输入数据应该是归一化后的特征
  float test_input[75] = {
    // 在这里填入你的输入特征数据
    // 例如：从传感器读取并提取的特征
  };
  
  // 3. 执行推理（只返回类别）
  int predicted_class = inference(test_input);
  
  Serial.print("预测类别: ");
  Serial.println(predicted_class);
  
  // 4. 执行推理（返回类别和概率）
  float probabilities[classes_cnt];
  int predicted_class_with_probs = inferenceWithProbabilities(test_input, probabilities);
  
  Serial.println("各类别概率:");
  for (int i = 0; i < classes_cnt; i++) {
    Serial.print("  类别 ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(probabilities[i], 6);
  }
  Serial.print("预测类别: ");
  Serial.println(predicted_class_with_probs);
}

#endif

