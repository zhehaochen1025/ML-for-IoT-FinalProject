/*
  IMU Classifier - C数组推理版本

  这个版本使用C数组方式替代TensorFlow Lite进行推理
  优势：更轻量、更快、内存占用更小

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  改编自原始 TensorFlow Lite 版本
*/

#include <Arduino_LSM9DS1.h>
#include <TinyMLShield.h>

// ========== 神经网络相关头文件 ==========
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// NN parameters（需要与训练时一致）
#define LEARNING_RATE 0.001
#define EPOCH 50
#define DATA_TYPE_FLOAT

// 网络参数（推理时只需要这两个常量，不需要整个 data.h）
const int first_layer_input_cnt = 75;  // 输入特征维度
const int classes_cnt = 4;              // 类别数量

// 为了兼容 NN_functions.h，定义这些常量（推理时不会被使用）
// 使用小的非零值以避免数组声明问题
const int train_data_cnt = 1;
const int validation_data_cnt = 1;
const int test_data_cnt = 1;

// 为了兼容 NN_functions.h，定义这些空数组（推理时不会被使用）
// 这些变量仅在训练相关的函数中使用，推理代码不会调用这些函数
static const float train_data[1][75] = {{0}};
static const int train_labels[1] = {0};
static const float validation_data[1][75] = {{0}};
static const int validation_labels[1] = {0};
static const float test_data[1][75] = {{0}};
static const int test_labels[1] = {0};

// 网络结构（需要与训练时一致，必须与训练代码中的 NN_def 一致）
static const int NN_def[] = {first_layer_input_cnt, 32, classes_cnt};
#include "NN_functions.h"   // 神经网络函数
#include "inference.h"      // 训练好的权重数组

// 推理函数：获取所有类别的概率
// 参数: input_data - 输入特征向量
//       probabilities - 输出概率数组（必须至少有 classes_cnt 个元素）
// 返回: 预测类别的索引
int inferenceWithProbabilities(const DATA_TYPE* input_data, DATA_TYPE* probabilities) {
  // 1. 将输入数据复制到 input 数组
  for (int i = 0; i < IN_VEC_SIZE; i++) {
    input[i] = input_data[i];
  }
  
  // 2. 执行前向传播（forwardProp 会自动进行 softmax 归一化）
  forwardProp();
  
  // 3. 复制概率到输出数组
  int maxIndx = 0;
  for (int j = 0; j < OUT_VEC_SIZE; j++) {
    probabilities[j] = y[j];
    if (j > 0 && y[maxIndx] < y[j]) {
      maxIndx = j;
    }
  }
  
  return maxIndx;
} 
// ========== IMU 参数 ==========
// 训练配置：2秒窗口
// 如果训练时使用100Hz采样率：2秒 × 100Hz = 200个样本
// 如果训练时使用119Hz采样率：2秒 × 119Hz = 238个样本
// 注意：必须与训练时的样本数量一致！

// 根据你的训练配置选择：
// 选项1：如果训练时是100Hz，2秒窗口 = 200个样本
// const int numSamples = 200;
// const int targetSampleRate = 100;  // 目标采样率（Hz）

// 选项2：如果训练时是119Hz，2秒窗口 = 238个样本（推荐，因为Arduino默认119Hz）
const int numSamples = 238;
const int targetSampleRate = 119;

// 选项3：如果训练时实际用的是119个样本（约1秒窗口）
// const int numSamples = 119;
// const int targetSampleRate = 119;

const unsigned long sampleIntervalMs = 1000 / targetSampleRate;  // 采样间隔（毫秒）：1000ms / 100Hz = 10ms

int samplesRead = numSamples;

// ========== 数据缓冲区 ==========
// 存储IMU原始数据：numSamples个样本 × 6个值（ax, ay, az, gx, gy, gz）
float imu_buffer[238][9];  // 根据numSamples调整大小（最大238）

// ========== 类别名称 ==========
const char* GESTURES[] = {
  "circle",  // 0
  "other",   // 1
  "peak",    // 2
  "wave"     // 3
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// ========== 特征提取函数 ==========
// 从IMU缓冲区提取75维特征向量
// 此函数必须与 data_prepare.ipynb 中的 extract_features() 逻辑完全一致
// Python版本：从 (T, 9) 数据提取 75 维特征
void extractFeatures(float features[75]) {
  // 使用全局的numSamples，确保与训练时一致
  int num_samples = numSamples;
  
  // 临时数组：直接使用9维IMU数据
  // 训练数据包含：ax, ay, az, gx, gy, gz, magX, magY, magZ (9维)
  // CBOR原始数据就是这9维，不需要转换！
  static float data_9d[238][9];  // 最大238，根据实际numSamples使用
  
  for (int i = 0; i < num_samples; i++) {
    // 直接复制所有9维数据
    data_9d[i][0] = imu_buffer[i][0]; // ax
    data_9d[i][1] = imu_buffer[i][1]; // ay
    data_9d[i][2] = imu_buffer[i][2]; // az
    data_9d[i][3] = imu_buffer[i][3]; // gx
    data_9d[i][4] = imu_buffer[i][4]; // gy
    data_9d[i][5] = imu_buffer[i][5]; // gz
    data_9d[i][6] = imu_buffer[i][6]; // mx
    data_9d[i][7] = imu_buffer[i][7]; // my
    data_9d[i][8] = imu_buffer[i][8]; // mz
  }
  
  int feat_idx = 0;
  
  // 1) 全局统计：mean, std, min, max -> 9 * 4 = 36维
  // 与Python: means = v.mean(axis=0), stds = v.std(axis=0), mins = v.min(axis=0), maxs = v.max(axis=0)
  for (int d = 0; d < 9; d++) {
    float sum = 0.0, sum_sq = 0.0, min_val = data_9d[0][d], max_val = data_9d[0][d];
    
    for (int i = 0; i < num_samples; i++) {
      float val = data_9d[i][d];
      sum += val;
      sum_sq += val * val;
      if (val < min_val) min_val = val;
      if (val > max_val) max_val = val;
    }
    
    // mean
    float mean = sum / num_samples;
    features[feat_idx++] = mean;
    
    // std: sqrt(mean(x^2) - mean(x)^2)
    float variance = (sum_sq / num_samples) - (mean * mean);
    features[feat_idx++] = sqrt(max(0.0, variance));
    
    // min
    features[feat_idx++] = min_val;
    
    // max
    features[feat_idx++] = max_val;
  }
  
  // 2) 时间分三段，每段算 mean -> 3 * 9 = 27维
  // 与Python: segments = np.array_split(v, 3, axis=0), seg_mean = seg.mean(axis=0)
  int segment_size = num_samples / 3;
  for (int seg = 0; seg < 3; seg++) {
    int start = seg * segment_size;
    int end = (seg == 2) ? num_samples : (seg + 1) * segment_size;
    int seg_len = end - start;
    
    for (int d = 0; d < 9; d++) {
      float sum = 0.0;
      for (int i = start; i < end; i++) {
        sum += data_9d[i][d];
      }
      features[feat_idx++] = sum / seg_len;
    }
  }
  
  // 3) 每个通道的 energy（均方）-> 9维
  // 与Python: energy = (v ** 2).mean(axis=0)
  for (int d = 0; d < 9; d++) {
    float sum_sq = 0.0;
    for (int i = 0; i < num_samples; i++) {
      float val = data_9d[i][d];
      sum_sq += val * val;
    }
    features[feat_idx++] = sum_sq / num_samples;
  }
  
  // 4) 三个向量模长的均值（与Python代码一致：先计算模长，再求均值）-> 3维
  // accel magnitude: sqrt((ax^2 + ay^2 + az^2)) 的均值
  float accel_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][0]*data_9d[i][0] + 
                     data_9d[i][1]*data_9d[i][1] + 
                     data_9d[i][2]*data_9d[i][2]);
    accel_mag_sum += mag;
  }
  features[feat_idx++] = accel_mag_sum / num_samples;
  
  // gyro magnitude: sqrt((gx^2 + gy^2 + gz^2)) 的均值
  float gyro_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][3]*data_9d[i][3] + 
                     data_9d[i][4]*data_9d[i][4] + 
                     data_9d[i][5]*data_9d[i][5]);
    gyro_mag_sum += mag;
  }
  features[feat_idx++] = gyro_mag_sum / num_samples;
  
  // magnetometer magnitude: sqrt((magX^2 + magY^2 + magZ^2)) 的均值
  float ori_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][6]*data_9d[i][6] + 
                     data_9d[i][7]*data_9d[i][7] + 
                     data_9d[i][8]*data_9d[i][8]);
    ori_mag_sum += mag;
  }
  features[feat_idx++] = ori_mag_sum / num_samples;
  
  // 验证特征维度
  if (feat_idx != 75) {
    Serial.print("错误：特征维度不匹配！期望75，实际");
    Serial.println(feat_idx);
    // 填充剩余维度为0（防止数组越界）
    while (feat_idx < 75) {
      features[feat_idx++] = 0.0;
    }
  }
  
  // 特征提取完成，与 data_prepare.ipynb 中的 extract_features() 逻辑一致
  // 下一步：归一化（在 inference() 函数内部自动完成）
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  
  Serial.println("IMU Classifier - C数组推理版本");
  Serial.println("================================");
  
  // 初始化 TinyML Shield（包含按钮）
  initializeShield();
  
  // 初始化 IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  // 打印 IMU 采样率
  int actualAccelRate = IMU.accelerationSampleRate();
  int actualGyroRate = IMU.gyroscopeSampleRate();
  
  Serial.print("Accelerometer sample rate = ");
  Serial.print(actualAccelRate);
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(actualGyroRate);
  Serial.println(" Hz");
  
  Serial.print("目标采样率 = ");
  Serial.print(targetSampleRate);
  Serial.println(" Hz");
  Serial.print("窗口大小 = ");
  Serial.print(numSamples);
  Serial.print(" 个样本 (");
  Serial.print((float)numSamples / targetSampleRate, 2);
  Serial.println(" 秒)");
  Serial.print("采样间隔 = ");
  Serial.print(sampleIntervalMs);
  Serial.println(" ms");
  Serial.println();
  
  // 警告：如果实际采样率与目标不一致
  if (abs(actualAccelRate - targetSampleRate) > 5) {
    Serial.print("⚠️  警告：实际采样率(");
    Serial.print(actualAccelRate);
    Serial.print("Hz) 与目标采样率(");
    Serial.print(targetSampleRate);
    Serial.println("Hz) 不一致！");
    Serial.println("   这可能导致推理结果不准确。");
    Serial.println();
  }
  
  // ========== 初始化神经网络 ==========
  Serial.println("正在初始化神经网络...");
  
  // 计算权重数量
  int weights_bias_cnt = calcTotalWeightsBias();
  Serial.print("权重数量: ");
  Serial.println(weights_bias_cnt);
  
  // 分配权重内存
  DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));
  
  // 设置网络
  setupNN(WeightBiasPtr);
  
  // 加载训练好的权重
  Serial.println("正在加载模型权重...");
  loadModel(trained_weights);
  
  Serial.println("模型加载完成！");
  Serial.println("按TinyShield按钮开始采集数据...");
  Serial.println();
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ, mX, mY, mZ;
  
  // ========== 等待按钮按下 ==========
  while (samplesRead == numSamples) {
    bool buttonClicked = readShieldButton();
    if (buttonClicked) {
      Serial.println("按钮按下，开始采集数据...");
      samplesRead = 0;
      break;
    }
  }
  
  // ========== 采集IMU数据 ==========
  // 使用定时采样，确保采样率与训练时一致
  static unsigned long lastSampleTime = 0;
  
  // 如果是第一次采样，初始化时间
  if (samplesRead == 0) {
    lastSampleTime = millis();
  }
  
  while (samplesRead < numSamples) {
    unsigned long currentTime = millis();
    
    // 检查是否到了采样时间
    static float lastMX = 0, lastMY = 0, lastMZ = 0;

if (currentTime - lastSampleTime >= sampleIntervalMs) {

  // accel + gyro 是主频
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);
    IMU.readGyroscope(gX, gY, gZ);

    // mag 有就更新，没有就用上次的
    if (IMU.magneticFieldAvailable()) {
      IMU.readMagneticField(lastMX, lastMY, lastMZ);
    }

    imu_buffer[samplesRead][0] = aX;
    imu_buffer[samplesRead][1] = aY;
    imu_buffer[samplesRead][2] = aZ;
    imu_buffer[samplesRead][3] = gX;
    imu_buffer[samplesRead][4] = gY;
    imu_buffer[samplesRead][5] = gZ;
    imu_buffer[samplesRead][6] = lastMX;
    imu_buffer[samplesRead][7] = lastMY;
    imu_buffer[samplesRead][8] = lastMZ;

    lastSampleTime += sampleIntervalMs; // 比 lastSampleTime=currentTime 更稳
    samplesRead++;
  }
}
  }
  
  // ========== 采集完成后进行推理 ==========
  if (samplesRead == numSamples) {
    // 1. 提取特征
    float features[75];
    extractFeatures(features);
    
    // 2. 执行推理并获取所有类别的概率
    float probabilities[classes_cnt];
    int predicted_class = inferenceWithProbabilities(features, probabilities);
    
    // 4. 输出结果
    Serial.println("--- 推理结果 ---");
    Serial.print("预测类别: ");
    Serial.print(predicted_class);
    Serial.print(" (");
    if (predicted_class < NUM_GESTURES) {
      Serial.print(GESTURES[predicted_class]);
    }
    Serial.print(")");
    Serial.print(" - 置信度: ");
    Serial.print(probabilities[predicted_class] * 100, 2);
    Serial.println("%");
    
    // 输出所有类别的概率
    Serial.println("所有类别概率:");
    for (int i = 0; i < NUM_GESTURES; i++) {
      Serial.print("  ");
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(probabilities[i] * 100, 2);
    }
    Serial.println();
  }
}

