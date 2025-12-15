/*
  Arduino Nano 33 BLE Gesture Classifier (Final Fix)
  
  Fixes:
  1. Removed 'TFLITE_SCHEMA_VERSION' check to prevent compilation error.
  2. Fixed 'MicroInterpreter' constructor arguments (added error_reporter).
  3. Includes Feature Extraction logic.
*/

// --- 1. 头文件 ---
#include <Arduino_LSM9DS1.h> 
#include <MadgwickAHRS.h>
#include <float.h> // FLT_MAX

// TFLite 头文件
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_error_reporter.h> // 必须包含

// 引入你的模型文件
#include "net.h" 

// --- 2. 参数定义 ---
const float SAMPLE_RATE = 119.0f;
#define RAW_DIM 9           // (ax, ay, az, gx, gy, gz, roll, pitch, yaw)
#define FEATURE_DIM 75      // 特征数
#define SAMPLES_PER_GESTURE 119 

const float accelerationThreshold = 1.5; 

// --- 3. 全局对象 ---
Madgwick filter; 
tflite::MicroInterpreter* tflite_interpreter = nullptr;
TfLiteTensor* tflite_input = nullptr;
TfLiteTensor* tflite_output = nullptr;
tflite::ErrorReporter* error_reporter = nullptr; 

// TFLite 内存池
const int kTensorArenaSize = 64 * 1024; 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// 数据缓冲区
float raw_input_buffer[SAMPLES_PER_GESTURE * 9];
float feature_buffer[75];
int samples_read = 0;

// --- 4. 你的分类标签 ---
const char *GESTURES[] = {
    "circle",
    "other",
    "peak",
    "wave"
};
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))


// --- 5. 特征提取函数 ---
void extract_features(float* raw_data, int T, float* features_out) {
    float sums[RAW_DIM] = {0};
    float sum_sqs[RAW_DIM] = {0}; 
    float mins[RAW_DIM];
    float maxs[RAW_DIM];
    
    for(int i=0; i<RAW_DIM; i++) {
        mins[i] = FLT_MAX;
        maxs[i] = -FLT_MAX;
    }

    int seg_len_1 = T / 3 + (T % 3 > 0 ? 1 : 0); 
    int seg_len_2 = T / 3 + (T % 3 > 1 ? 1 : 0); 
    
    float seg_sums[3][RAW_DIM] = {0}; 
    float mag_sum_acc = 0, mag_sum_gyro = 0, mag_sum_ori = 0;

    for (int t = 0; t < T; t++) {
        int row_idx = t * RAW_DIM;
        float acc_sq_sum = 0, gyro_sq_sum = 0, ori_sq_sum = 0;

        for (int d = 0; d < RAW_DIM; d++) {
            float val = raw_data[row_idx + d];
            sums[d] += val;
            sum_sqs[d] += val * val;
            if (val < mins[d]) mins[d] = val;
            if (val > maxs[d]) maxs[d] = val;

            if (t < seg_len_1) seg_sums[0][d] += val;
            else if (t < seg_len_1 + seg_len_2) seg_sums[1][d] += val;
            else seg_sums[2][d] += val;

            if (d < 3) acc_sq_sum += val * val;
            else if (d < 6) gyro_sq_sum += val * val;
            else ori_sq_sum += val * val;
        }
        mag_sum_acc += sqrt(acc_sq_sum);
        mag_sum_gyro += sqrt(gyro_sq_sum);
        mag_sum_ori += sqrt(ori_sq_sum);
    }

    int ptr = 0;
    // 1) Mean
    for (int d = 0; d < RAW_DIM; d++) features_out[ptr++] = sums[d] / T;
    // 2) Std 
    for (int d = 0; d < RAW_DIM; d++) {
        float mean = sums[d] / T;
        float mean_sq = sum_sqs[d] / T;
        float var = mean_sq - (mean * mean);
        features_out[ptr++] = sqrt(var > 0 ? var : 0);
    }
    // 3) Min
    for (int d = 0; d < RAW_DIM; d++) features_out[ptr++] = mins[d];
    // 4) Max
    for (int d = 0; d < RAW_DIM; d++) features_out[ptr++] = maxs[d];
    
    // 5) Seg Mean
    int seg_counts[3] = {seg_len_1, seg_len_2, T - seg_len_1 - seg_len_2};
    for (int s = 0; s < 3; s++) {
        for (int d = 0; d < RAW_DIM; d++) {
            features_out[ptr++] = seg_sums[s][d] / seg_counts[s];
        }
    }

    // 6) Energy
    for (int d = 0; d < RAW_DIM; d++) features_out[ptr++] = sum_sqs[d] / T;

    // 7) Mag Mean
    features_out[ptr++] = mag_sum_acc / T;
    features_out[ptr++] = mag_sum_gyro / T;
    features_out[ptr++] = mag_sum_ori / T;
}

// --- 6. Setup 函数 ---
void setup() {
    Serial.begin(9600);
    // while (!Serial); // 调试时可打开

    // 6.1 初始化 IMU
    if (!IMU.begin()) {
        Serial.println("IMU init failed!");
        while (1);
    }

    // 6.2 初始化 Filter
    filter.begin(SAMPLE_RATE);

    Serial.println();
    Serial.println("Initializing TFLite...");

    // 6.3 设置错误报告器
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // 6.4 加载模型 (trained_model 来自 net.h)
    // 注意：net.h 里的数组名必须是 trained_model，如果不是请在这里修改
    const tflite::Model* model = tflite::GetModel(net); 
    
    // [已删除] 导致报错的版本检查代码已移除
    // if (model->version() != TFLITE_SCHEMA_VERSION) ...

    // 6.5 注册算子
    static tflite::AllOpsResolver resolver;

    // 6.6 创建解释器 (注意：这里有5个参数)
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite_interpreter = &static_interpreter;

    // 6.7 分配内存
    if (tflite_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed!");
        while (1);
    }

    // 6.8 获取指针
    tflite_input = tflite_interpreter->input(0);
    tflite_output = tflite_interpreter->output(0);

    // 检查维度
    if (tflite_input->dims->data[1] != FEATURE_DIM) {
        Serial.print("Error: Input dimension mismatch. Expected 75, got ");
        Serial.println(tflite_input->dims->data[1]);
        while(1);
    }

    Serial.println("System initialized. Perform gestures: circle, other, peak, wave");
}

// --- 7. Loop 函数 ---
void loop() {
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    
    float ax, ay, az, gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    filter.updateIMU(gx, gy, gz, ax, ay, az);
    float roll = filter.getRoll();
    float pitch = filter.getPitch();
    float yaw = filter.getYaw();

    int offset = samples_read * 9;
    raw_input_buffer[offset + 0] = ax;
    raw_input_buffer[offset + 1] = ay;
    raw_input_buffer[offset + 2] = az;
    raw_input_buffer[offset + 3] = gx;
    raw_input_buffer[offset + 4] = gy;
    raw_input_buffer[offset + 5] = gz;
    raw_input_buffer[offset + 6] = roll;
    raw_input_buffer[offset + 7] = pitch;
    raw_input_buffer[offset + 8] = yaw;

    samples_read++;

    if (samples_read == SAMPLES_PER_GESTURE) {
      // A. 特征提取
      extract_features(raw_input_buffer, SAMPLES_PER_GESTURE, feature_buffer);

      // B. 填入模型
      for (int i = 0; i < FEATURE_DIM; i++) {
        tflite_input->data.f[i] = feature_buffer[i];
      }

      // C. 推理
      if (tflite_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
      }

      // D. 解析结果
      float max_score = 0;
      int max_index = -1;
      
      for (int i = 0; i < NUM_GESTURES; i++) {
        float score = tflite_output->data.f[i];
        if (score > max_score) {
          max_score = score;
          max_index = i;
        }
      }

      if (max_index >= 0) {
        Serial.print("Detected: ");
        Serial.print(GESTURES[max_index]);
        Serial.print(" (Score: ");
        Serial.print(max_score);
        Serial.println(")");
      }

      samples_read = 0;
    }
  }
}