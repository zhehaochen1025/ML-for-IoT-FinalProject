#include <Arduino_LSM9DS1.h>
#include <TinyMLShield.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define LEARNING_RATE 0.001
#define EPOCH 50
#define DATA_TYPE_FLOAT

const int first_layer_input_cnt = 75;
const int classes_cnt = 4;

const int train_data_cnt = 1;
const int validation_data_cnt = 1;
const int test_data_cnt = 1;

static const float train_data[1][75] = {{0}};
static const int train_labels[1] = {0};
static const float validation_data[1][75] = {{0}};
static const int validation_labels[1] = {0};
static const float test_data[1][75] = {{0}};
static const int test_labels[1] = {0};

static const int NN_def[] = {first_layer_input_cnt, 32, classes_cnt};
#include "NN_functions.h"
#include "inference.h"

// 根据你的训练配置选择：
// 选项1：如果训练时是100Hz，2秒窗口 = 200个样本
// const int numSamples = 200;
// const int targetSampleRate = 100;  // 目标采样率（Hz）

// 选项2：如果训练时是119Hz，2秒窗口 = 238个样本（推荐，因为Arduino默认119Hz）
const int numSamples = 238;
const int targetSampleRate = 119;
const unsigned long sampleIntervalMs = 1000 / targetSampleRate;

int samplesRead = numSamples;

float imu_buffer[238][9];

const char* GESTURES[] = {
  "circle",
  "other",
  "peak",
  "wave"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void extractFeatures(float features[75]) {
  int num_samples = numSamples;
  static float data_9d[238][9];
  
  for (int i = 0; i < num_samples; i++) {
    data_9d[i][0] = imu_buffer[i][0];
    data_9d[i][1] = imu_buffer[i][1];
    data_9d[i][2] = imu_buffer[i][2];
    data_9d[i][3] = imu_buffer[i][3];
    data_9d[i][4] = imu_buffer[i][4];
    data_9d[i][5] = imu_buffer[i][5];
    data_9d[i][6] = imu_buffer[i][6];
    data_9d[i][7] = imu_buffer[i][7];
    data_9d[i][8] = imu_buffer[i][8];
  }
  
  int feat_idx = 0;
  
  for (int d = 0; d < 9; d++) {
    float sum = 0.0, sum_sq = 0.0, min_val = data_9d[0][d], max_val = data_9d[0][d];
    
    for (int i = 0; i < num_samples; i++) {
      float val = data_9d[i][d];
      sum += val;
      sum_sq += val * val;
      if (val < min_val) min_val = val;
      if (val > max_val) max_val = val;
    }
    
    float mean = sum / num_samples;
    features[feat_idx++] = mean;
    
    float variance = (sum_sq / num_samples) - (mean * mean);
    features[feat_idx++] = sqrt(max(0.0, variance));
    
    features[feat_idx++] = min_val;
    features[feat_idx++] = max_val;
  }
  
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
  
  for (int d = 0; d < 9; d++) {
    float sum_sq = 0.0;
    for (int i = 0; i < num_samples; i++) {
      float val = data_9d[i][d];
      sum_sq += val * val;
    }
    features[feat_idx++] = sum_sq / num_samples;
  }
  
  float accel_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][0]*data_9d[i][0] + 
                     data_9d[i][1]*data_9d[i][1] + 
                     data_9d[i][2]*data_9d[i][2]);
    accel_mag_sum += mag;
  }
  features[feat_idx++] = accel_mag_sum / num_samples;
  
  float gyro_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][3]*data_9d[i][3] + 
                     data_9d[i][4]*data_9d[i][4] + 
                     data_9d[i][5]*data_9d[i][5]);
    gyro_mag_sum += mag;
  }
  features[feat_idx++] = gyro_mag_sum / num_samples;
  
  float ori_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][6]*data_9d[i][6] + 
                     data_9d[i][7]*data_9d[i][7] + 
                     data_9d[i][8]*data_9d[i][8]);
    ori_mag_sum += mag;
  }
  features[feat_idx++] = ori_mag_sum / num_samples;
  
  if (feat_idx != 75) {
    Serial.print("Error: Feature dimension mismatch! Expected 75, got ");
    Serial.println(feat_idx);
    while (feat_idx < 75) {
      features[feat_idx++] = 0.0;
    }
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  
  Serial.println("IMU Classifier - C Array Inference Version");
  Serial.println("================================");
  
  initializeShield();
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  int actualAccelRate = IMU.accelerationSampleRate();
  int actualGyroRate = IMU.gyroscopeSampleRate();
  
  Serial.print("Accelerometer sample rate = ");
  Serial.print(actualAccelRate);
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(actualGyroRate);
  Serial.println(" Hz");
  
  Serial.print("Target sample rate = ");
  Serial.print(targetSampleRate);
  Serial.println(" Hz");
  Serial.print("Window size = ");
  Serial.print(numSamples);
  Serial.print(" samples (");
  Serial.print((float)numSamples / targetSampleRate, 2);
  Serial.println(" seconds)");
  Serial.print("Sample interval = ");
  Serial.print(sampleIntervalMs);
  Serial.println(" ms");
  Serial.println();
  
  if (abs(actualAccelRate - targetSampleRate) > 5) {
    Serial.print("Warning: Actual sample rate(");
    Serial.print(actualAccelRate);
    Serial.print("Hz) differs from target(");
    Serial.print(targetSampleRate);
    Serial.println("Hz)!");
    Serial.println("This may cause inaccurate inference results.");
    Serial.println();
  }
  
  Serial.println("Initializing neural network...");
  
  int weights_bias_cnt = calcTotalWeightsBias();
  Serial.print("Total weights count: ");
  Serial.println(weights_bias_cnt);
  
  DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));
  
  setupNN(WeightBiasPtr);
  
  Serial.println("Loading model weights...");
  loadModel(trained_weights);
  
  Serial.println("Model loaded successfully!");
  Serial.println("Press TinyShield button to start data collection...");
  Serial.println();
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ, mX, mY, mZ;
  
  while (samplesRead == numSamples) {
    bool buttonClicked = readShieldButton();
    if (buttonClicked) {
      Serial.println("Button pressed, starting data collection...");
      samplesRead = 0;
      break;
    }
  }
  
  static unsigned long lastSampleTime = 0;
  
  if (samplesRead == 0) {
    lastSampleTime = millis();
  }
  
  while (samplesRead < numSamples) {
    unsigned long currentTime = millis();
    static float lastMX = 0, lastMY = 0, lastMZ = 0;

    if (currentTime - lastSampleTime >= sampleIntervalMs) {
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(aX, aY, aZ);
        IMU.readGyroscope(gX, gY, gZ);

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

        lastSampleTime += sampleIntervalMs;
        samplesRead++;
      }
    }
  }
  
  if (samplesRead == numSamples) {
    float features[75];
    extractFeatures(features);
    
    int predicted_class = inference(features);
    
    float probabilities[classes_cnt];
    inferenceWithProbabilities(features, probabilities);
    
    Serial.println("--- Inference Result ---");
    Serial.print("Predicted class: ");
    Serial.print(predicted_class);
    Serial.print(" (");
    if (predicted_class < NUM_GESTURES) {
      Serial.print(GESTURES[predicted_class]);
    }
    Serial.print(")");
    Serial.print(" - Confidence: ");
    Serial.print(probabilities[predicted_class] * 100, 2);
    Serial.println("%");
    
    Serial.println("All class probabilities:");
    for (int i = 0; i < NUM_GESTURES; i++) {
      Serial.print("  ");
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(probabilities[i] * 100, 2);
    }
    Serial.println();
  }
}

