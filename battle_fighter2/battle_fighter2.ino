// NodeB.ino - BLE Central + IMU Inference Node (receives A's result and determines winner)
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <TinyMLShield.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const char* TARGET_NAME_A = "NODE_A";
const char* SERVICE_UUID = "180C";
const char* CHAR_INFERENCE = "2A58";

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

const int numSamples = 238;
const int targetSampleRate = 119;
const unsigned long sampleIntervalMs = 1000 / targetSampleRate;
float imu_buffer[238][9];
int samplesRead = numSamples;

const char* GESTURES[] = {"circle", "other", "peak", "wave"};

typedef struct {
  int predicted_class;
  float confidence;
  unsigned long timestamp_ms;
} inference_result_t;

const char* getBattleResult(int classA, int classB) {
  if (classA == classB) {
    return "Draw";
  }
  
  if (classA == 1 || classB == 1) {
    return "Invalid Match";
  }
  
  if ((classA == 0 && classB == 3) ||  // circle beats wave
      (classA == 3 && classB == 2) ||  // wave beats peak
      (classA == 2 && classB == 0)) {   // peak beats circle
    return "A Wins";
  } else {
    return "B Wins";
  }
}

inference_result_t resultA = {0, 0.0f, 0};

void connectAndReceiveInference(BLEDevice peripheral);
void checkBattleResult(inference_result_t resultA, inference_result_t resultB);

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
    float mag = sqrt(data_9d[i][0]*data_9d[i][0] + data_9d[i][1]*data_9d[i][1] + data_9d[i][2]*data_9d[i][2]);
    accel_mag_sum += mag;
  }
  features[feat_idx++] = accel_mag_sum / num_samples;
  
  float gyro_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][3]*data_9d[i][3] + data_9d[i][4]*data_9d[i][4] + data_9d[i][5]*data_9d[i][5]);
    gyro_mag_sum += mag;
  }
  features[feat_idx++] = gyro_mag_sum / num_samples;
  
  float ori_mag_sum = 0.0;
  for (int i = 0; i < num_samples; i++) {
    float mag = sqrt(data_9d[i][6]*data_9d[i][6] + data_9d[i][7]*data_9d[i][7] + data_9d[i][8]*data_9d[i][8]);
    ori_mag_sum += mag;
  }
  features[feat_idx++] = ori_mag_sum / num_samples;
}

int inferenceWithProbabilities(const DATA_TYPE* input_data, DATA_TYPE* probabilities) {
  for (int i = 0; i < IN_VEC_SIZE; i++) {
    input[i] = input_data[i];
  }
  forwardProp();
  int maxIndx = 0;
  for (int j = 0; j < OUT_VEC_SIZE; j++) {
    probabilities[j] = y[j];
    if (j > 0 && y[maxIndx] < y[j]) {
      maxIndx = j;
    }
  }
  return maxIndx;
}

void checkBattleResult(inference_result_t resultA, inference_result_t resultB) {
  unsigned long timeDiff = (resultA.timestamp_ms > resultB.timestamp_ms) ? 
                           (resultA.timestamp_ms - resultB.timestamp_ms) : 
                           (resultB.timestamp_ms - resultA.timestamp_ms);
  
  if (timeDiff < 4000) {
    Serial.println("========================================");
    Serial.println("--- BATTLE RESULT (Valid Match) ---");
    Serial.print("NODE_A: ");
    Serial.print(GESTURES[resultA.predicted_class]);
    Serial.print(" (");
    Serial.print(resultA.confidence * 100, 2);
    Serial.print("%)");
    Serial.print(" - Time: ");
    Serial.print(resultA.timestamp_ms);
    Serial.println(" ms");
    
    Serial.print("NODE_B: ");
    Serial.print(GESTURES[resultB.predicted_class]);
    Serial.print(" (");
    Serial.print(resultB.confidence * 100, 2);
    Serial.print("%)");
    Serial.print(" - Time: ");
    Serial.print(resultB.timestamp_ms);
    Serial.println(" ms");
    
    Serial.print("Time Difference: ");
    Serial.print(timeDiff);
    Serial.println(" ms");
    
    Serial.print("Result: ");
    Serial.println(getBattleResult(resultA.predicted_class, resultB.predicted_class));
    Serial.println("========================================");
  } else {
    Serial.println("========================================");
    Serial.println("--- Invalid Match (Time Difference Too Large) ---");
    Serial.print("NODE_A Time: ");
    Serial.print(resultA.timestamp_ms);
    Serial.println(" ms");
    Serial.print("NODE_B Time: ");
    Serial.print(resultB.timestamp_ms);
    Serial.println(" ms");
    Serial.print("Time Difference: ");
    Serial.print(timeDiff);
    Serial.println(" ms (>= 4000ms)");
    Serial.println("Both results reset. Waiting for new match...");
    Serial.println("========================================");
  }
}

void setup() {
  Serial.begin(9600);
  unsigned long start = millis();
  while (!Serial && millis() - start < 3000);

  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  initializeShield();
  
  if (!IMU.begin()) {
    Serial.println("Node B: Failed to initialize IMU!");
  } else {
    Serial.println("Node B: IMU initialized");
  }
  
  Serial.println("Node B: Initializing neural network...");
  int weights_bias_cnt = calcTotalWeightsBias();
  Serial.print("Node B: Total weights count: ");
  Serial.println(weights_bias_cnt);
  
  WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));
  if (WeightBiasPtr == NULL) {
    Serial.println("Node B: Failed to allocate memory for weights!");
    while(1);
  }
  
  setupNN(WeightBiasPtr);
  Serial.println("Node B: Loading trained model weights...");
  loadModel(trained_weights);
  Serial.println("Node B: Neural network initialized successfully!");
  Serial.println();
  
  Serial.println("Node B: Scanning for NODE_A...");
  BLE.scanForName(TARGET_NAME_A);
}

void loop() {
  BLEDevice peripheral = BLE.available();

  if (peripheral) {
    String deviceName = peripheral.localName();
    Serial.print("Node B: found device '");
    Serial.print(deviceName);
    Serial.print("' [");
    Serial.print(peripheral.address());
    Serial.println("]");

    if (deviceName == TARGET_NAME_A) {
      Serial.println("Node B: connecting to NODE_A...");
      connectAndReceiveInference(peripheral);
      Serial.println("Node B: resume scanning...");
      BLE.scanForName(TARGET_NAME_A);
    }
  }

  BLE.poll();
}

void connectAndReceiveInference(BLEDevice peripheral) {
  BLE.stopScan();

  if (!peripheral.connect()) {
    Serial.println("Node B: connect to NODE_A failed.");
    return;
  }

  Serial.println("Node B: connected to NODE_A, discovering attributes...");
  
  if (!peripheral.discoverAttributes()) {
    Serial.println("Node B: attribute discovery failed.");
    peripheral.disconnect();
    return;
  }

  BLEService inferenceService = peripheral.service(SERVICE_UUID);
  if (!inferenceService) {
    Serial.println("Node B: Inference service not found.");
    peripheral.disconnect();
    return;
  }

  BLECharacteristic inferenceResultChar = inferenceService.characteristic(CHAR_INFERENCE);
  if (!inferenceResultChar) {
    Serial.println("Node B: Inference characteristic not found.");
    peripheral.disconnect();
    return;
  }
  
  if (!inferenceResultChar.subscribe()) {
    Serial.println("Node B: Failed to subscribe to inference results.");
  } else {
    Serial.println("Node B: Subscribed to NODE_A inference results.");
  }

  inference_result_t lastInferenceResult = {0, 0.0f, 0};
  inference_result_t cachedResultB = {0, 0.0f, 0};
  unsigned long lastCheckTime = 0;
  const unsigned long CHECK_INTERVAL = 100;  // Check every 100ms
  
  while (peripheral.connected()) {
    if (inferenceResultChar.valueUpdated()) {
      inference_result_t result;
      int infLen = inferenceResultChar.readValue((uint8_t*)&result, sizeof(result));
      if (infLen == sizeof(result) && result.timestamp_ms != lastInferenceResult.timestamp_ms) {
        resultA = result;
        
        Serial.println("========================================");
        Serial.print("--- Inference Result from NODE_A ---");
        Serial.println();
        Serial.print("  Predicted Class: ");
        Serial.print(result.predicted_class);
        Serial.print(" (");
        if (result.predicted_class >= 0 && result.predicted_class < 4) {
          Serial.print(GESTURES[result.predicted_class]);
        }
        Serial.print(")");
        Serial.print(" - Confidence: ");
        Serial.print(result.confidence * 100, 2);
        Serial.print("%");
        Serial.print(" - Timestamp: ");
        Serial.print(result.timestamp_ms);
        Serial.println(" ms");
        
        lastInferenceResult = result;
      }
    }
    
    // Poll both results periodically using timestamp as lock
    unsigned long currentTime = millis();
    if (currentTime - lastCheckTime >= CHECK_INTERVAL) {
      lastCheckTime = currentTime;
      if (resultA.timestamp_ms != 0 && cachedResultB.timestamp_ms != 0) {
        checkBattleResult(resultA, cachedResultB);
        resultA.timestamp_ms = 0;
        cachedResultB.timestamp_ms = 0;
      }
    }
    
    float aX, aY, aZ, gX, gY, gZ, mX, mY, mZ;
    
    while (samplesRead == numSamples) {
      bool buttonClicked = readShieldButton();
      if (buttonClicked) {
        Serial.println("Node B: Button pressed, starting data collection...");
        samplesRead = 0;
        break;
      }
      BLE.poll();
      delay(10);
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
      
      BLE.poll();
    }
    
    if (samplesRead == numSamples) {
      float features[75];
      extractFeatures(features);
      
      float probabilities[classes_cnt];
      int predicted_class = inferenceWithProbabilities(features, probabilities);
      float confidence = probabilities[predicted_class];
      
      cachedResultB.predicted_class = predicted_class;
      cachedResultB.confidence = confidence;
      cachedResultB.timestamp_ms = millis();
      
      Serial.println("--- Inference Result (Node B) ---");
      Serial.print("  Predicted Class: ");
      Serial.print(predicted_class);
      Serial.print(" (");
      Serial.print(GESTURES[predicted_class]);
      Serial.print(")");
      Serial.print(" - Confidence: ");
      Serial.print(confidence * 100, 2);
      Serial.print("%");
      Serial.print(" - Timestamp: ");
      Serial.print(cachedResultB.timestamp_ms);
      Serial.println(" ms");
      
      if (resultA.timestamp_ms != 0) {
        checkBattleResult(resultA, cachedResultB);
        resultA.timestamp_ms = 0;
        cachedResultB.timestamp_ms = 0;
      }
      
      Serial.println("Press button to collect next sample...");
      samplesRead = numSamples;
    }

    BLE.poll();
    delay(50);
  }

  peripheral.disconnect();
  Serial.println("Node B: disconnected from NODE_A");
}
