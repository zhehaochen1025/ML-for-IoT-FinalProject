/*
 * Test Set Inference - Evaluate model performance on test set
 * 
 * Features:
 * 1. Load trained model weights
 * 2. Perform inference on each sample in test set
 * 3. Output true labels and predicted labels (format: true_label,predicted_label)
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<TinyMLShield.h>

#define LEARNING_RATE 0.001
#define EPOCH 50
#define DATA_TYPE_FlOAT

#include "data_byh.h"
static const unsigned int NN_def[] = {first_layer_input_cnt, 64, classes_cnt};
#include "NN_functions.h"
#include "inference_byh.h"

const char* class_names[] = {
  "circle",
  "other",
  "peak",
  "wave"
};
 
void setup() {
  Serial.begin(9600);
  delay(5000);
  while (!Serial);
  
  initializeShield();
  
  Serial.println("======================================");
  Serial.println("Test Set Inference");
  Serial.println("======================================");
  Serial.println();
  
  int weights_bias_cnt = calcTotalWeightsBias();
  Serial.print("Total weights count: ");
  Serial.println(weights_bias_cnt);
  
  int weights_array_size = sizeof(trained_weights) / sizeof(trained_weights[0]);
  if (weights_array_size == 0 || (weights_array_size == 1 && trained_weights[0] == 0.0f)) {
    Serial.println("Error: Weights array is empty or uninitialized!");
    Serial.println("Please run BP.ino to train the model first, then copy weights array here.");
    Serial.print("Current weights array size: ");
    Serial.println(weights_array_size);
    while(1);
  }
  
  Serial.print("Weights array size: ");
  Serial.println(weights_array_size);
  
  DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));
  
  setupNN(WeightBiasPtr);
  
  Serial.println("Loading model weights...");
  loadModel(trained_weights);
  
  Serial.println();
  Serial.println("Starting test set inference...");
  Serial.println("Output format: sample_index,true_label,predicted_label,true_label_name,predicted_label_name");
  Serial.println("--------------------------------------");
  
  int correct_count = 0;
  for (int i = 0; i < test_data_cnt; i++) {
    int true_label = test_labels[i];
    int predicted_label = inference(test_data[i]);
    
    if (predicted_label == true_label) {
      correct_count++;
    }
    
    Serial.print(i);
    Serial.print(",");
    Serial.print(true_label);
    Serial.print(",");
    Serial.print(predicted_label);
    Serial.print(",");
    Serial.print(class_names[true_label]);
    Serial.print(",");
    Serial.println(class_names[predicted_label]);
  }
  
  Serial.println("--------------------------------------");
  Serial.print("Test accuracy: ");
  Serial.print(correct_count);
  Serial.print("/");
  Serial.print(test_data_cnt);
  Serial.print(" = ");
  Serial.print((float)correct_count / test_data_cnt * 100.0, 2);
  Serial.println("%");
  Serial.println();
  Serial.println("Inference complete! Copy the output above to Python script to generate confusion matrix.");
  Serial.println("======================================");
}

void loop() {
  delay(1000);
}
 
 