#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<TinyMLShield.h>

// NN parameters, set these yourself! 
#define LEARNING_RATE 0.0015    // The learning rate used to train your network
#define EPOCH 50         // The maximum number of epochs 
#define DATA_TYPE_FLOAT      // The data type used: Set this to DATA_TYPE_DOUBLE for higher precision. However, it is better to keep this Float if you want to submit the result via BT

extern const int first_layer_input_cnt;
extern const int classes_cnt;

// You define your network in NN_def
// Right now, the network consists of three layers: 
// 1. An input layer with the size of your input as defined in the variable first_layer_input_cnt in cnn_data.h 
// 2. A hidden layer with 20 nodes
// 3. An output layer with as many classes as you defined in the variable classes_cnt in cnn_data.h 
static const unsigned int NN_def[] = {first_layer_input_cnt, 64, classes_cnt};

#include "data.h"       // The data, labels and the sizes of all objects are stored here 
#include "NN_functions.h"   // All NN functions are stored here 

int iter_cnt = 0;           // This keeps track of the number of epochs you've trained on the Arduino
#define DEBUG 0             // This prints the weights of your network in case you want to do debugging (set to 1 if you want to see that)

// This function contains your training loop 
void do_training() {

  // Print the weights if you want to debug 
#if DEBUG      
  Serial.println("Now Training");
  PRINT_WEIGHTS();
#endif

  // Print the epoch number 
  Serial.print("Epoch count (training count): ");
  Serial.print(++iter_cnt);
  Serial.println();

  // reordering the index for more randomness and faster learning
  shuffleIndx();
  
  // starting forward + Backward propagation
  for (int j = 0;j < numTrainData;j++) {
    generateTrainVectors(j);  
    forwardProp();
    backwardProp();
  }

  Serial.println("Accuracy after local training:");
  printAccuracy();

}


void setup() {
  // put your setup code here, to run once:
  
  // Initialize random seed 
  srand(0); 
  
  Serial.begin(9600); 
  delay(5000);
  while (!Serial); 

  // Initialize the TinyML Shield 
  initializeShield();

  // Calculate how many weights and biases we're training on the device. 
  int weights_bias_cnt = calcTotalWeightsBias(); 

  Serial.print("The total number of weights and bias used for on-device training on Arduino: ");
  Serial.println(weights_bias_cnt);

  // Allocate common weight vector, and pass to setupNN, setupBLE
  DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));

  setupNN(WeightBiasPtr);  // CREATES THE NETWORK BASED ON NN_def[]
  Serial.print("The accuracy before training");
  printAccuracy();
  
  Serial.println("Use the on-shield button to start and stop the loop code ");
  
}

void loop() {
  // put your main code here, to run repeatedly:

  // see if the button is pressed and turn off or on recording accordingly
  bool clicked = readShieldButton();
  // Serial.println("Please Click the Button");
  if (iter_cnt < EPOCH){
    
    // Serial.println("yes, we clicked the button");
    do_training(); // Local training 
    
  } else if (iter_cnt == EPOCH) {
    // 训练完成后保存模型
    Serial.println("\n训练完成！正在保存模型...");
    saveModel(iter_cnt);
    Serial.println("模型已保存到串口输出。请复制权重数据。");
    Serial.println("按复位键重新开始训练。");
    iter_cnt++;  // 防止重复保存
  }

}
