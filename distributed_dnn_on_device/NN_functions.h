// Code developed by Nikhil Challa as part of ML in IOT Course - year : 2022
// Team members : Simon Erlandsson
// To get full understanding of the code, please refer to below document in git
// https://github.com/niil87/Machine-Learning-for-IOT---Fall-2022-Batch-Lund-University/blob/main/Math_for_Understanding_Deep_Neural_Networks.pdf
// The equations that corresponds to specific code will be listed in the comments next to code

#define fRAND ( rand()*1.0/RAND_MAX-0.5 )*2   // random number generator between -1 and +1 
#define ACT(a) max(a,0)    // RELU(a)


#ifdef DATA_TYPE_FLOAT 
  #define DATA_TYPE float
  #define EXP_LIMIT 78.0  // limit 88.xx but we need to factor in accumulation for softmax
  #define EXP(a) expl(a)
#else
  #define DATA_TYPE double
  #define EXP_LIMIT 699.0 // limit is 709.xx but we need to factor in accumulation for softmax
  #define EXP(a) exp(a)
#endif

#define IN_VEC_SIZE first_layer_input_cnt
#define OUT_VEC_SIZE classes_cnt

// size of different vectors
size_t numTestData = test_data_cnt;
size_t numValData = validation_data_cnt;
size_t numTrainData = train_data_cnt;


size_t numLayers = sizeof(NN_def) / sizeof(NN_def[0]);
// size of the input to NN


// dummy input for testing
DATA_TYPE input[IN_VEC_SIZE];

// dummy output for testing
DATA_TYPE hat_y[OUT_VEC_SIZE];    // target output
DATA_TYPE y[OUT_VEC_SIZE];        // output after forward propagation


// creating array index to randomnize order of training data
int indxArray[train_data_cnt];

// 归一化已移除 - 输入数据应该是已经归一化的

// Convention: Refer to 
typedef struct neuron_t {
	int numInput;
	DATA_TYPE* W;
	DATA_TYPE B;
	DATA_TYPE X;

	// For back propagation, convention, dA means dL/dA or partial derivative of Loss over Accumulative output
	DATA_TYPE* dW;
	DATA_TYPE dA;
	DATA_TYPE dB;

} neuron;

typedef struct layer_t {
	int numNeuron;
	neuron* Neu;
} layer;

// initializing the layer as global parameter
layer* L = NULL;

// Weights written to here will be sent/received via bluetooth. 
DATA_TYPE* WeightBiasPtr = NULL;

// Equation (8)
DATA_TYPE AccFunction (unsigned int layerIndx, int nodeIndx) {
	DATA_TYPE A = 0;

	for (int k = 0; k < NN_def[layerIndx - 1]; k++) {

	// updating weights/bais and resetting gradient value if non-zero
	if (L[layerIndx].Neu[nodeIndx].dW[k] != 0.0 ) {
		L[layerIndx].Neu[nodeIndx].W[k] += L[layerIndx].Neu[nodeIndx].dW[k];
		L[layerIndx].Neu[nodeIndx].dW[k] = 0.0;
	}

	A += L[layerIndx].Neu[nodeIndx].W[k] * L[layerIndx - 1].Neu[k].X;

	}

	if (L[layerIndx].Neu[nodeIndx].dB != 0.0 ) {
		L[layerIndx].Neu[nodeIndx].B += L[layerIndx].Neu[nodeIndx].dB;
		L[layerIndx].Neu[nodeIndx].dB = 0.0;
	}
	A += L[layerIndx].Neu[nodeIndx].B;

	return A;
}


// NEED HANDLING TO ENSURE NO WEIGHTS AND BIAS ARE CREATED FOR FIRST LAYER OR THROW ERROR IF ACCESSED ACCIDENTLY
// EVEN THOUGH WE HAVENT EXPLICITLY CALLED CREATED THE NEURON FOR FIRST LAYER, HOW WAS L[i].Neu[j].X SUCCESSFUL IN FORWARD PROPAGATION!!
neuron createNeuron(int numInput) {

	neuron N1;

	N1.B = fRAND;
	N1.numInput = numInput;
	N1.W = (DATA_TYPE*)calloc(numInput, sizeof(DATA_TYPE));
	N1.dW = (DATA_TYPE*)calloc(numInput, sizeof(DATA_TYPE));
	// initializing values of W to rand and dW to 0
	//int Sum = 0;
	for (int i = 0; i < numInput; i++) {
		N1.W[i] = fRAND / sqrt((float)numInput);
		N1.dW[i] = 0.0;
	}
	N1.dA = 0.0;
	N1.dB = 0.0;

	return N1;

}

layer createLayer (int numNeuron) {
	layer L1;
	L1.numNeuron = numNeuron;
	L1.Neu = (neuron*)calloc(numNeuron, sizeof(neuron));
	return L1;
}

void createNetwork() {

	L = (layer*)calloc(numLayers, sizeof(layer));

	// First layer has no input weights
	L[0] = createLayer(NN_def[0]);

	for (unsigned int i = 1; i < numLayers; i++) {
		L[i] = createLayer(NN_def[i]);
		for (unsigned int j = 0; j < NN_def[i]; j++) {
			L[i].Neu[j] = createNeuron(NN_def[i - 1]);
		}
	}

	// creating indx array for shuffle function to be used later
	for (unsigned int i = 0; i <  numTrainData; i ++ ) {
		indxArray[i] = i;
	}

}


// this function is to calculate dA
DATA_TYPE dLossCalc( unsigned int layerIndx, unsigned int nodeIndx) {

	DATA_TYPE Sum = 0;
	// int outputSize = NN_def[numLayers - 1];
	// for the last layer, we use complex computation
	if (layerIndx == numLayers - 1) {	
		Sum = y[nodeIndx] - hat_y[nodeIndx];										// Equation (17)
	// for all except last layer, we use simple aggregate of dA
	} else if (AccFunction(layerIndx, nodeIndx) > 0)  {   							
		for (unsigned int i = 0; i < NN_def[layerIndx + 1]; i++) {
			Sum += L[layerIndx + 1].Neu[i].dA * L[layerIndx + 1].Neu[i].W[nodeIndx]; 	// Equation (24)
		}
	} else {   																		// refer to "Neat Trick" and Equation (21)
		Sum = 0;
	}

	return Sum;
}

void forwardProp()
{
	
	DATA_TYPE Fsum = 0;
	int maxIndx = 0;
	// Propagating through network
	for (unsigned int i = 0; i < numLayers; i++) {
		// assigning node values straight from input for first layer
		if (i == 0) {
			for (unsigned int j = 0; j < NN_def[0];j++) {
				L[i].Neu[j].X = input[j];
			}
		} else if (i == numLayers - 1) {
      // softmax functionality but require normalizing performed later
			for (unsigned int j = 0; j < NN_def[i];j++) {
				y[j] = AccFunction(i,j);
				// tracking the max index
				if ( ( j > 0 ) && (abs(y[maxIndx]) < abs(y[j])) ) {
					maxIndx = j;
				}
			}
		} else {	
			// for subsequent layers, we need to perform RELU
			for (unsigned int j = 0; j < NN_def[i];j++) {
				L[i].Neu[j].X = ACT(AccFunction(i,j));				// Equation (21)	
			}	
		}
	}

  // performing exp but ensuring we dont exceed 709 or 88 in any terms 
	DATA_TYPE norm = abs(y[maxIndx]);
	if (norm > EXP_LIMIT) {
#if DEBUG
		Serial.print("Max limit exceeded for exp:");
		Serial.println(norm);
#endif
		norm = norm / EXP_LIMIT;
#if DEBUG
		Serial.print("New divising factor:");
		Serial.println(norm);
#endif
	} else {
		norm = 1.0;
	}
	for (unsigned int j = 0; j < NN_def[numLayers-1];j++) {
		// int flag = 0;
		y[j] = EXP(y[j]/norm);
		Fsum += y[j];
	}

  // final normalizing for softmax
	for (unsigned int j = 0; j < NN_def[numLayers-1];j++) {
		y[j] = y[j]/Fsum;
	}
}

void backwardProp() {
	for (unsigned int i = numLayers - 1; i > 1; i--) {
    // tracing each node in the layer.
		for (unsigned int j = 0; j < NN_def[i]; j++) {
		// first checking if drivative of activation function is 0 or not! NEED TO UPGRADE TO ALLOW ACTIVATION FUNCTION OTHER THAN RELU
		L[i].Neu[j].dA = dLossCalc(i, j);

		for (int k = 0; k < NN_def[i - 1]; k++) {
			L[i].Neu[j].dW[k] = -LEARNING_RATE * L[i].Neu[j].dA * L[i - 1].Neu[k].X;
		}
		L[i].Neu[j].dB = -LEARNING_RATE * L[i].Neu[j].dA;
    }
  }
}


// function to set the input and output vectors for training or inference
void generateTrainVectors(int indx) {

	// Train Data
	for (unsigned int j = 0; j < OUT_VEC_SIZE; j++) {
		hat_y[j] = 0.0;
	}
	hat_y[ train_labels[ indxArray[indx] ] ] = 1.0;

	for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
		input[j] = train_data[ indxArray[indx] ][j];
	}
}

void shuffleIndx()
{
  for (unsigned int i = 0; i < train_data_cnt - 1; i++)
  {
    size_t j = i + rand() / (RAND_MAX / (train_data_cnt - i) + 1);
    unsigned int t = indxArray[j];
    indxArray[j] = indxArray[i];
    indxArray[i] = t;
  }
}

int calcTotalWeightsBias()
{
	int Count = 0;
	for (unsigned int i = 0; i < numLayers - 1; i++) {
		Count += NN_def[i] * NN_def[i + 1] + NN_def[i + 1];
	}

	return Count;
}

// --- [修改 5] 在计算准确率时也要归一化 ---
void printAccuracy() {
  int correctCount = 0;

  // 1. 训练集
  for (unsigned int i = 0; i < numTrainData; i++) {
    int maxIndx = 0;
    for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = train_data[i][j];
    }
    forwardProp();
    for (unsigned int j = 1; j < OUT_VEC_SIZE; j++) {
      if (y[maxIndx] < y[j]) maxIndx = j;
    }
    if (maxIndx == train_labels[i]) correctCount += 1;
  }
  Serial.print("Training Accuracy: ");
  Serial.println(correctCount * 1.0 / numTrainData);

  // 2. 验证集
  correctCount = 0;
  for (unsigned int i = 0; i < numValData; i++) {
    int maxIndx = 0;
    for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = validation_data[i][j];
    }
    forwardProp();
    for (unsigned int j = 1; j < OUT_VEC_SIZE; j++) {
      if (y[maxIndx] < y[j]) maxIndx = j;
    }
    if (maxIndx == validation_labels[i]) correctCount += 1;
  }
  Serial.print("Validation Accuracy: ");
  Serial.println(correctCount * 1.0 / numValData);

  // 3. 测试集
  correctCount = 0;
  for (unsigned int i = 0; i < numTestData; i++) {
    int maxIndx = 0;
    for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = test_data[i][j];
    }
    forwardProp();
    for (unsigned int j = 1; j < OUT_VEC_SIZE; j++) {
      if (y[maxIndx] < y[j]) maxIndx = j;
    }
    if (maxIndx == test_labels[i]) correctCount += 1;
  }
  Serial.print("Test Accuracy: ");
  Serial.println(correctCount * 1.0 / numTestData);
}


#define PACK 0
#define UNPACK 1
#define AVERAGE 2
// 0 -> pack vector for creating vector based on local NN for bluetooth transmission
// 1 -> unpack vector for updating weights on local NN after receiving vector via bluetooth transmission
// 2 -> average between values in pointer and location network values, and update both local NN and pointer value
void packUnpackVector(int Type)
{
  unsigned int ptrCount = 0;
  if (Type == PACK) {
    // Propagating through network, we store all weights first and then bias.
    // we start with left most layer, and top most node or lowest to highest index
    for (unsigned int i = 1; i < numLayers; i++) {
      for (unsigned int j = 0; j < NN_def[i]; j++) {
        for (unsigned int k = 0; k < L[i].Neu[j].numInput; k++) {
          WeightBiasPtr[ptrCount] = L[i].Neu[j].W[k];
          ptrCount += 1;
        }
        WeightBiasPtr[ptrCount] = L[i].Neu[j].B;
        ptrCount += 1;
      }
    }

    //Serial.print("Total count when packing:");
    //Serial.println(ptrCount);

  } else if (Type == UNPACK) {
    // Propagating through network, we store all weights first and then bias.
    // we start with left most layer, and top most node or lowest to highest index
    for (unsigned int i = 1; i < numLayers; i++) {
      for (unsigned int j = 0; j < NN_def[i]; j++) {
        for (unsigned int k = 0; k < L[i].Neu[j].numInput; k++) {
          L[i].Neu[j].W[k] = WeightBiasPtr[ptrCount];
          ptrCount += 1;
        }
        L[i].Neu[j].B = WeightBiasPtr[ptrCount];
        ptrCount += 1;
      }
    }
  } else if (Type == AVERAGE) {
    // Propagating through network, we store all weights first and then bias.
    // we start with left most layer, and top most node or lowest to highest index
    for (unsigned int i = 1; i < numLayers; i++) {
      for (unsigned int j = 0; j < NN_def[i]; j++) {
        for (unsigned int k = 0; k < L[i].Neu[j].numInput; k++) {
          L[i].Neu[j].W[k] = (WeightBiasPtr[ptrCount] + L[i].Neu[j].W[k] ) / 2;
          WeightBiasPtr[ptrCount] = L[i].Neu[j].W[k];
          ptrCount += 1;
        }
        L[i].Neu[j].B = (WeightBiasPtr[ptrCount] + L[i].Neu[j].B ) / 2;
        WeightBiasPtr[ptrCount] = L[i].Neu[j].B;
        ptrCount += 1;
      }
    }
  }
}

// Called from main in setup-function
void setupNN(DATA_TYPE* wbptr) {
  WeightBiasPtr = wbptr;
  createNetwork();
}

// 保存模型权重到串口
void saveModel(int epochs_trained) {
  // 首先将网络权重打包到 WeightBiasPtr
  packUnpackVector(PACK);
  
  // 计算权重总数
  int weights_bias_cnt = calcTotalWeightsBias();
  
  Serial.println("\n========== 模型权重保存 ==========");
  Serial.print("总权重和偏置数量: ");
  Serial.println(weights_bias_cnt);
  
  // 输出模型元数据（JSON格式，方便解析）
  Serial.println("\n// 模型元数据 (JSON格式):");
  Serial.print("// MODEL_METADATA:{\"layers\":[");
  for (unsigned int i = 0; i < numLayers; i++) {
    Serial.print(NN_def[i]);
    if (i < numLayers - 1) Serial.print(",");
  }
  Serial.print("],\"learning_rate\":");
  Serial.print(LEARNING_RATE, 6);
  Serial.print(",\"epochs\":");
  Serial.print(epochs_trained);
  Serial.print(",\"weights_count\":");
  Serial.print(weights_bias_cnt);
  Serial.println("}");
  
  // 输出C数组格式的权重数据
  Serial.println("\n// 权重数组 (可直接复制到 inference.h 中使用):");
  Serial.println("const float trained_weights[] = {");
  Serial.print("  ");
  
  // 输出权重数据，每行最多10个值
  for (int i = 0; i < weights_bias_cnt; i++) {
    Serial.print(WeightBiasPtr[i], 6);  // 6位小数精度
    if (i < weights_bias_cnt - 1) {
      Serial.print("f, ");
      if ((i + 1) % 10 == 0) {  // 每10个值换行
        Serial.println();
        Serial.print("  ");
      }
    } else {
      Serial.print("f");
    }
  }
  
  Serial.println("\n};");
  Serial.println("==================================\n");
  Serial.println("提示: 复制上面的权重数组到 inference.h 文件中，然后使用 loadModel() 函数加载。\n");
}

// 从权重数组加载模型
// 参数: weights_array - 保存的权重数组指针
void loadModel(const float* weights_array) {
  // 将权重数组复制到 WeightBiasPtr
  int weights_bias_cnt = calcTotalWeightsBias();
  for (int i = 0; i < weights_bias_cnt; i++) {
    WeightBiasPtr[i] = weights_array[i];
  }
  
  // 使用 UNPACK 将权重加载到网络结构中
  packUnpackVector(UNPACK);
  
  Serial.println("模型权重已成功加载！");
}

// 推理函数：对输入数据进行预测
// 参数: input_data - 输入特征向量（75维）
// 返回: 预测类别的索引
int inference(const float* input_data) {
  // 1. 将输入数据复制到 input 数组
  for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
    input[j] = input_data[j];
  }
  
  // 2. 前向传播（输入数据应该是已经归一化的）
  forwardProp();
  
  // 4. 找到最大概率的类别
  int maxIndx = 0;
  for (unsigned int j = 1; j < OUT_VEC_SIZE; j++) {
    if (y[maxIndx] < y[j]) {
      maxIndx = j;
    }
  }
  
  return maxIndx;
}

// 推理函数（带概率输出）
// 参数: input_data - 输入特征向量（75维）
//       probabilities - 输出概率数组（需要预先分配，大小为 classes_cnt）
// 返回: 预测类别的索引
int inferenceWithProbabilities(const float* input_data, float* probabilities) {
  // 执行推理
  int predicted_class = inference(input_data);
  
  // 复制概率值
  for (unsigned int i = 0; i < OUT_VEC_SIZE; i++) {
    probabilities[i] = y[i];
  }
  
  return predicted_class;
}
