/*
 * 测试集推理 - 用于在测试集上评估模型性能
 * 
 * 功能：
 * 1. 加载训练好的模型权重
 * 2. 对测试集中的每个样本进行推理
 * 3. 输出真实标签和预测标签（格式：真实标签,预测标签）
 * 
 * 使用方法：
 * 1. 将训练后保存的权重数组粘贴到下面的 trained_weights 数组中
 * 2. 上传到 Arduino
 * 3. 打开串口监视器查看结果
 * 4. 复制输出结果到 Python 脚本生成混淆矩阵
 */

 #include<stdio.h>
 #include<stdlib.h>
 #include<math.h>
 #include<TinyMLShield.h>
 
 // NN parameters（需要与训练时一致）
 #define LEARNING_RATE 0.001
 #define EPOCH 50
 #define DATA_TYPE_FlOAT
 
 
// 网络结构（需要与训练时一致）

 #include "data.h"       // 包含测试数据和标签（需要 test_data, test_labels, test_data_cnt）
 static const unsigned int NN_def[] = {first_layer_input_cnt, 64, classes_cnt};
 #include "NN_functions.h"   // 神经网络函数
 #include "inference_byh.h"      // 训练好的权重数组 

 
 // ========== 将训练后保存的权重数组粘贴到这里 ==========
 // 从 BP.ino 训练完成后的串口输出中复制权重数组
 // 类别名称映射
 const char* class_names[] = {
   "circle",  // 0
   "other",   // 1
   "peak",    // 2
   "wave"     // 3
 };
 
 void setup() {
   Serial.begin(9600);
   delay(5000);
   while (!Serial);
   
   // 初始化 TinyML Shield
   initializeShield();
   
   Serial.println("======================================");
   Serial.println("测试集推理程序");
   Serial.println("======================================");
   Serial.println();
   
   // 计算权重数量
   int weights_bias_cnt = calcTotalWeightsBias();
   Serial.print("权重数量: ");
   Serial.println(weights_bias_cnt);
   
   // 检查权重数组是否为空（如果数组只有一个元素且为0，可能表示未初始化）
   int weights_array_size = sizeof(trained_weights) / sizeof(trained_weights[0]);
   if (weights_array_size == 0 || (weights_array_size == 1 && trained_weights[0] == 0.0f)) {
     Serial.println("错误：权重数组为空或未初始化！");
     Serial.println("请先运行 BP.ino 训练模型，然后将权重数组复制到这里。");
     Serial.print("当前权重数组大小: ");
     Serial.println(weights_array_size);
     while(1); // 停止执行
   }
   
   Serial.print("权重数组大小: ");
   Serial.println(weights_array_size);
   
   // 分配权重内存
   DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));
   
   // 设置网络
   setupNN(WeightBiasPtr);
   
   // 加载训练好的权重
   Serial.println("正在加载模型权重...");
   loadModel(trained_weights);
   
   Serial.println();
   Serial.println("开始测试集推理...");
   Serial.println("输出格式：样本索引,真实标签,预测标签,真实标签名称,预测标签名称");
   Serial.println("--------------------------------------");
   
   // 对测试集中的每个样本进行推理
   int correct_count = 0;
   for (int i = 0; i < test_data_cnt; i++) {
     // 获取真实标签
     int true_label = test_labels[i];
     
     // 执行推理
     int predicted_label = inference(test_data[i]);
     
     // 统计正确预测
     if (predicted_label == true_label) {
       correct_count++;
     }
     
     // 输出结果：真实标签,预测标签（用于Python脚本处理）
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
   Serial.print("测试集准确率: ");
   Serial.print(correct_count);
   Serial.print("/");
   Serial.print(test_data_cnt);
   Serial.print(" = ");
   Serial.print((float)correct_count / test_data_cnt * 100.0, 2);
   Serial.println("%");
   Serial.println();
   Serial.println("推理完成！请复制上面的输出结果到Python脚本生成混淆矩阵。");
   Serial.println("======================================");
 }
 
 void loop() {
   // 推理只执行一次，完成后停止
   // 如需重新运行，请按复位键
   delay(1000);
 }
 
 