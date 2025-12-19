# Federated Spellcasting Battle System

Team Members: Haoran Li, Jiuen Feng,  Zhehao Chen, Yuhe Bian

## Usage Instructions

### data_prepare.ipynb
- Description: Preprocess raw data from Edge Impulse
- Output: Normalized dataset with shape (75,)

### PC_training/
- Description: PC-based training scripts for centralized and federated learning

### BP/
- Description: Backpropagation training on Arduino (based on tutorial 4)
- Steps:
  1. Run `data_prepare.ipynb` to generate `data.h`
  2. Move `data.h` to BP folder and update `#include "data_byh.h"`
  3. Upload to Arduino, copy weights from serial output, save as `trained_weights[]` in `inference.h`

### BP_test_inference/
- Description: Test BP models on test dataset
- Steps:
  1. Copy the generated `inference.h` and dataset to this folder
  2. Update data path and weights path in `BP_test_inference.ino`
  3. Run `BP_test_inference.ino`

### IMU_Classifier/
- Description: Live demo for BP model inference
- Steps:
  1. Copy the generated `inference.h` to this folder
  2. Upload `IMU_Classifier.ino` to Arduino
  3. Press button to start recording (2 seconds), inference result will be output

### distributed_dnn_on_device/
- Description: Federated learning on two Arduino devices via BLE (based on https://github.com/niil87/Machine-Learning-for-IOT---Fall-2022-Batch-Lund-University)
- Steps:
  1. Copy two datasets to this folder
  2. Prepare two Arduino boards, set `DEVICE_TYPE` to `WORKER` and `LEADER` respectively
  3. Update dataset paths for WORKER and LEADER in the code
  4. Upload `distributed_dnn_on_device.ino` with corresponding `DEVICE_TYPE` to each device
  5. Save the trained weights as `trained_weights[]` in `inference.h`

### distributed_test/
- Description: Test federated learning model on test dataset
- Steps:
  1. Copy the generated `inference.h` and test dataset to this folder
  2. Update data path and weights path in `distributed_test.ino`
  3. Run `distributed_test.ino`

### distributed_classifier/
- Description: Live demo for federated learning model inference
- Steps:
  1. Copy the generated `inference.h` to this folder
  2. Upload `distributed_classifier.ino` to Arduino
  3. Press button to start recording (2 seconds), inference result will be output

### battle_fighter / battle_fighter2 / battle_center/
- Description: Real-time spellcasting battle system
- Steps:
  1. Copy the trained `inference.h` to both `battle_fighter/` and `battle_fighter2/` folders
  2. Upload `battle_fighter.ino` and `battle_fighter2.ino` to two separate Arduino boards
  3. Close serial monitor in Arduino IDE
  4. Update `SERIAL_PORT` in `battle_center/battle_visualizer.py` to match the board running `battle_fighter2.ino`
  5. Press reset buttons on both boards to initialize timers, then run `battle_center/battle_visualizer.py`
  6. A valid battle occurs when both boards press buttons and capture valid gestures within 2 seconds

### Results
Confusion Matrix:
BP_test_inference/output/confusion_matrix_mbyh_dbyh.png
BP_test_inference/output/confusion_matrix_mbyh_dfje.png
BP_test_inference/output/confusion_matrix_mbyh_dczh.png
BP_test_inference/output/confusion_matrix_mfje_dfje.png
BP_test_inference/output/confusion_matrix_mfje_dbyh.png
BP_test_inference/output/confusion_matrix_mfje_dczh.png

## Project Directory Structure

```
ML-for-IoT-FinalProject/
├── README.md
├── data_prepare.ipynb                    # Data preprocessing script
├── demo.mp4                              # Demo video
├── plot_fd_leader.png                    # Federated learning training curve
│
├── battle_center/                        # Battle center for battle visualization
│   └── battle_visualizer.py
│
├── battle_fighter/                       # Battle device 1
│   ├── battle_fighter.ino
│   ├── inference.h
│   └── NN_functions.h
│
├── battle_fighter2/                      # Battle device 2 as central
│   ├── battle_fighter2.ino
│   ├── inference.h
│   └── NN_functions.h
│
├── BP/                                   # Backpropagation training
│   ├── BP.ino
│   ├── convert_to_tflite.py
│   ├── data_byh.h
│   ├── data_fje.h
│   └── NN_functions.h
│
├── BP_test_inference/                    # BP inference on test dataset
│   ├── BP_test_inference.ino
│   ├── data_byh.h
│   ├── data_czh.h
│   ├── data.h
│   ├── generate_confusion_matrix.py
│   ├── inference_byh.h
│   ├── inference_fje.h
│   ├── log_byh.txt
│   ├── log_fje.txt
│   ├── NN_functions.h
│   ├── plot_training.py
│   └── output/                           # Test results output
│       ├── acc_plot_byh.png
│       ├── acc_plot_fje.png
│       ├── confusion_matrix_*.png
│       └── result_*.txt
│
├── data/                                 # Processed Data directory
│   ├── data_byh_maxmin_norm.h
│   ├── data_czh_maxmin_norm.h
│   ├── data_fje_maxmin_norm.h
│   ├── imu_raw/                          # Raw IMU data
│   │   ├── ei-tangerine-project-1-syntiant-imu-X_testing.4.npy
│   │   ├── ei-tangerine-project-1-syntiant-imu-X_training.4.npy
│   │   ├── ei-tangerine-project-1-syntiant-imu-y_testing.4.npy
│   │   └── ei-tangerine-project-1-syntiant-imu-y_training.4.npy
│   ├── haoranli-project-1-export/        # User data export
│   ├── jiuenfeng-project-1-export/
│   ├── tangerine-project-1-export/
│   └── yuhe_b-project-1-export/
│
├── distributed_classifier/               # Distributed inference(Live demo)
│   ├── distributed_classifier.ino
│   ├── data_byh.h
│   ├── data_czh.h
│   ├── data_fje.h
│   ├── inference.h
│   └── NN_functions.h
│
├── distributed_dnn_on_device/            # Distributed DNN on device (based on https://github.com/niil87/Machine-Learning-for-IOT---Fall-2022-Batch-Lund-University)
│   ├── distributed_dnn_on_device.ino
│   ├── BLE_central.h                     # BLE central device
│   ├── BLE_peripheral.h                  # BLE peripheral device
│   ├── data_byh.h
│   ├── data_fje.h
│   ├── inference.h
│   └── NN_functions.h
│
├── distributed_test/                     # Distributed learning model inference on test dataset
│   ├── distributed_test.ino
│   ├── data_byh.h
│   ├── data_czh.h
│   ├── data_fje.h
│   ├── generate_confusion_matrix.py
│   ├── inference.h
│   ├── log_fd_leader.txt
│   ├── log_worker.txt
│   ├── NN_functions.h
│   ├── plot_fd_leader.png
│   ├── plot_fd_training.py
│   ├── plot_fd_worker.png
│   └── output/                           # Test results output
│
├── IMU_Classifier/                       # BP inference (live demo)
│   ├── IMU_Classifier.ino
│   ├── inference.h
│   ├── model_old.h
│   └── NN_functions.h
│
├── PC_training/                          # PC training scripts
│   ├── central_training_pc.py         
│   ├── fed_training_mpi.py         
│   ├── federated_mpi.py               
│   ├── plot_data_distribution.py      
│   ├── plot_training_curves.py        
│   ├── training_pc_numpy.py           
│   ├── training_pc.py                    
│   └── data/                            
│
├── Slides/                                # Presentation slides
│   ├── slides.pdf
│   ├── slides.tex
│   ├── demo.mp4
│   └── figures/                          # Figures
│
├── jiuenfeng-project-1-custom-v1/        # Edge Impulse deployment
│   ├── edge-impulse-sdk/         
│   ├── model-parameters/               
│   ├── tflite-model/                    
│   ├── trained.h5.zip
│   ├── trained.savedmodel.zip
│   └── trained.tflite
│
└── yuhe_b-project-1-custom-v1/           # Edge Impulse deployment
    ├── edge-impulse-sdk/
    ├── model-parameters/
    ├── tflite-model/
    ├── trained.h5.zip
    └── trained.savedmodel.zip
```
