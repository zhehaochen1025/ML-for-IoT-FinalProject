# Data – Federated Spellcasting Battle System

This folder contains the raw and prepared datasets used for gesture
classification and federated learning experiments.

## 1. Contents
```
data/  
├── data_byh_maxmin_norm.h        # Haoran’s data (raw + per-feature max/min)  
├── data_czh_maxmin_norm.h        # Zhehao’s data (raw + per-feature max/min)  
├── data_fje_maxmin_norm.h        # Jiuen’s data (raw + per-feature max/min)  
│  
├── haoranli-project-1-export/    # Edge Impulse export (Haoran)  
├── jiuenfeng-project-1-export/   # Edge Impulse export (Jiuen)  
├── tangerine-project-1-export/   # Edge Impulse export (Zhehao)  
└── yuhe_b-project-1-export/      # Edge Impulse export (Yuhe)  
```
## 2. Data collection & processing

- Device: **Arduino Nano 33 BLE Sense**  
- Sensors: accelerometer (3), gyroscope (3), orientation (3) → 9D IMU  
- Sampling rate: ~100 Hz, each sample window ≈ 2 s (~200 timesteps)  
- Gesture classes: `circle`, `peak`, `wave`, `other` (idle / non-spell)  

IMU streams were recorded on a PC using the
[Edge Impulse CLI serial daemon](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli/serial-daemon),
which forwarded data from the Nano 33 BLE Sense over USB into Edge Impulse Studio.
In Edge Impulse, the data was segmented into ~2-second windows and each window
was labelled with one of the four gesture classes.

`data_prepare.ipynb` (in the repository root) reads data from the
`*-project-1-export/` folders, reshapes the sequences, computes per-feature
max/min values, and generates the `data_*_maxmin_norm.h` header files used by
the Arduino training/inference code (BP, distributed DNN on device, and test
scripts).

## 3. maxmin_norm.h files

Files named `*_maxmin_norm.h` contain:
- the **raw IMU samples**, and  
- the corresponding **per-feature max and min values** for each dataset.

The header files themselves **do not perform normalization**.  
In the final version of this project, the Arduino models were trained and
evaluated on the **raw values** (no normalization was applied on-device),
but the stored max/min values can be used if normalization is added later.
