# SEDS-YOLOv8: Enhanced YOLOv8 for Floating Object Detection


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Official implementation of **"Enhanced YOLOv8 for Accurate and Efficient Floating Object Detection on Water Surfaces"** ( FROM *The Visual Computer*).  

SEDS-YOLOv8 is an optimized YOLOv8 variant designed for accurate and efficient detection of floating objects on water surfaces, addressing challenges like reflections, waves, and low-contrast targets.
---

## 📌 Features
- **SEDSConv Module**: Integrates SE Attention and DSConv for efficient feature extraction
- **EIoU Loss**: Enhanced localization accuracy for cluttered water surfaces
- **Lightweight Design**: 2.9M parameters, 7.6 GFLOPs, 103.69 FPS on RTX4090
- **Robust Performance**: 88.82% mAP@0.5 on aquatic debris detection

---

## 🛠️ Installation
```bash
# Clone repo
git clone https://github.com/hackerjackL/SEDS-YOLOv8.git
cd SEDS-YOLOv8

# Install dependencies (Python 3.8+ required)
example: pip install ultralytics pytorch torchvivsion 

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

📦 Dataset Preparation
Dataset Structure
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
Dataset Extraction
Windows:
Install 7-Zip

Combine split files:

powershell
copy /b dataset.z01 + dataset.z02 + ... + dataset.zip combined.zip
7z x combined.zip

Linux:
# Install p7zip
sudo apt install p7zip-full

# Combine and extract
cat dataset.z* > combined.zip
7z x combined.zip

Dataset Configuration
nc:9
names: ['can', 'leaf', 'branch', 'grass', 'bottle', 'plastic-box', 
        'milk-box', 'plastic-bag', 'paper']
🏗️ Model Architecture
Key modules located in:
ultralytics/
├── nn/
│   ├── modules/
│   │   ├── seattention.py   # SE Attention implementation
│   │   └── dsconv.py        # DSConv implementation
└── cfg/
    └── models/
        └── v8/
            └── SEDS-YOLOv8.yaml  # Model configuration
🚀 Training
YOLO CLI Command
This is an example:
yolo detect train \
  data=mydata.yaml \
  model=ultralytics/cfg/models/v8/SEDS-YOLOv8.yaml \
  epochs=300 \
  imgsz=640 \
  batch=32 \
  device=0 \
  optimizer=AdamW \
  lr0=0.001 \
  weight_decay=0.0005 \
  cache=disk \
  name=SEDS-YOLOv8_training
