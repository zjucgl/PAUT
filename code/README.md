# Geometry-Aware Pseudo-Defect Recognition in Phased Array Ultrasonic Testing via Continuous Distance Field Encoding

This repository contains the official implementation of our paper:
**"Geometry-Aware Pseudo-Defect Recognition in Phased Array Ultrasonic Testing via Continuous Distance Field Encoding"** (Submitted to *Structural Durability & Health Monitoring*).

---

## 📦 Requirements

The code is tested with the following dependencies:

```
h5py==3.8.0
matplotlib==3.5.3
numpy==1.20.3
opencv_python==4.10.0.84
opencv_python_headless==4.10.0.84
pandas==1.3.5
Pillow==8.4.0
PyWavelets==1.3.0
scikit_learn==1.0.2
scipy==1.7.3
seaborn==0.13.2
torch==1.13.1+cu117
torchvision==0.14.1+cu117
tqdm==4.67.1
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

The dataset is placed in the `./datasets` folder.

* **datasets-raw-s**: original S-scan images.
* **datasets-raw-s-weld**: cropped S-scan images based on weld boundaries and heat-affected zones.

Each folder contains four subfolders representing different classes:

```
datasets/
├── nd/
├── pd/
└── td/

```

Each subfolder contains `.png` format images for the corresponding class.

---

## 🚀 Training

Run the following command to train the model:

```bash
python train.py --name test --net resnet18 --pre
```

Additional common options (examples):

```bash
# 使用预训练权重并指定输出目录
python train.py --name exp1 --net resnet18 --pre

# 指定训练超参数示例
python train.py --name exp_lr --net new --pre --lr 0.01 --batch_size 32 --epochs 100 --datasets 1 --extra
```

---

## 🔍 Testing

Run the following command to evaluate the model:

```bash
python test.py --name test --net resnet18
```

---

## 📖 Results

Experimental results reported in the submitted manuscript

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{he2025misalignment,
  title={A Misalignment-Robust PAUT Geometric Artifact Identification Model Integrating Weld Geometric Priors and Distance Descriptors},
  author={He, BinHua and Chen, Genlang and Song, Guanhui},
  journal={Journal of Advanced Mechanical Design, Systems, and Manufacturing},
  year={2025}
}
```

---

## 👥 Authors

* **BinHua He**
  School of Computer Science and Technology, Zhejiang Sci-Tech University
  Hangzhou, 310018, China
  Email: [2023220603022@mails.zstu.edu.cn](mailto:2023220603022@mails.zstu.edu.cn)

* **Genlang Chen**
  School of Computer Science and Data Engineering, NingboTech University
  Ningbo, 315100, China
  Email: [cgl@zju.edu.cn](mailto:cgl@zju.edu.cn)

* **Guanhui Song**
  School of Computer Science and Data Engineering, NingboTech University
  Ningbo, 315100, China
  Email: [songgh@nbt.edu.cn](mailto:songgh@nbt.edu.cn)

---

## 📧 Contact

For any questions, please contact:
**BinHua He** - [2023220603022@mails.zstu.edu.cn](mailto:2023220603022@mails.zstu.edu.cn)

