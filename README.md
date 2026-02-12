# SARBA-Net
This repository provides the implementation of our paper:
**"SARBA-Net: A Boundary Awareness Network for Impervious Surface Extraction of SAR Images"**

---

## Introduction

Our research is dedicated to exploring **impervious surface (IS) extraction solely from SAR imagery**, aiming to achieve **IS extraction with clear boundary** even under cloudy or rainy conditions when optical data are unavailable.  

The core idea of **SARBA-Net** is to **leverage boundary perception** to guide deep learning models in recognizing impervious surfaces.

---

## Method Highlights

The key contributions of this work include two boundary-aware loss functions:

- **Boundary Contrast Loss (BCL)**  
- **Boundary Consistency Auxiliary Loss (BCAL)**  

📌 Files containing `be_bc` correspond to the implementation of the proposed SARBA-Net.

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Data Preparation
### 1. Download Datasets

#### SARBud Dataset
Download from:

#### WHU-OPT-SAR Dataset
Download from:

Note:

- The original WHU-OPT-SAR is multi-class and has a large image size. We provide a category merging script and a cropping script to preprocess the dataset.

Cropping script location: lib/datasets/process


### 2. Dataset Organization

```bash
dataset_root/
│
├── whusar/
│ ├── image/
│ ├── label/
│
├── sarbud/
│ ├── image/
│ ├── label/
```

### 3. Dataset Split Files

Training, validation, and testing samples are defined in: './SARBA-Net/lib/name_list/'
Example:
```bash
whusar/train.txt
whusar/val.txt
whusar/test.txt
```
The SARBud dataset follows the same structure.

---

## Training and Testing

Run the following command to train the model:
```bash
bash scripts/sarbud/resnet38/run_wr_be_bc.sh train <experiment_name>
```

Run the following command to iference the model:
```bash
bash scripts/sarbud/resnet38/run_wr_be_bc.sh test_max <experiment_name>
```
<experiment_name> is used to distinguish different training runs and to store checkpoints and logs




