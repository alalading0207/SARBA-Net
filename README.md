# SARBA-Net
This repository provides the official implementation of our paper:
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

Clone the repository and install dependencies:

```bash
git clone <your-repository-link>
cd SARBA-Net
pip install -r requirements.txt

---

## Data Preparation
### 1. Download Datasets

#### SARBud Dataset
Download from:

#### WHU-OPT-SAR Dataset
Download from:

Note:

- The original WHU-OPT-SAR images are very large.
- We provide cropping scripts to preprocess the dataset.

Cropping script location:


### 1. Download Datasets
