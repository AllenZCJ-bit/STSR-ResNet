# STSR-ResNet: Spatiotemporal Interpolation for Meteorological Data

This repository contains the official PyTorch implementation of the paper:  
**"Spatiotemporal Interpolation of Sparse In-Situ Observational Meteorological Data Using STSR-ResNet"**.

## 🚀 Overview
The STSR-ResNet (SpatioTemporal Separable Reorganization ResNet) is a deep learning model designed to accurately interpolate missing values in sparse meteorological station data by effectively capturing complex spatiotemporal dependencies.

## ✨ Key Features
- Implements the novel **STSR module** for separate yet integrated learning of spatial and temporal features.
- Provides scripts for **data preprocessing, model training, and evaluation**.
- Supports **seven meteorological variables** (e.g., temperature, humidity, wind speed).
- Includes **attention visualization tools** for model interpretability.

## 📋 Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`
- *(Optional)* CUDA-capable GPU for accelerated training.

You can install the dependencies via:
```bash
pip install -r requirements.txt
