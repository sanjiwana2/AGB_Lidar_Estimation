# 🌲 AGB Estimation using Shallow and Deep Learning

This repository implements the study:

**"Comparison of Shallow and Deep Learning Algorithms for Aboveground Biomass Estimation Using LiDAR: Case Study of Eucalyptus Forest in Ibu Kota Nusantara (IKN), East Kalimantan, Indonesia"**

It provides a unified framework for training and comparing shallow learning models and deep learning architectures for aboveground biomass (AGB) estimation using LiDAR-derived features.

---

## 🚀 Features

- Comparison of shallow learning vs deep learning models
- 1D and 2D CNN architectures for LiDAR feature learning
- Attention-enhanced CNN models (Squeeze-and-Excitation)
- KAN-based neural network architectures
- Residual CNN-KAN hybrid models
- Modular model registry for easy experimentation
- Supports regression-based AGB prediction

---

## 🧠 Model Architectures

### CNN Models
- CNN1D, CNN2D
- CNN1D_MLP, CNN2D_MLP
- CNN1D_SE, CNN2D_SE
- CNN1D_SE_MLP, CNN2D_SE_MLP

### KAN Models
- CNN1D_KAN
- CNN2D_KAN
- CNN1D_KAN_SE
- CNN2D_KAN_SE

---

## 📦 Model Import Example

```python
from Model import cnn_model, cnn_model2
from Model.cnn_model_kan import CNN1D_KAN, CNN2D_KAN, CNN1D_KAN_SE, CNN2D_KAN_SE
