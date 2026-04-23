# Skin-Cancer-Classification
# Skin Cancer Classification using Deep Learning

## Overview
Multi-class skin lesion classification on the HAM10000 dataset 
comparing Custom CNN, ResNet-50, and EfficientNet-B3.

## Contents
- `skin_cancer_classification.ipynb` — full training and evaluation code
- `skin_cancer_paper.pdf` — IEEE conference paper
- Figures — confusion matrices and training curves

## Results
| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Custom CNN | 42.61% | 0.2215 | 0.8458 |
| ResNet-50 | 76.12% | 0.5862 | 0.9332 |
| EfficientNet-B3 | 75.92% | 0.6709 | 0.9509 |

## Dataset
HAM10000 from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Dependencies
torch, torchvision, scikit-learn, pandas, numpy, matplotlib, seaborn
