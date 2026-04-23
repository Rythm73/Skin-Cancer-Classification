# Skin Cancer Classification using Deep Learning

A multi-class dermoscopic image classification project on the 
HAM10000 dataset, comparing a custom CNN baseline against two 
transfer learning architectures: ResNet-50 and EfficientNet-B3.

## Project Overview
Skin cancer is one of the most common and potentially fatal diseases 
worldwide. This project builds an automated classification system for 
seven types of skin lesions using deep learning, addressing real-world 
challenges including severe class imbalance and limited minority class 
representation.

## Dataset
- **Source:** [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Size:** 10,015 dermoscopic images
- **Classes:** 7 (nv, mel, bkl, bcc, akiec, vasc, df)
- **Challenge:** Heavy class imbalance — nv accounts for 66.95% of samples

## Methodology
- Lesion-level train/val/test split (70/15/15) to prevent data leakage
- Minority class oversampling to 500 images per class
- Weighted cross-entropy loss (up to 13.3× penalty for rarest class)
- Data augmentation on training set only (flips, rotation, color jitter)
- ImageNet normalization for transfer learning compatibility

## Models
| Model | Architecture | Trainable Params |
|---|---|---|
| Custom CNN | 3-block CNN from scratch | All |
| ResNet-50 | ImageNet pretrained, layer4 unfrozen | 15.5M (64.5%) |
| EfficientNet-B3 | ImageNet pretrained, features[-3:] unfrozen | 8.9M (80.3%) |


EfficientNet-B3 is the recommended model for clinical use due to its 
superior Macro F1 and Melanoma recall (0.66 vs 0.51 for ResNet-50).

## Contents
- `skin_cancer_classification.ipynb` — full pipeline including data 
preprocessing, model training, evaluation, and visualization
- `skin_cancer_paper.pdf` — IEEE conference paper with full methodology 
and analysis

## Dependencies
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn Pillow
```

## Environment
- Python 3.10
- PyTorch 2.x
- Google Colab with NVIDIA Tesla T4 GPU

