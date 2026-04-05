# Vehicle Damage Classification Using Convolutional Neural Networks

A deep learning pipeline for classifying six categories of vehicle damage from photographs, built for insurance claim verification.

## Overview

This project implements a complete CNN-based image classification system that can distinguish between crack, scratch, tire flat, dent, glass shatter, and broken damage from vehicle photographs. The work follows a structured experimental approach: baseline model, regularisation experiments, hyperparameter tuning, overfitting analysis, and final evaluation.

## What This Project Covers

- Data exploration and class distribution analysis for six damage types
- Baseline CNN architecture with 3x3 convolutional blocks, batch normalisation, and max-pooling
- Regularisation through dropout, L2 weight decay, and on-the-fly data augmentation
- Hyperparameter tuning across learning rate, batch size, and base filter count
- Overfitting analysis comparing training and validation dynamics
- Class imbalance handling using computed class weights
- Per-class evaluation with confusion matrix and classification report

## Technical Stack

- Python, TensorFlow/Keras, scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn, PIL
- Keras ImageDataGenerator for augmentation and disk streaming
- EarlyStopping with best-weight restoration

## Key Design Decisions

- **Class weights over resampling** to avoid duplication artifacts in image data
- **Simple baseline first** to establish a clear reference before adding complexity
- **224x224 input resolution** balancing spatial detail with computational cost
- **Stratified splits** preserving class proportions in train/validation sets

## Project Structure

```
Vehicle_Damage_Classification_CNN.ipynb   # Complete pipeline notebook
README.md                                  # This file
```

## Case Study

A detailed reflective case study for this project is available on my [portfolio site](https://chimezie-ai-portfolio.netlify.app/case-study/vehicle-damage-cnn).

## Author

Chimezie Onuchukwu
- [Portfolio](https://chimezie-ai-portfolio.netlify.app)
- [LinkedIn](https://www.linkedin.com/in/onuchukwu-joseph-589912148)
