# DarkVision Project

This project aims to train a model on animal pictures so that it can accurately classify animals in **night vision** images. The challenge arises because the night vision dataset is imbalanced (some classes are underrepresented compared to others), making it more difficult for standard models to learn effectively.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Usage](#usage)  

---

## Project Overview

The goal is to build and compare different models (such as pre-trained ResNet18 and fine-tuned ResNet18) for animal classification. We first train on regularly lit animal images, then evaluate and refine our model to perform well on **night vision images**, which are more challenging and imbalanced.

**Key Points:**

- **Multiclass classification** of animal images (e.g., Cat, Dog, Cow, Goat, Hen, etc.).  
- **Night vision scenario**: The dataset has fewer examples, leading to imbalance and making the task more difficult.  
- **Benchmarking** with both a deep learning model (Fine-Tuned ResNet18) and a simpler baseline (Pre-trained ResNet18).  

---

## Dataset

We use a dataset which is a collection of animal pictures in standard lighting conditions with additional category Night Vision. This label is imbalanced, providing fewer examples.

**Night Vision Dataset Link**: [Night Vision Animal Images](https://data.mendeley.com/datasets/fk29shm2kn/2)  

For some experiments, the project relies on CSV files (e.g., `train_dataset.csv`, `test_dataset.csv`, `val_dataset.csv`) that list image paths and corresponding labels.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone 
   cd DarkVision
   ```
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

---

