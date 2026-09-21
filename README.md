# Iris Flower Classification using K-Nearest Neighbors (KNN)

This repository contains an implementation of the **K-Nearest Neighbors (KNN)** classification algorithm on the classic **Iris dataset** using `scikit-learn`.

---

## 📌 Overview

**K-Nearest Neighbors (KNN)** is an instance-based, non-parametric, and **lazy learning algorithm**. Instead of explicitly learning a model during training, it memorizes the dataset and performs computation at inference time.

### Algorithm Workflow
1. **Memorize**: Store training data points during training (`fit`).
2. **Calculate Distance**: Compute distance metrics (e.g., Euclidean distance) between query data points and all stored training points.
3. **Identify Neighbors**: Select the top $k$ nearest data points based on proximity.
4. **Majority Vote**: Assign the class label with the majority vote among the $k$ neighbors.

> **Tip:** Choose an **odd value for $k$** (e.g., $k=3$) to avoid tie votes in multi-class classification.

---

## 🛠️ Tech Stack & Dependencies

- **Python** (>= 3.8)
- **pandas** — Data handling
- **matplotlib** — Data visualization
- **scikit-learn** — Dataset loading, train/test split, KNN algorithm, and model evaluation

---

## 📊 Dataset Details

The **Iris dataset** consists of 150 samples across 3 species classes:
- **Classes**: `setosa` (0), `versicolor` (1), `virginica` (2) — 50 samples per class
- **Features**:
  - Sepal Length ($\text{cm}$)
  - Sepal Width ($\text{cm}$)
  - Petal Length ($\text{cm}$)
  - Petal Width ($\text{cm}$)

---

## 🚀 Getting Started

### Installation

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
