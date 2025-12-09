# 🩸 Leukemia Classification using EfficientNetB3
### *Acute Lymphoblastic Leukemia (ALL) Detection System*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.X-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?style=for-the-badge&logo=keras)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 📊 Project Overview

This project utilizes Deep Learning and Transfer Learning techniques to classify white blood cells from microscopic images. The goal is to accurately distinguish between normal cells and those affected by **Acute Lymphoblastic Leukemia (ALL)** using the C-NMC dataset.

**Performance Metric:**
> **95.50% Test Accuracy** achieved using custom optimization strategies.

---

## 🧠 The Architecture: EfficientNetB3

We employ **EfficientNetB3**, a state-of-the-art convolutional neural network that balances depth, width, and resolution using a compound coefficient.

### 🏗️ Model Structure
| Stage | Component | Description |
| :--- | :--- | :--- |
| **1. Base** | **EfficientNetB3** | Pre-trained on ImageNet. Used for feature extraction (Top layers removed). |
| **2. Pooling** | **Global Max Pooling** | Reduces spatial dimensions while retaining the most prominent features. |
| **3. Norm** | **Batch Normalization** | Stabilizes learning by normalizing inputs ($axis=-1$). |
| **4. Dense** | **Fully Connected (256)** | Custom layer with L1/L2 regularization and ReLU activation. |
| **5. Dropout** | **Dropout (0.45)** | Randomly sets 45% of neurons to 0 to prevent overfitting. |
| **6. Output** | **Dense (2)** | Softmax layer for binary classification probabilities. |

---

## 🧪 Techniques & Mathematics

### 1. Compound Scaling (EfficientNet)
Unlike traditional scaling (just adding layers), EfficientNet scales three dimensions uniformly:
$$depth: \alpha^\phi, \quad width: \beta^\phi, \quad resolution: \gamma^\phi$$
Subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

### 2. Regularization (Elastic Net)
To combat overfitting in medical imaging, we apply both L1 and L2 regularization to the dense layer weights and bias.
*   **L1 (Lasso):** Promotes sparsity.
    $$Loss_{L1} = \lambda_1 \sum |w_i|$$
*   **L2 (Ridge):** Prevents large weights.
    $$Loss_{L2} = \lambda_2 \sum w_i^2$$

### 3. Optimization (Adamax)
We utilize **Adamax**, a variant of Adam based on the infinity norm ($\ell_\infty$). It provides stable parameter updates, often beneficial in models with embeddings or sparse updates.
$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{u_t} \cdot m_t$$

### 4. Custom Callback Strategy
A custom callback loop monitors training to implement:
1.  **Learning Rate Scheduling:** Decays LR by factor $0.5$ if validation accuracy plateaus.
2.  **Early Stopping:** Halts training after specific patience thresholds to save resources.
3.  **Model Checkpointing:** Saves the weights of the epoch with the highest validation accuracy.

---

## 📂 The Dataset

The project uses the **C-NMC (Classification of Normal vs Malignant Cells)** Leukemia dataset.

*   **Source:** [Kaggle - Leukemia Classification](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
*   **Classes:** 
    1.  `Hem` (Normal)
    2.  `All` (Malignant / Leukemia)
*   **Split:**
    *   Training: 70%
    *   Validation: 15%
    *   Test: 15%

---

## 🚀 Installation & Cloning

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
Open your terminal or command prompt and run:

```bash
# Clone the project
git clone <YOUR_REPOSITORY_URL_HERE>

# Navigate into the directory
cd leukemia-classification
