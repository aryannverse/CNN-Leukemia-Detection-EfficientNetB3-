# 🩸 Leukemia Classification using EfficientNetB3
### *Acute Lymphoblastic Leukemia (ALL) Detection System*

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.X-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?style=for-the-badge&logo=keras)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 📊 Project Overview

This project utilizes Deep Learning and Transfer Learning techniques to classify white blood cells from microscopic images. The goal is to accurately distinguish between normal cells and those affected by **Acute Lymphoblastic Leukemia (ALL)** using the C-NMC dataset.

Acute lymphoblastic leukemia (ALL) is the most common type of childhood cancer and accounts for approximately 25% of the pediatric cancers. These cells have been segmented from microscopic images and are representative of images in the real-world because they contain some staining noise and illumination errors, although these errors have largely been fixed in the course of acquisition. The task of identifying immature leukemic blasts from normal cells under the microscope is challenging due to morphological similarity and thus the ground truth labels were annotated by an expert oncologist.

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

--> Download the EfficientNetB3 from this [Google Drive Link](https://drive.google.com/file/d/1MwS0mp9BwDKNzX06uT21iTxRw8qsHtxK/view?usp=sharing).

<img width="1568" height="1580" alt="output" src="https://github.com/user-attachments/assets/43b6b0d9-bff5-4819-b06e-94e1af3fc23a" />

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

<img width="1611" height="712" alt="output2" src="https://github.com/user-attachments/assets/467387dc-dd12-4ec0-a366-80a4fae2be20" />

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
git clone https://github.com/aryannverse/CNN-Leukemia-Detection-EfficientNetB3-.git
```

### 2. Install Dependencies
Install the required libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Download Data
1.  Download the dataset from the [Kaggle Link](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification).
2.  Extract the downloaded folder.
3.  Download the EfficientNetB3 from this [Google Drive Link](https://drive.google.com/file/d/1MwS0mp9BwDKNzX06uT21iTxRw8qsHtxK/view?usp=sharing).
4.  **Important:** Ensure the path in the notebook matches your local data location:
    ```python
    data_dir = 'C-NMC_Leukemia/training_data'
    ```

### 4. Run the Notebook
Launch Jupyter to view and run the training process.

```bash
jupyter notebook Leukemia_Classification.ipynb
```

---

## 🧩 Model Evaluation: Confusion Matrix

The confusion matrix below visualizes the performance of the classification model on the **Test Set** (1600 images).

### Visual Representation

<img width="962" height="978" alt="output3" src="https://github.com/user-attachments/assets/d8cf90e2-2c0c-49af-b11d-8c1b4d5e8044" />

### Numerical Breakdown
Based on the classification report, the model distinguishes between **ALL (Leukemia)** and **HEM (Normal)** cells with high precision.

| | **Predicted: ALL** | **Predicted: HEM** |
| :--- | :---: | :---: |
| **Actual: ALL** | **1068** *(True Positives)* | 23 *(False Negatives)* |
| **Actual: HEM** | 49 *(False Positives)* | **460** *(True Negatives)* |

### Classification Report Summary
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **ALL** | 0.96 | 0.98 | 0.97 |
| **HEM** | 0.95 | 0.90 | 0.93 |
| **Overall Accuracy** | | | **95%** |



