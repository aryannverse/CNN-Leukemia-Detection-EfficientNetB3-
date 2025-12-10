import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax

def plot_training(history):

    tr_acc = history['accuracy']
    tr_loss = history['loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    plt.style.use('fivethirtyeight')

    ax1.plot(Epochs, tr_loss, 'r', label='Training loss')
    ax1.plot(Epochs, val_loss, 'g', label='Validation loss')
    ax1.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    ax2.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    ax2.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write('Normalized Confusion Matrix')
    else:
        st.write('Confusion Matrix, Without Normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return fig, cm


sim_history = {
    'loss': [5.135, 2.546, 1.479, 0.888, 0.560, 0.392, 0.296, 0.237, 0.185, 0.164, 0.145, 0.132, 0.118, 0.118, 0.107,
             0.099, 0.104, 0.092, 0.090],
    'accuracy': [0.82485, 0.88756, 0.91088, 0.92428, 0.93648, 0.94184, 0.94666, 0.95417, 0.97293, 0.97856, 0.98258,
                 0.98459, 0.99049, 0.98847, 0.99317, 0.99558, 0.99223, 0.99571, 0.99611],
    'val_loss': [3.63725, 2.38163, 1.14977, 0.86442, 0.54691, 0.43709, 0.29600, 0.30395, 0.25357, 0.22330, 0.22930,
                 0.22191, 0.20185, 0.19495, 0.19349, 0.18045, 0.18132, 0.18307, 0.18058],
    'val_accuracy': [0.70169, 0.73671, 0.88993, 0.86116, 0.88430, 0.87805, 0.93183, 0.91307, 0.93684, 0.95122, 0.94371,
                     0.94809, 0.95810, 0.95872, 0.95935, 0.96185, 0.96248, 0.95872, 0.96060]
}


sim_scores = {
    'Train Loss': 0.0843,
    'Train Accuracy': 0.9962,
    'Validation Loss': 0.1711,
    'Validation Accuracy': 0.9638,
    'Test Loss': 0.2188,
    'Test Accuracy': 0.9550
}


sim_classes = ['all', 'hem']
sim_cm = np.array([[1068, 23], [49, 460]])
sim_clf_report = """
              precision    recall  f1-score   support

         all       0.96      0.98      0.97      1091
         hem       0.95      0.90      0.93       509

    accuracy                           0.95      1600
   macro avg       0.95      0.94      0.95      1600
weighted avg       0.95      0.95      0.95      1600
"""


sim_model_summary = """
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 efficientnetb3 (Functional  (None, 1536)              10783535  
 )                                                               

 batch_normalization (Batch  (None, 1536)              6144      
 Normalization)                                                  

 dense (Dense)               (None, 256)               393472    

 dropout (Dropout)           (None, 256)               0         

 dense_1 (Dense)             (None, 2)                 514       

=================================================================
Total params: 11183665 (42.66 MB)
Trainable params: 11093290 (42.32 MB)
Non-trainable params: 90375 (353.03 KB)
_________________________________________________________________
"""



st.title("🔬 Leukemia Classification Model Analysis")

st.markdown("""
This application displays the results from the `Leukemia_Classification.ipynb` Jupyter Notebook,
which trains an **EfficientNetB3** model for leukemia image classification.
""")


st.header("1. Model Architecture (Sequential Model)")
st.code(sim_model_summary, language='text')


st.header("2. Training History")
st.subheader("Loss and Accuracy Across Epochs")
st.pyplot(plot_training(sim_history))


st.header("3. Model Performance Scores")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Training Scores")
    st.metric("Loss", f"{sim_scores['Train Loss']:.4f}")
    st.metric("Accuracy", f"{sim_scores['Train Accuracy']:.4f}")

with col2:
    st.subheader("Validation Scores")
    st.metric("Loss", f"{sim_scores['Validation Loss']:.4f}")
    st.metric("Accuracy", f"{sim_scores['Validation Accuracy']:.4f}")

with col3:
    st.subheader("Test Scores")
    st.metric("Loss", f"{sim_scores['Test Loss']:.4f}")
    st.metric("Accuracy", f"{sim_scores['Test Accuracy']:.4f}")

st.header("4. Test Set Results")

st.subheader("4.1. Confusion Matrix")
fig_cm, cm_data = plot_confusion_matrix(sim_cm, sim_classes, title='Confusion Matrix for Test Data')
st.pyplot(fig_cm)
st.markdown("The numbers in the matrix are:")
st.dataframe(pd.DataFrame(cm_data, index=sim_classes, columns=sim_classes))

st.info("The Confusion Matrix shows: "
        f"**{cm_data[0, 0]}** 'all' correctly classified, "
        f"**{cm_data[1, 1]}** 'hem' correctly classified, "
        f"**{cm_data[0, 1]}** 'all' incorrectly classified as 'hem', and "
        f"**{cm_data[1, 0]}** 'hem' incorrectly classified as 'all'.")

st.subheader("4.2. Classification Report")
st.code(sim_clf_report, language='text')

st.markdown(
    f"***Note:*** *The final model achieved a test accuracy of approximately **{sim_scores['Test Accuracy']:.2%}**.*")

st.markdown("---")
st.caption("Data source: Leukemia Classification Notebook.")