import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch
from model import NeuralNet
from sklearn.svm import SVC
from scipy.special import softmax

# Load the trained FNN model
data = torch.load('data.pth')
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data['model_state'])
model.eval()

# Get the tags from the loaded data
tags = data['tags']

# Load X_train and y_train for evaluation
X_train = np.load('X_train.npy')  # Replace 'X_train.npy' with the actual path to your data file
y_train = np.load('y_train.npy')  # Replace 'y_train.npy' with the actual path to your data file

# Evaluate the FNN model
with torch.no_grad():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.Tensor(X_train).to(device)
    outputs = model(inputs)
    probabilities = softmax(outputs.cpu().numpy(), axis=1)
    fnn_y_train = np.argmax(probabilities, axis=1)

# Load the trained SVM model
svm_model = SVC()
svm_model = svm_model.load('svm_model.pth')

# Evaluate the SVM model
svm_y_train = svm_model.predict(X_train)

# Print confusion matrix and classification report for FNN
fnn_cm = confusion_matrix(y_train, fnn_y_train)
fnn_cr = classification_report(y_train, fnn_y_train, target_names=tags)
print("FNN Confusion Matrix:")
print(fnn_cm)
print("FNN Classification Report:")
print(fnn_cr)

# Print confusion matrix and classification report for SVM
svm_cm = confusion_matrix(y_train, svm_y_train)
svm_cr = classification_report(y_train, svm_y_train, target_names=tags)
print("SVM Confusion Matrix:")
print(svm_cm)
print("SVM Classification Report:")
print(svm_cr)

# Plot confusion matrices
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# FNN Confusion Matrix
ax[0].matshow(fnn_cm, cmap=plt.cm.Blues)
ax[0].set_title("FNN Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("True")
for i in range(len(tags)):
    for j in range(len(tags)):
        ax[0].text(j, i, str(fnn_cm[i, j]), ha="center", va="center")

# SVM Confusion Matrix
ax[1].matshow(svm_cm, cmap=plt.cm.Blues)
ax[1].set_title("SVM Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("True")
for i in range(len(tags)):
    for j in range(len(tags)):
        ax[1].text(j, i, str(svm_cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

# Generate ROC curves and compute AUC-ROC for FNN
fpr_fnn = dict()
tpr_fnn = dict()
roc_auc_fnn = dict()
for i in range(len(tags)):
    fpr_fnn[i], tpr_fnn[i], _ = roc_curve(y_train, probabilities[:, i], pos_label=i)
    roc_auc_fnn[i] = auc(fpr_fnn[i], tpr_fnn[i])

# Generate ROC curve and compute AUC-ROC for SVM
fpr_svm, tpr_svm, _ = roc_curve(y_train, svm_y_train, pos_label=1)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(tags)):
    plt.plot(fpr_fnn[i], tpr_fnn[i], label=f'FNN {tags[i]} (AUC = {roc_auc_fnn[i]:.2f})')

plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
