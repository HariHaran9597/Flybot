import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Define and assign values to the corpus variable
corpus = [
    "book_flight",
    "book_ticket",
    "flight_inquiry",
    "ticket_price",
    "book_ticket",
    "flight_inquiry",
    "flight_inquiry",
    "book_flight",
    "flight_duration",
    "flight_inquiry"
]

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the corpus
features = vectorizer.fit_transform(corpus)

# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Define and assign values to y_test and y_pred
y_test = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # True labels
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]  # Predicted labels

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(feature_names))
plt.xticks(tick_marks, feature_names, rotation=45)
plt.yticks(tick_marks, feature_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
