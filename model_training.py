from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import data_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import json

# Load the dataset from intents.json
with open('intents.json') as file:
    dataset = json.load(file)

# Data preprocessing
preprocessed_data = data_preprocessing.preprocess_data(dataset)

# Feature extraction using Bag-of-Words (BoW)
corpus = [' '.join(tokens) for tokens, _ in preprocessed_data]
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(corpus)

# Split the dataset into training and testing sets
X = features
y = [tag for _, tag in preprocessed_data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)
