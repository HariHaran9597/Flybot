from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from data_preprocessing import preprocess_data

def evaluate_model(preprocessed_data):
    # Extract features using Bag-of-Words
    vectorizer = CountVectorizer()
    corpus = [' '.join(tokens) for tokens, _ in preprocessed_data]
    features = vectorizer.fit_transform(corpus)

    # Split the dataset into training and testing sets
    X = features
    y = [tag for _, tag in preprocessed_data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Make predictions on the training set
    train_predictions = svm_model.predict(X_train)

    # Make predictions on the testing set
    test_predictions = svm_model.predict(X_test)

    # Calculate evaluation metrics
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions, average='weighted')
    train_recall = recall_score(y_train, train_predictions, average='weighted')
    train_f1 = f1_score(y_train, train_predictions, average='weighted')

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions, average='weighted')
    test_recall = recall_score(y_test, test_predictions, average='weighted')
    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    # Print the evaluation metrics
    print("Training Set Metrics:")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1-Score: {train_f1:.4f}")
    print()
    print("Testing Set Metrics:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

# Load the dataset from intents.json
with open('intents.json') as file:
    dataset = json.load(file)

# Preprocess the data
preprocessed_data = preprocess_data(dataset)

# Evaluate the model using the preprocessed data
evaluate_model(preprocessed_data)
