from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Load the dataset
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Preprocess the dataset
corpus = []
labels = []
for intent in intents['intents']:
    for example in intent['examples']:
        corpus.append(example)
        labels.append(intent['tag'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Build the pipeline with the vectorizer and classifier
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', SVC(kernel='linear'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the trained model
import joblib
joblib.dump(pipeline, 'intent_model.joblib')


# Define the classify_intent function
def classify_intent(query):
    # Load the trained model
    model = joblib.load('intent_model.joblib')

    # Preprocess the query
    query_features = vectorizer.transform([query])

    # Classify the intent
    predicted_intent = model.predict(query_features)[0]

    return predicted_intent
