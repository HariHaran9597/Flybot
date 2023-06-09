from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  # Add this line
import json
import data_preprocessing

# Load dataset from intents.json
with open('intents.json') as file:
    dataset = json.load(file)

# Preprocessed data
preprocessed_data = data_preprocessing.preprocess_data(dataset)

# Extract features using Bag-of-Words
corpus = [' '.join(tokens) for tokens, _ in preprocessed_data]
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(corpus)

# Get the vocabulary (unique words in the dataset)
vocabulary = vectorizer.get_feature_names_out()

# Print the features and vocabulary
for i, doc in enumerate(corpus):
    print(f"Query: {doc}")
    print(f"Features: {features[i].toarray().flatten()}")
    print()

print(f"Vocabulary: {vocabulary}")
