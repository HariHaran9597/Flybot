from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import re
import string
import random


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


class Chatbot:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.intent_classifier = LogisticRegression()

    def preprocess_data(self):
        self.tags = []
        self.inputs = []
        self.responses = []

        # Iterate over the dataset and extract inputs and responses
        for intent in self.dataset["intents"]:
            for pattern in intent["patterns"]:
                self.inputs.append(pattern)
                self.responses.append(intent["responses"])
                self.tags.append(intent["tag"])

        # Fit the vectorizer with the inputs
        self.vectorizer.fit(self.inputs)

        # Preprocess the inputs
        self.processed_inputs = [preprocess_text(input_text) for input_text in self.inputs]

        # Convert tags to categorical labels
        self.label_encoder.fit(self.tags)
        self.encoded_tags = self.label_encoder.transform(self.tags)

        # Train the intent classification model
        self.intent_classifier.fit(self.vectorizer.transform(self.processed_inputs), self.encoded_tags)

    def get_response(self, user_input):
        user_input_processed = preprocess_text(user_input)
        user_input_vectorized = self.vectorizer.transform([user_input_processed]).toarray()
        intent_prediction = self.intent_classifier.predict(user_input_vectorized)
        intent_label = self.label_encoder.inverse_transform(intent_prediction)[0]
        response = random.choice(self.responses[self.tags.index(intent_label)])
        return response
