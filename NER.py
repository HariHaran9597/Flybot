import json
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize


class Flybot:
    def __init__(self):
        self.intents = []
        self.responses = []
        self.labels = []
        self.training_data = []
        self.intent_classifier = None
        self.vectorizer = None

    def load_dataset(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        for intent in data['intents']:
            self.intents.append(intent['tag'])
            self.responses.extend(intent['responses'])
            for pattern in intent['patterns']:
                self.training_data.append((pattern.lower(), intent['tag']))
        self.labels = list(set(self.intents))

    def preprocess_text(self, text):
        # Tokenize the text into individual words
        tokens = word_tokenize(text.lower())
        # Remove punctuation and special characters
        tokens = [token for token in tokens if re.match(r'\w+', token)]
        # Join the tokens back into a string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def train_intent_classifier(self):
        # Preprocess the training data
        preprocessed_training_data = [self.preprocess_text(text) for text, label in self.training_data]
        # Create a CountVectorizer to convert text into numerical features
        self.vectorizer = CountVectorizer()
        features = self.vectorizer.fit_transform(preprocessed_training_data)
        # Train a classifier using cosine similarity as the distance metric
        self.intent_classifier = cosine_similarity(features)

    def recognize_intent(self, query):
        preprocessed_query = self.preprocess_text(query)
        query_features = self.vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_features, self.intent_classifier)[0]
        # Get the index of the intent with the highest similarity score
        intent_index = similarities.argmax()
        intent = self.labels[intent_index]
        return intent

    def extract_entities(self, query):
        entities = []
        for intent in self.intents:
            patterns = [pattern.lower() for pattern, label in self.training_data if label == intent]
            for pattern in patterns:
                matches = re.findall(r'\[(.*?)\]', pattern)
                for match in matches:
                    if match in query:
                        entities.append({'entity': match.upper(), 'value': query})
        return entities

    def generate_response(self, intent):
        response = random.choice([resp for resp in self.responses if intent in resp])
        return response

    def process_query(self, query):
        intent = self.recognize_intent(query)
        entities = self.extract_entities(query)
        response = self.generate_response(intent)
        return intent, entities, response


# Instantiate the Flybot and load the dataset
flybot = Flybot()
flybot.load_dataset('intents.json')

# Train the intent classifier
flybot.train_intent_classifier()

# Test the Flybot with some queries
queries = [
    "Hello",
    "What are the ticket prices?",
    "I would like to book a flight",
    "Check flight status",
    "How can I cancel my flight?",
    "How can I pay for my ticket?",
    "Thank you",
    "I want to travel to London",
    "I'll be traveling on July 15th",
    "Confirm my booking",
    "What can you do?"
]

for query in queries:
    intent, entities, response = flybot.process_query(query)
    print("Query:", query)
    print("Intent:", intent)
    print("Entities:", entities)
    print("Response:", response)
    print()
