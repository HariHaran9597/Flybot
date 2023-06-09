import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

#nltk.download('punkt')
#nltk.download('stopwords')

def preprocess_data(dataset):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    preprocessed_data = []

    for intent in dataset['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            # Tokenization
            tokens = word_tokenize(pattern)

            # Text normalization
            tokens = [token.lower() for token in tokens if token not in string.punctuation]

            # Stop word removal
            tokens = [token for token in tokens if token not in stop_words]

            # Stemming
            tokens = [stemmer.stem(token) for token in tokens]

            preprocessed_data.append((tokens, tag))

    return preprocessed_data

