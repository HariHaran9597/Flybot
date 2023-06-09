from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib


def classify_intent(query):
    # Preprocess the query
    tokens = word_tokenize(query.lower())
    tokens = [token for token in tokens if token not in string.punctuation]

    # Convert query to feature vector
    query_features = vectorizer.transform([' '.join(tokens)])

    # Predict the intent
    predicted_intent = svm_model.predict(query_features)[0]

    return predicted_intent


def get_response(intent):
    # Define intent-specific responses
    responses = {
        'greeting': 'Hello! How can I assist you?',
        'fare_inquiry': 'The ticket prices vary based on the destination and travel dates. Could you please provide more details?',
        'flight_booking': 'Sure! I can help you with flight bookings. Please provide the necessary details.',
        'flight_status': 'To check the flight status, I need the flight details. Could you please provide the flight information?',
        'cancellation': 'For cancellation, please refer to our cancellation policy or provide your booking details for assistance.',
        'payment': 'We offer various payment options, including credit cards, online banking, and mobile wallets. Which method would you prefer?',
        'thanks': 'You\'re welcome! If you have any more questions, feel free to ask.'
    }

    # Get the response based on the predicted intent
    return responses.get(intent, 'I\'m sorry, but I couldn\'t understand your request.')


# Load the pre-trained SVM model
svm_model = joblib.load('svm_model.pkl')

# Create and save the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_data['sentences'])
joblib.dump(vectorizer, 'vectorizer.pkl')

# Example usage
user_query = "Hello, I want to book a flight from New York to London."

# Classify the intent
predicted_intent = classify_intent(user_query)

# Get the appropriate response
response = get_response(predicted_intent)

# Print the response
print(response)
