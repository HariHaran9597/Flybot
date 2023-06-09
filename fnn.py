import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# Load the intents.json file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Prepare the training data
queries = []  # List of training queries
labels = []  # List of corresponding intent labels

# Iterate over the intents and extract the queries and labels
for intent in intents['intents']:
    for pattern in intent['patterns']:
        queries.append(pattern)
        labels.append(intent['tag'])

# Tokenize the queries
tokenizer = Tokenizer()
tokenizer.fit_on_texts(queries)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
sequences = tokenizer.texts_to_sequences(queries)
max_sequence_length = max(len(seq) for seq in sequences)

# Pad the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert the labels to numerical labels
label_indices = {label: index for index, label in enumerate(set(labels))}
numerical_labels = [label_indices[label] for label in labels]

# Convert the training data to numpy arrays
X_train = np.array(padded_sequences)
y_train = np.array(numerical_labels)

# Define the FNN model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sequence_length))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the training data
y_pred = model.predict(X_train)

# Print the predicted intents for each query
for i in range(len(X_train)):
    query = queries[i]
    intent = np.argmax(y_pred[i])
    predicted_intent = list(label_indices.keys())[list(label_indices.values()).index(intent)]
    print(f"Query: {query} | Predicted Intent: {predicted_intent}")
