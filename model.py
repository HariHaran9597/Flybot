import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Define the NeuralNet class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Load the raw training data
X_train = ['Hello', 'How are you?', 'Goodbye']
y_train = ['greeting', 'greeting', 'farewell']

# Preprocess the input data
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train).toarray()

# Preprocess the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the training process
def train_model(X_train, y_train):
    input_size = X_train.shape[1]
    hidden_size = 50  # Adjust the number of hidden units as desired
    output_size = len(np.unique(y_train))

    # Convert data to tensors
    inputs = torch.Tensor(X_train)
    labels = torch.LongTensor(y_train)

    # Define the model
    model = NeuralNet(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    num_epochs = 100  # Adjust the number of epochs as desired
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for tracking progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model checkpoint
    checkpoint = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'model_state': model.state_dict()
    }
    torch.save(checkpoint, 'data.pth')

# Update the input size and train the model
train_model(X_train_transformed, y_train_encoded)
