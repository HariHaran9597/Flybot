import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# Create your training data
patterns = [
    "How are you?",
    "How is the weather today?",
    "What is your name?",
    "What time is it?",
]
tags = [
    "greeting",
    "weather",
    "identity",
    "time",
]

# Perform feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Perform label encoding for the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Convert data to tensors
inputs = torch.Tensor(X_train)
labels = torch.LongTensor(y_train)

# Define the model
input_size = inputs.shape[1]
hidden_size = 50
output_size = len(set(labels))
model = NeuralNet(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 100
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
    'model_state': model.state_dict(),
    'label_encoder': label_encoder
}
torch.save(checkpoint, 'data.pth')
