import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(2, size=100)

# Initialize weights and bias
np.random.seed(42)
weights = np.random.rand(2)
bias = np.random.rand()

# Hyperparameters
learning_rate = 0.1
num_epochs = 1000

# Training using gradient descent
for epoch in range(num_epochs):
    z = np.dot(X, weights) + bias
    predictions = 1 / (1 + np.exp(-z))
    
    dz = predictions - y
    dw = np.dot(X.T, dz) / 100
    db = np.sum(dz) / 100
    
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    if epoch % 100 == 0:
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the trained model
test_samples = np.array([[0.1, 0.2], [0.8, 0.9]])
test_predictions = 1 / (1 + np.exp(-np.dot(test_samples, weights) - bias))
print("Test Predictions:", test_predictions)
