import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Define the XOR input and target data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = SGD(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X, y, epochs=10000, verbose=0)

# Make predictions
predictions = model.predict(X)
print(predictions)
