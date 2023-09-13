import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Features (two variables)
y = np.array([3, 4.8, 6.6, 8.5, 10.3])  # Target variable

# Create and fit the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)
