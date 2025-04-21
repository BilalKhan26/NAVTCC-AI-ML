import time
import numpy as np

# Non-vectorized gradient descent (from earlier)
def univariate_linear_regression_non_vectorized(X, y, alpha=0.01, iterations=100):
    m = len(X)
    theta_0 = 0
    theta_1 = 0

    for it in range(iterations):
        total_error_0 = 0
        total_error_1 = 0

        for i in range(m):
            prediction = theta_0 + theta_1 * X[i]
            error = prediction - y[i]
            total_error_0 += error
            total_error_1 += error * X[i]

        # Update parameters
        theta_0 -= alpha * (total_error_0 / m)
        theta_1 -= alpha * (total_error_1 / m)

    return theta_0, theta_1

# Vectorized gradient descent
def vectorized_linear_regression(X, y, alpha=0.01, iterations=100):
    m = len(X)
    X = np.array(X).reshape(m, 1)
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (column of 1s)
    y = np.array(y).reshape(m, 1)
    theta = np.zeros((2, 1))  # Initialize θ0 and θ1

    for i in range(iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * X_b.T.dot(errors)
        theta -= alpha * gradients

    return theta

# Set up a small dataset
X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Measure time for non-vectorized
start_time = time.time()
univariate_linear_regression_non_vectorized(X, y, alpha=0.01, iterations=1)
non_vectorized_time = time.time() - start_time
print(f"Non-vectorized time for 1 iteration: {non_vectorized_time:.6f} seconds")

# Measure time for vectorized
start_time = time.time()
vectorized_linear_regression(X, y, alpha=0.01, iterations=1)
vectorized_time = time.time() - start_time
print(f"Vectorized time for 1 iteration: {vectorized_time:.6f} seconds")


#--------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def vectorized_linear_regression_with_cost(X, y, alpha=0.01, iterations=100):
    m = len(X)
    X = np.array(X).reshape(m, 1)
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (column of 1s)
    y = np.array(y).reshape(m, 1)
    theta = np.zeros((2, 1))  # Initialize θ0 and θ1

    cost_history = []  # To store the cost at each iteration

    for i in range(iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)

        gradients = (1 / m) * X_b.T.dot(errors)
        theta -= alpha * gradients

    return theta, cost_history

# Set up dataset
X = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Perform gradient descent and get cost history
theta, cost_history = vectorized_linear_regression_with_cost(X, y, alpha=0.01, iterations=100)

# Plot the cost over iterations
plt.plot(range(1, 101), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (J)')
plt.title('Cost Function Over Iterations')
plt.grid(True)
plt.show()
#--------------------------------------------------

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split data into train and test sets (80% train, 20% test)
X_train = [1, 2, 3]
y_train = [2, 4, 6]
X_test = [4]
y_test = [8]

# Train the model
theta, _ = vectorized_linear_regression_with_cost(X_train, y_train, alpha=0.01, iterations=100)

# Predict on test set
X_test_b = np.c_[np.ones((len(X_test), 1)), np.array(X_test).reshape(-1, 1)]  # Add bias term
predictions = X_test_b.dot(theta)

# Evaluate performance
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
#--------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Function to make predictions (from earlier)
def predict(x, theta):
    return theta[0] + theta[1] * x

# Train the model (using the same data)
X_train = [1, 2, 3]
y_train = [2, 4, 6]
theta, _ = vectorized_linear_regression_with_cost(X_train, y_train, alpha=0.01, iterations=100)

# Test data (one point for testing)
X_test = [4]
y_test = [8]

# Predict using the trained model
X_range = np.linspace(0, 5, 100)  # For plotting the line
y_range = predict(X_range, theta)

# Plot training data, test data, and best-fit line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_range, y_range, color='green', label='Best-Fit Line')

# Customize the plot
plt.xlabel('X')
plt.ylabel('y')
plt.title('Best-Fit Line for Univariate Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
#--------------------------------------------------
