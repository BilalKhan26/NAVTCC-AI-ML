# salary_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load Dataset
# df = pd.read_csv('Salary_Data.csv')
import random

# Simulate random dataset
np.random.seed(42)  # For reproducibility
years_experience = np.round(np.random.uniform(1, 10, 30), 2)  # 30 random values between 1 and 10
salary = np.round(30000 + years_experience * 8000 + np.random.normal(0, 5000, 30), 2)  # Add some noise

df = pd.DataFrame({
    'YearsExperience': years_experience,
    'Salary': salary
})



# 2. EDA
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Scatter Plot
plt.scatter(df['YearsExperience'], df['Salary'], color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()

# 3. Linear Regression using Gradient Descent
X = df['YearsExperience'].values
y = df['Salary'].values
m = len(y)

theta0 = 0
theta1 = 0
alpha = 0.01
iterations = 1000

def compute_cost(theta0, theta1, X, y):
    predictions = theta0 + theta1 * X
    return np.mean((predictions - y) ** 2) / 2

# Gradient Descent Loop
for i in range(iterations):
    predictions = theta0 + theta1 * X
    error = predictions - y
    d_theta0 = (1/m) * np.sum(error)
    d_theta1 = (1/m) * np.sum(error * X)
    
    theta0 -= alpha * d_theta0
    theta1 -= alpha * d_theta1

    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {compute_cost(theta0, theta1, X, y):.2f}")

print(f"\nGradient Descent Result:\nIntercept (θ0): {theta0:.2f}, Slope (θ1): {theta1:.2f}")

# 4. Evaluate with MSE
predictions = theta0 + theta1 * X
mse = mean_squared_error(y, predictions)
print(f"\nMSE: {mse:.2f}")

# 5. Plot Regression Line
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Gradient Descent Regression Line')
plt.show()

# 6. Interpret Coefficients
print(f"\nInterpretation:")
print(f"Base Salary (0 yrs): ${theta0:.2f}")
print(f"Increment per year: ${theta1:.2f}")

# 7. Predict salary for 6.5 years
years = 6.5
predicted_salary = theta0 + theta1 * years
print(f"\nPredicted salary for 6.5 years of experience: ${predicted_salary:.2f}")

# 8. Compare with Scikit-learn
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
print(f"\nScikit-learn Linear Regression:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Prediction for 6.5 years: ${model.predict([[6.5]])[0]:.2f}")
