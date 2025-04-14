# 4. Study Hours vs. Exam Scores
# Scenario: A school collects data on how many hours students studied and their corresponding exam scores.
# Question: Create a scatter plot to visualize the relationship between study hours and scores. Is there a visible correlation? If yes, is it positive or negative?
import matplotlib.pyplot as plt
import random as r
import numpy as np
#Having 100 number of students and study_hours in 1-5 range
study_hours = np.random.normal(5,1,10)
print(study_hours)
#Exam score is random between 0 and 100
scores = np.random.randint(0,100,10)
print(scores)
# Calculate correlation
correlation = np.corrcoef(study_hours, scores)[0, 1]
print(f"Correlation coefficient: {correlation:.2f}")
print(f"corelation {correlation}")
#plotting the scatter plot
plt.scatter(study_hours,scores)
plt.xlabel('Study Hours')
plt.ylabel('Scores')
plt.title('Study Hours vs. Exam Scores')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate more realistic data with some correlation
study_hours = np.clip(np.random.normal(3, 1, 100), 0, 6)  # 100 students, hours clipped 0-6
scores = 50 + 10*study_hours + np.random.normal(0, 10, 100)  # Base score + study effect + noise
scores = np.clip(scores, 0, 100)  # Ensure scores stay 0-100

# Calculate correlation
correlation = np.corrcoef(study_hours, scores)[0, 1]
print(f"Correlation coefficient: {correlation:.2f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(study_hours, scores, alpha=0.6)
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.title('Study Hours vs. Exam Scores (Positive Correlation)')
plt.grid(True)
plt.show()