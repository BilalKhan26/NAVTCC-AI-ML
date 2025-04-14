# 2. Exam Scores Analysis
# Scenario: A university professor collected the final exam scores of 100 students.
# Question: Visualize the score distribution using an appropriate univariate plot. Identify if the data is normally distributed and check for any unusually low or high scores.

import matplotlib.pyplot as plt
import numpy as np

def generate_scores(n):
    return np.random.normal(70, 15, n)
scores = generate_scores(100)
plt.hist(scores, bins=20,edgecolor='black',alpha=0.7)
plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(np.median(scores), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(np.std(scores), color='blue', linestyle='dashed', linewidth=2, label='Standard Deviation')
plt.title("Exam Scores Distribution")
plt.xlabel("Score")
plt.ylabel("Freqency")
plt.tight_layout()
plt.show()

# The scores are normally distributed with a mean of 70 and a standard deviation of 15. The data is not unusually low or high.