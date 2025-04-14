# 1. Customer Spending Distribution
# Scenario: You’re analyzing monthly spending data for customers at an online store.
# Question: Using a univariate plot, explore the distribution of monthly spending. What kind of plot (histogram/boxplot/KDE) would you use, and what does it reveal about customer spending behavior (e.g., skewness, outliers)?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau


file_path = "Amazon 2_Raw.xlsx"
df = pd.read_excel(file_path)

print(df)

df["Sales"] = pd.to_numeric(df["Sales"],errors='coerce')
print(df["Sales"])
df["Quantity"] = pd.to_numeric(df["Quantity"],errors='coerce')
mean =  df["Sales"].mean()
print(f"{mean:.2f}")

#Using Histogram 
plt.figure(figsize=(10,8))
plt.hist(df["Sales"],bins = 20, color = "red")
plt.axvline(mean, color = "black", linestyle = "dashed", linewidth = 2)
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Customer Spending Distribution")
plt.show()

#Using BoxPlot
plt.figure(figsize=(10,8))
plt.boxplot(df["Sales"], vert = False, widths = 0.5)
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Customer Spending Distribution")
plt.show()

#Using Correlation
x = df["Sales"]
y = df["Quantity"]
# Calculate Pearson Correlation Coefficient
correlation_coefficient, _ = pearsonr(x, y)
#Spearman Rank Correlation
spearman_corr, _ = spearmanr(x, y)
print(f"Spearman Rank Correlation: {spearman_corr:.2f}")

# Kendall Tau Correlation
kendall_corr, _ = kendalltau(x, y)
print(f"Kendall Tau Correlation: {kendall_corr:.2f}")

# Visualize the data and correlation line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', label=f'Correlation line (r = {correlation_coefficient:.2f})')
plt.xlabel('Sales')
plt.ylabel('Quantity')
plt.title('Correlation Between X and Y')
plt.legend()
plt.show()

# Print correlation coefficient
print(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}")

