import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm

# Set up matplotlib to avoid overlap
plt.rcParams['figure.autolayout'] = True

# Load the Excel data
df = pd.read_excel("6th class result 25.xlsx", skiprows=2)  # Skip header rows
percentages = df['Percentage'].dropna().values  # Extract percentages as a numpy array

# --- Measures of Dispersion ---
mean_value = np.mean(percentages)
variance_value = np.var(percentages, ddof=1)  # Sample variance (ddof=1)
std_dev = np.std(percentages, ddof=1)  # Sample standard deviation
cv_value = (std_dev / mean_value) * 100  # Coefficient of variation (%)

# --- Skewness and Kurtosis ---
skewness_value = skew(percentages)
kurtosis_value = kurtosis(percentages)

# --- Quartiles ---
Q1 = np.percentile(percentages, 25)  # 25th percentile
Q2 = np.percentile(percentages, 50)  # Median (50th percentile)
Q3 = np.percentile(percentages, 75)  # 75th percentile

# --- Z-Scores ---
z_scores = (percentages - mean_value) / std_dev

# --- Print Results ---
print("Measures of Dispersion:")
print(f"Variance: {variance_value:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Coefficient of Variation (CV%): {cv_value:.2f}%")
print(f"Skewness: {skewness_value:.2f}")
print(f"Kurtosis: {kurtosis_value:.2f}")
print("\nQuartiles:")
print(f"Q1 (25th Percentile): {Q1:.2f}")
print(f"Q2 (Median, 50th Percentile): {Q2:.2f}")
print(f"Q3 (75th Percentile): {Q3:.2f}")
print("\nZ-Scores (first 5 students for brevity):")
print(z_scores[:5])

# --- Visualization 1: Histogram with Dispersion, Skewness, and Kurtosis ---
plt.figure(figsize=(10, 6))
plt.hist(percentages, bins=10, alpha=0.6, color='skyblue', edgecolor='black', density=True, label='Data')
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(mean_value + std_dev, color='blue', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev:.2f}')
plt.axvline(mean_value - std_dev, color='blue', linestyle='dashed', linewidth=2)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_value, std_dev)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.text(80, 0.02, f'Variance: {variance_value:.2f}', fontsize=10)
plt.text(80, 0.018, f'Skewness: {skewness_value:.2f}', fontsize=10)
plt.text(80, 0.016, f'Kurtosis: {kurtosis_value:.2f}', fontsize=10)
plt.xlabel('Percentage (%)')
plt.ylabel('Density')
plt.title('Histogram of Student Percentages with Dispersion Measures')
plt.legend()
plt.show()

# --- Visualization 2: Box Plot for Quartiles ---
plt.figure(figsize=(8, 5))
plt.boxplot(percentages, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Box Plot of Student Percentages Showing Quartiles')
plt.xlabel('Percentage (%)')
plt.yticks([])  # Remove y-axis labels as it's a single dataset
plt.text(Q1, 1.1, f'Q1: {Q1:.2f}', ha='center')
plt.text(Q2, 1.1, f'Q2 (Median): {Q2:.2f}', ha='center')
plt.text(Q3, 1.1, f'Q3: {Q3:.2f}', ha='center')
plt.show()

# --- Visualization 3: Z-Scores Plot (for top 10 students) ---
plt.figure(figsize=(12, 6))
top_10_indices = np.arange(10)  # First 10 students for clarity
top_10_z_scores = z_scores[:10]
top_10_names = df['Name of Student'].iloc[:10]
plt.bar(top_10_names, top_10_z_scores, color='salmon')
plt.axhline(0, color='black', linewidth=1)  # Z-score = 0 line
plt.title('Z-Scores of Top 10 Students')
plt.xlabel('Student Name')
plt.ylabel('Z-Score')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(top_10_z_scores):
    plt.text(i, v + 0.05 if v > 0 else v - 0.1, f'{v:.2f}', ha='center')
plt.show()