# Data_Collection.py
import pandas as pd

# Load dataset
data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Show initial data
print("Initial data:")
print(data.head())

# Unique species values
print("Unique species:", data['species'].unique())
#Result Nominal Data 