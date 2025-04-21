# Data_Wrangling.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Data_Collection as dw

# Copy the data
data = dw.data.copy()

# Encode categorical target variable
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
print("Encoded species:", data['species'].values)

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Feature engineering: petal ratio
data['petal_ratio'] = data['petal_length'] / data['petal_width']
