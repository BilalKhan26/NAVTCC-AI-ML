# Training.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from Data_Wrangling import data
import joblib

# Prepare data
X = data.drop('species', axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# # Save model
# joblib.dump(model, 'iris_model.pkl')

# Make objects available for import
__all__ = ['model', 'X_test', 'y_test']
