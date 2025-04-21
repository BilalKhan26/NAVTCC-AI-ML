# Evaluation.py
from sklearn.metrics import accuracy_score, classification_report
from Training import model, X_test, y_test

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
import joblib

# Save model
joblib.dump(model, 'iris_model2.pkl')