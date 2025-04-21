import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('iris_model2.pkl')

# UI Header
st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower measurements to predict its species.")

# Input sliders
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Calculate petal_ratio
petal_ratio = petal_length / petal_width

# Predict button
if st.button("Predict"):
    # Create the input data including the petal_ratio feature
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width, petal_ratio]],
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'petal_ratio'])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Map prediction to species
    species = ['setosa', 'versicolor', 'virginica'][prediction]
    
    # Display the result
    st.success(f"The predicted species is **{species}** 🌼")
