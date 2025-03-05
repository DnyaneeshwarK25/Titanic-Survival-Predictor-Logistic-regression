import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import sklearn  # Ensure scikit-learn is available

# Function to load the trained model
@st.cache_data
def load_model():
    try:
        with open("logistic_regression_titanic.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'logistic_regression_titanic.pkl' is in the same directory.")
        return None

# Function to make predictions
def predict_survival(features):
    model = load_model()
    if model is None:
        return None, None
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])

    # Check if the model expects a different number of features
    expected_features = model.n_features_in_
    actual_features = features_df.shape[1]

    if actual_features != expected_features:
        st.error(f"Feature mismatch: Expected {expected_features}, but received {actual_features}.")
        st.write("Received Features:", list(features.keys()))
        return None, None

    # Predict
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)[:, 1]
    
    return prediction[0], probability[0]

# Streamlit App Layout
st.title("Titanic Survival Prediction")

st.sidebar.header("Passenger Information")

# Collecting user input
features = {
    "Pclass": st.sidebar.selectbox("Passenger Class", [1, 2, 3]),
    "Sex": st.sidebar.selectbox("Sex", ["male", "female"]),
    "Age": st.sidebar.slider("Age", 1, 100, 25),
    "SibSp": st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0),
    "Parch": st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0),
    "Fare": st.sidebar.number_input("Fare Price", 0.0, 500.0, 50.0),
    "Embarked": st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
}

if st.sidebar.button("Predict Survival"):
    prediction, probability = predict_survival(features)

    if prediction is None:
        st.error("Prediction cannot be made due to feature mismatch or missing model.")
    else:
        outcome = "Survived" if prediction == 1 else "Did Not Survive"
        st.subheader(f"Prediction: {outcome}")
        st.write(f"Survival Probability: {probability:.2%}")
