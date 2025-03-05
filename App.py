import streamlit as st
import pandas as pd
import pickle
import numpy as np

def load_model():
    with open("/mnt/data/logistic_regression_titanic.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to make predictions
def predict_survival(features):
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)[:, 1]
    return prediction[0], probability[0]

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# Sidebar for user input
st.sidebar.header("User Input")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 100, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 30.0)

# Convert categorical values
sex = 1 if sex == "male" else 0

features = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
}

if st.sidebar.button("Predict Survival"):
    prediction, probability = predict_survival(features)
    
    st.subheader("Prediction Result")
    st.write(f"Survival Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
    st.write(f"Survival Probability: {probability:.2%}")

st.markdown("---")
st.write("Built with Streamlit | Logistic Regression Model")
