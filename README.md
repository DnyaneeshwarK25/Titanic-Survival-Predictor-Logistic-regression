# Titanic Survival Prediction - Logistic Regression

This project builds a Logistic Regression model to predict Titanic passenger survival. The model is trained on the Titanic dataset and deployed using Streamlit.

## Features
- Data preprocessing (handling missing values, encoding categorical features, outlier detection & handling).
- Logistic Regression model for classification.
- Performance evaluation (accuracy, precision, recall, F1-score, ROC-AUC).
- Streamlit web app for user-friendly prediction.
- Model saved using Pickle for easy loading and deployment.

## Dataset
Uses the Titanic dataset:
- `Titanic_train.csv`: Training data.
- `Titanic_test.csv`: Testing data.

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/Titanic-Logistic-Regression.git
cd Titanic-Logistic-Regression
```

### **2. Install Dependencies**
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn scipy
```

### **3. Run the Model Training Script**
```bash
python train_model.py
```
This will train the Logistic Regression model and save it as `logistic_regression_titanic.pkl`.

## Running the Web App
To launch the Streamlit app, run:
```bash
streamlit run app.py
```
This will open the Titanic Survival Prediction app in your browser.

## Deployment on Streamlit Community Cloud
1. Upload your project to **GitHub**.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and select your repository.
4. Set `app.py` as the main file.
5. Click **Deploy**.

## Usage
- Select passenger details (class, age, gender, fare, etc.).
- Click **Predict Survival**.
- Get survival probability output.

## License
This project is open-source and free to use.

