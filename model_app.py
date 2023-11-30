
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model selection import train_test_split
from sklearn.pipeline import Pipeline
# Load the trained model
model = joblib.load('customer_churn_model.pkl') 

# Streamlit app code
st.title("Your Model Deployment App")

# Add input components (example: file upload)
uploaded_file = st.file_uploader("file")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.dataframe(input_data)
    
    model.fit(input_data)
    # Make predictions using the trained model
    predictions = model.predict(input_data)

    # Display the predictions
    st.write("Predictions:")
    st.write(predictions)
