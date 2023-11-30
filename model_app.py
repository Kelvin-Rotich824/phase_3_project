
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

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
    def binary_feature(target_value):
        if target_value == 'yes':
            return 1
        else:
            return 0

# Applying the function to the selected columns.
    for column in input_data[['international plan', 'voice mail plan']]:
        input_data[column] = input_data[column].apply(binary_feature)

    y = input_data['churn']
    X = input_data.drop(['churn', 'state', 'area code', 'phone number'], axis=1)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model.fit(X, y)
    # Make predictions using the trained model
    predictions = model.predict(X)

    # Display the predictions
    st.write("Predictions:")
    st.write(predictions)
