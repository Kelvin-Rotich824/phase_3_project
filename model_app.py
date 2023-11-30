
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np
import joblib

def predict(features):
    # Scale the features using the MinMaxScaler
    minmax_scaler = MinMaxScaler()
    scaled_features = minmax_scaler.fit_transform(features)

    # Make predictions using the XGBoost model
    model = joblib.load('customer_churn_model.pkl')
    prediction = model.predict(scaled_features)

    return prediction

def main():
    st.title("XGBoost Model Deployment with Streamlit")

    # Create input sliders for 16 features
    feature_sliders = [st.slider(f"Feature {i+1}", 0.0, 1.0, 0.5) for i in range(17)]

    # Create a button to make predictions
    if st.button("Make Prediction"):
        # Prepare input features as a numpy array
        input_features = np.array(feature_sliders).reshape(1, -1)

        # Get the prediction
        model = joblib.load('customer_churn_model.pkl')
        prediction = model.predict(input_features)

        # Display the prediction
        st.success(f"The prediction is: {prediction[0]}")

if __name__ == "__main__":
    main()
