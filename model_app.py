
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np
import joblib

def predict(features):
    # Create a mapping function and apply it to the selected columns
    def binary_feature(target_value):
        if target_value == 'yes':
            return 1
        else:
            return 0

# Applying the function to the selected columns.
    for column in data[['international plan', 'voice mail plan']]:
        data[column] = data[column].apply(binary_feature)
        print(f'\n{data[column].value_counts()}')
        # Scaling the features using the MinMaxScaler
        minmax_scaler = MinMaxScaler()
        scaled_features = minmax_scaler.fit_transform(features)

        # Making predictions using the model
        model = joblib.load('customer_churn_model.pkl')
        y = [True, False]
        for item in y:
            model.fit(scaled_features, item)
        prediction = model.predict(scaled_features)

        return prediction

def main():
    st.title("Model Deployment with Streamlit")

    # Creating input sliders for 17 features
    feature_sliders = [st.slider(f"Feature {i+1}", 0.0, 1.0, 0.5) for i in range(18)]

    # Creating a button to make predictions
    if st.button("Make Prediction"):
        # Preparing input features as a numpy array
        input_features = np.array(feature_sliders).reshape(1, -1)

        # Getting the prediction
        model = joblib.load('customer_churn_model.pkl')
        y = [True, False]
        for item in y:
            model.fit(input_features, item)
        prediction = model.predict(input_features)

        # Displaying the prediction
        st.success(f"The prediction is: {prediction[0]}")

if __name__ == "__main__":
    main()
