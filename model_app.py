
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import pandas as pd

# Load the data and perform preprocessing steps
df = pd.read_csv('customer_churn.csv')
df = df.drop(['area code', 'state', 'phone number'], axis=1)

# Map binary features
binary_mapping = {'yes': 1, 'no': 0}
for column in ['international plan', 'voice mail plan']:
    df[column] = df[column].map(binary_mapping)

features = df.drop(['churn'], axis=1)
target = df['churn']

# Scaling the features using the MinMaxScaler   
minmax_scaler = MinMaxScaler()
scaled_features = minmax_scaler.fit_transform(features)

def predict(values):
    # Making predictions using the model
    model = joblib.load('customer_churn_model.pkl')
    model.fit(features, target)
    prediction = model.predict(values.reshape(1, -1))
    return prediction

def main():
    st.title("Customer Churn Prediction")
    st.header("Enter your details below to see whether you are likely to churn or not.")
    
    # Input form using Streamlit widgets
    with st.form(key='my_form'):
        for col in features.columns:
            st.text_input(col, key=col)

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        input_values = [st.session_state[col] for col in features.columns]
        scaled_input_values = minmax_scaler.transform(input_values)
        result = predict(scaled_input_values)
        st.write(f"Churn Prediction: {'Churn' if result[0] == True else 'No Churn'}")

if __name__ == "__main__":
    main()
