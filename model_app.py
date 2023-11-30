
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import numpy as np
import joblib

app = Flask(__name__)

# Example XGBoost model and MinMaxScaler
# Replace this with your trained model and scaler
model = joblib.load('customer_churn_model.pkl')
minmax_scaler = MinMaxScaler()

# Endpoint to receive input data and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive input data as JSON
        data = request.get_json()

        # Assuming input features are in a list named 'features'
        features = np.array(data['features']).reshape(1, -1)

        # Scale the features using the MinMaxScaler
        scaled_features = minmax_scaler.transform(features)

        # Make predictions using the XGBoost model
        prediction = xgb_model.predict(scaled_features)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(port=5000)
