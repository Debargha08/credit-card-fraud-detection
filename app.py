from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Credit Card Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({'error': 'Missing input features'}), 400

    features = np.array(data['features']).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

