# Credit Card Fraud Detection System 🚨

A real-time credit card fraud detection system using a trained XGBoost model served through a Flask API.

## 🔍 Overview

This project detects fraudulent credit card transactions with ~95% accuracy using machine learning. It includes:

- Real-time prediction API using Flask
- Trained XGBoost model
- Scaled and balanced input data
- Evaluation via confusion matrix

---

## 🗂️ Project Structure

credit-card-fraud-detection/
├── .gitignore
├── app.py # Flask API for model inference
├── client.py # Sends requests to Flask API
├── creditcard.csv # Dataset (Kaggle Credit Card Fraud)
├── fraud_detection_model.pkl # Trained XGBoost model
├── main.py # Training script
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── scaler.pkl # Fitted MinMaxScaler
