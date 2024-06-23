import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess the input data
def preprocess_input(data, label_encoders, scaler):
    # Convert input data to DataFrame
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    for column in ['Gender', 'Blood Type', 'Insurance Provider', 'Admission Type', 'Medication']:
        df[column] = df[column].map(lambda s: label_encoders[column].transform([s])[0] if s in label_encoders[column].classes_ else -1)
    
    # Standardize the features
    df = scaler.transform(df)
    
    return df

# Load the model, label encoders, and scaler
model = joblib.load('medical_condition_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Assuming these are the actual target classes from your dataset
target_classes = [
    'Hypertension',
    'Diabetes Mellitus',
    'Asthma',
    'Diabetes'
    'Chronic Obstructive Pulmonary Disease (COPD)',
    'Coronary Artery Disease',
    'Heart Failure',
    'Stroke',
    'Chronic Kidney Disease',
    'Osteoarthritis',
    'Rheumatoid Arthritis',
    'Depression',
    'Anxiety Disorder',
    'Alzheimer\'s Disease',
    'Parkinson\'s Disease',
    'Hyperlipidemia',
    'Hypothyroidism',
    'Hyperthyroidism',
    'Psoriasis',
    'Multiple Sclerosis',
    'Lupus',
    'Inflammatory Bowel Disease',
    'Gastroesophageal Reflux Disease (GERD)',
    'Hepatitis',
    'HIV/AIDS',
    'Tuberculosis',
    'Malaria',
    'COVID-19',
    'Cancer',
    'Migraine',
    'Epilepsy'
]
  # Replace with actual target conditions
le_target = LabelEncoder()
le_target.fit(target_classes)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from POST request
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Preprocess the input data
    preprocessed_data = preprocess_input(data, label_encoders, scaler)
    
    # Make a prediction
    prediction = model.predict(preprocessed_data)
    predicted_condition = le_target.inverse_transform(prediction)
    
    return jsonify({'predicted_condition': predicted_condition[0]})

if __name__ == '__main__':
    app.run(debug=True)
