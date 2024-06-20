import requests
import pandas as pd
import joblib

url = 'http://127.0.0.1:5000/predict'

# Load feature names used during training
feature_names_path = 'pkl/xgboost_feature_names_0620.pkl'
feature_names = joblib.load(feature_names_path)

# Create a sample input matching the feature names
sample_input = {feature: 0 for feature in feature_names}  # Set default values to 0
sample_input.update({
    "population_density": 2000,
    "median_income": 55000,
    "employment_rate": 0.95
})

response = requests.post(url, json=sample_input)
print(f"Response: {response.json()}")
