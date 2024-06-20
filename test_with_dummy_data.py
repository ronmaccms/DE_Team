import requests
import pandas as pd
import joblib
import os
import numpy as np

# Define the URL of your prediction endpoint
url = 'http://127.0.0.1:5000/predict'

# Define the URL from which to download the feature names file
url_to_download = 'https://github.com/ronmaccms/DE_Team/raw/working_test/pkl/xgboost_feature_names_0619.pkl'

# Local path where you want to save the downloaded file
local_path = './pkl/xgboost_feature_names_0619.pkl'

# Check if the file already exists locally, otherwise download it
if not os.path.exists(local_path):
    response = requests.get(url_to_download)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded file from {url_to_download}")

# Load feature names from the local file
feature_names = joblib.load(local_path)
print(f"Feature names loaded: {feature_names}")

# Load dummy_data.csv into a DataFrame
dummy_data_path = './dummy_data.csv'
dummy_data = pd.read_csv(dummy_data_path)
print(f"Columns in dummy data before alignment: {dummy_data.columns.tolist()}")

# Ensure dummy_data matches the expected feature names exactly
dummy_data = dummy_data[feature_names]
print(f"Columns in dummy data after alignment: {dummy_data.columns.tolist()}")

# Check for and replace NaN or infinite values
dummy_data.replace([np.inf, -np.inf], np.nan, inplace=True)
dummy_data.fillna(0, inplace=True)  # Replace NaN with 0 (or appropriate default)

# Convert the dummy data to the correct format for the request
sample_input = dummy_data.to_dict(orient='list')

# Make a POST request to the local server with the dummy_data
response = requests.post(url, json=sample_input)
print(f"Response: {response.json()}")
