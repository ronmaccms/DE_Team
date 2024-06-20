from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define paths
model_path = 'pkl/xgboost_0619.pkl'
scaler_path = 'pkl/xgboost_scaler_0619.pkl'

# Check if the files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

# Load the trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# List of features used during training
feature_names = ['population_density', 'median_income', 'employment_rate', 'climate_index', 'cost_of_living_index', 'health_care_index']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)
        # Assume data is a dictionary with feature names as keys and values as feature values
        input_data = pd.DataFrame([data])
        # Ensure all necessary columns are present, filling with zeros if they are missing
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        input_data = input_data[feature_names]  # Reorder columns to match training
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
