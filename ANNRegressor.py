# ANNRegressor.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_engineering import get_cleaned_data
from datetime import datetime

def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train):
    model = build_ann_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse, predictions

def create_report(model_name, train_rmse, test_rmse, current_date):
    report_content = f"""
Net Migration Analysis Report using {model_name} Regressor
=======================================================================

Overview
--------
This report provides an analysis and prediction of net migration using the {model_name} Regressor model. The model is trained on a dataset containing various features related to migration, and its performance is evaluated using RMSE (Root Mean Squared Error).

Model Performance
-----------------
- Train RMSE: {train_rmse:.2f}
- Test RMSE: {test_rmse:.2f}

Generated Plots and Their Interpretations
-----------------------------------------
The following plots are generated and saved in the `plots` directory:

1. Histogram of Net_Migration ({model_name}_histogram_{current_date}_01.png)
   - Shows the distribution of net migration values in the dataset.
   - Useful to understand the spread and central tendency of migration data.

2. Distribution of Residuals ({model_name}_distribution_{current_date}_02.png)
   - Displays the distribution of the residuals (errors) from the model's predictions.
   - Helps identify any patterns or biases in the model's predictions.

3. Scatter Plot of Predicted vs. Actual Values ({model_name}_scatter_{current_date}_03.png)
   - Plots the predicted net migration values against the actual values.
   - A good fit is indicated by points lying close to the diagonal line.

4. Residuals vs. Predicted Values Plot ({model_name}_residuals_vs_predicted_{current_date}_05.png)
   - Plots residuals against predicted values.
   - Useful for detecting heteroscedasticity and other patterns in the residuals.

5. Predicted vs. Actual Values Line Plot ({model_name}_predicted_vs_actual_line_{current_date}_06.png)
   - Line plot showing predicted and actual values across the test data points.
   - Useful for visually inspecting the model's performance over the dataset.

6. Cumulative Explained Variance by PCA ({model_name}_cumulative_explained_variance_{current_date}_07.png)
   - Shows the cumulative explained variance by the principal components.
   - Useful for understanding the variance explained by the PCA components.

7. Pairplot of Key Features ({model_name}_pairplot_{current_date}_08.png)
   - Pairwise plots of key features and their relationships.
   - Helps visualize correlations and interactions between features.

8. Comparison Plot ({model_name}_comparison_{current_date}_09.png)
   - Comparison of a specific feature's predicted vs. actual values.
   - Useful for detailed inspection of the model's performance on a particular feature.

9. Error Histogram ({model_name}_error_histogram_{current_date}_10.png)
    - Histogram of prediction errors.
    - Helps understand the distribution and magnitude of prediction errors.

Conclusion
----------
This project provides a comprehensive analysis of net migration using the {model_name} model. The generated plots and metrics help in understanding the factors influencing migration and the effectiveness of the model in predicting migration patterns.
"""
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    report_filename = os.path.join(report_dir, f'{model_name}_report_{current_date}.txt')
    with open(report_filename, 'w') as f:
        f.write(report_content)

if __name__ == "__main__":
    # Ensure the directories exist
    plots_dir = 'plots'
    pkl_dir = 'pkl'
    bins_dir = 'bins'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(bins_dir, exist_ok=True)
    
    # Get current date in MMDD format
    current_date = datetime.now().strftime("%m%d")

    net_mig_clean, df_pca, explained_variance = get_cleaned_data()
    
    features_list = [col for col in net_mig_clean.columns if col not in ['Net_Migration', 'country', 'city']]
    X = net_mig_clean[features_list]
    y = net_mig_clean['Net_Migration']
    
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    train_rmse, train_predictions = evaluate_model(model, X_train, y_train)
    test_rmse, test_predictions = evaluate_model(model, X_test, y_test)
    
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    # Print sample predictions for debugging
    print("Sample predictions:", test_predictions[:5])
    print("Actual values:", y_test.values[:5])
    
    # Save the model and scaler
    model_name = 'ann'
    model.save(os.path.join(pkl_dir, f'{model_name}_{current_date}.keras'))
    joblib.dump(scaler, os.path.join(pkl_dir, f'{model_name}_scaler_{current_date}.pkl'))

    # Create the report file
    create_report(model_name, train_rmse, test_rmse, current_date)

    # Visualize results and save plots with the specified naming convention
    plt.figure()
    plt.hist(net_mig_clean['Net_Migration'], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Net_Migration - {model_name}')
    plt.xlabel('Net_Migration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_histogram_{current_date}_01.png'))
    plt.close()

    residuals_test = y_test - test_predictions
    plt.figure()
    sns.histplot(residuals_test, bins=30, kde=True)
    plt.title(f'Distribution of Residuals - {model_name}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_distribution_{current_date}_02.png'))
    plt.close()

    plt.figure()
    plt.scatter(test_predictions, y_test, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'Scatter Plot of Predicted vs. Actual Values - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_scatter_{current_date}_03.png'))
    plt.close()
    
    # Residuals vs. Predicted Values Plot
    plt.figure()
    plt.scatter(test_predictions, residuals_test, alpha=0.5, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residuals vs. Predicted Values - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_residuals_vs_predicted_{current_date}_05.png'))
    plt.close()

    # Predicted vs. Actual Values Line Plot
    plt.figure()
    plt.plot(y_test.values, label='Actual Values', color='blue')
    plt.plot(test_predictions, label='Predicted Values', color='red', linestyle='dashed')
    plt.title(f'Predicted vs. Actual Values - {model_name}')
    plt.xlabel('Data Points')
    plt.ylabel('Net Migration')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_predicted_vs_actual_line_{current_date}_06.png'))
    plt.close()

    # Cumulative Explained Variance Plot
    plt.figure()
    plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b')
    plt.title(f'Cumulative Explained Variance by PCA - {model_name}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_cumulative_explained_variance_{current_date}_07.png'))
    plt.close()

    # Pairplot of Key Features
    key_features = features_list[:5]  # Select top 5 features for simplicity
    pairplot_data = net_mig_clean[key_features + ['Net_Migration']]
    sns.pairplot(pairplot_data, diag_kind='kde')
    plt.savefig(os.path.join(plots_dir, f'{model_name}_pairplot_{current_date}_08.png'))
    plt.close()
    
    # Plot comparison
    NetMig_test = X_test[:, 7]  # Assuming the 8th column (index 7) is of interest
    plt.figure()
    plt.scatter(NetMig_test, y_test, color='blue', label="Net Migration - True")
    plt.scatter(NetMig_test, test_predictions, color='red', label="Net Migration - Predicted")
    plt.xlabel('Net Migration Feature')
    plt.ylabel('Net Migration')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_comparison_{current_date}_09.png'))
    plt.close()
    
    # Error Histogram
    error = test_predictions - y_test
    plt.figure()
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_error_histogram_{current_date}_10.png'))
    plt.close()
