# XGBoostRegressor.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from data_engineering import get_cleaned_data
from datetime import datetime

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse, predictions

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    train_rmse, train_predictions = evaluate_model(model, X_train, y_train)
    test_rmse, test_predictions = evaluate_model(model, X_test, y_test)
    
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    
    # Save the model as .pkl and .bin with model name included
    model_name = 'xgboost'
    joblib.dump(model, os.path.join(pkl_dir, f'{model_name}_{current_date}.pkl'))
    with open(os.path.join(bins_dir, f'{model_name}_{current_date}.bin'), 'wb') as bin_file:
        joblib.dump(model, bin_file)

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
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
    plt.title(f'Scatter Plot of Predicted vs. Actual Values - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_scatter_{current_date}_03.png'))
    plt.close()
    
    # Feature Importance Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title(f"Feature Importance - {model_name}")
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]), [features_list[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_feature_importance_{current_date}_04.png'))
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
    NetMig_test = X_test.iloc[:, 7]  # Assuming the 8th column (index 7) is of interest
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
