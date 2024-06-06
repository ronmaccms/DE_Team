from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    av_train = np.average(y_train)
    av_test = np.average(y_test)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, predictions_train, squared=False) / av_train
    test_rmse = mean_squared_error(y_test, predictions_test, squared=False) / av_test

    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    residuals_train = (y_train - predictions_train)
    residuals_test = (y_test - predictions_test)

    return train_rmse, test_rmse, residuals_test, predictions_test
