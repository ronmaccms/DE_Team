# Notes:
# main.py: The main script orchestrates the loading, preprocessing, modeling, and plotting functions.
# data_processing.py: Contains functions related to data loading and preprocessing.
# models.py: Contains functions for training and evaluating the model.
# plots.py: Contains functions for plotting different visualizations.


from data_processing import load_csvs, preprocess_data, prepare_data_for_modeling
from models import train_and_evaluate_model
from plots import plot_pca, plot_correlation_matrix, plot_residuals, plot_predicted_vs_actual, plot_categorical_histograms, plot_numerical_histograms

def main():
    # List of file paths
    file_paths = [
        r"https://raw.githubusercontent.com/ronmaccms/DE_Team/main/modified_csvs/cities_kpis_final2.csv",
        r"https://raw.githubusercontent.com/ronmaccms/DE_Team/main/merged_csvs/merged_filled3.csv",
        r"https://raw.githubusercontent.com/ronmaccms/DE_Team/main/merged_csvs/cities_missing_data.csv"
    ]

    # Load and preprocess data
    all_cities_kpi = load_csvs(file_paths)
    migration_database, numerical_columns, categorical_cols = preprocess_data(all_cities_kpi)

    # Plot PCA and correlation matrix
    plot_pca(migration_database, numerical_columns)
    plot_correlation_matrix(migration_database, numerical_columns)

    # Prepare data for modeling
    modelling_db, X_train, X_test, y_train, y_test = prepare_data_for_modeling(migration_database)

    # Train and evaluate model
    train_rmse, test_rmse, residuals_test, predictions_test = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # Plot results
    plot_residuals(residuals_test)
    plot_predicted_vs_actual(predictions_test, y_test)

    # Plot histograms for categorical and numerical data
    plot_categorical_histograms(migration_database, categorical_cols)
    plot_numerical_histograms(migration_database, numerical_cols)

if __name__ == "__main__":
    main()
