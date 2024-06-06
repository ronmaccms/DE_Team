import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_pca(data, numerical_columns):
    numerical_data = data[numerical_columns].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numerical_data)
    pca = PCA()
    pca.fit(data_scaled)
    data_pca = pca.transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(data, numerical_columns):
    correlation_matrix = data[numerical_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix between Principal Components')
    plt.show()

def plot_residuals(residuals):
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_predicted_vs_actual(predictions, actual):
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, actual, alpha=0.5, color='blue')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2, color='red')
    plt.title('Scatter Plot of Predicted vs. Actual Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.grid(True)
    plt.show()

def plot_categorical_histograms(data, categorical_cols):
    for col in categorical_cols:
        counts = data[col].value_counts()
        plt.figure(figsize=(12, 6))
        counts.plot(kind='bar')
        plt.title(f'Number of Entries by {col}')
        plt.xlabel(col)
        plt.ylabel('Number of Entries')
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

def plot_numerical_histograms(data, numerical_cols):
    for col in numerical_cols:
        data_col = data[col].dropna()
        plt.figure(figsize=(12, 6))
        plt.hist(data_col, bins=30, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
