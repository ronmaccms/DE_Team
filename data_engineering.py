import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_csvs(file_paths):
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            print(f"Loaded {path} successfully.")
        except FileNotFoundError:
            print(f"File {path} not found.")
        except Exception as e:
            print(f"An error occurred while loading {path}: {e}")
    return dataframes

def clean_data(df):
    # Remove leading/trailing spaces from string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Calculate the mean for each numeric column grouped by country
    country_means = df.groupby('country')[numerical_cols].mean().round(1)
    for col in numerical_cols:
        for country in country_means.index:
            df.loc[(df['country'] == country) & (df[col].isna()), col] = country_means.loc[country, col]
    
    # Fill NaN values in numeric columns with the median of each column
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df

def preprocess_data(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def apply_pca(df, n_components=None):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df[numerical_cols])
    explained_variance = pca.explained_variance_ratio_.cumsum()
    return df_pca, explained_variance

def get_cleaned_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_paths = [os.path.join(base_path, "data_pool/net_mig_clean.csv")]
    all_cities_kpi = load_csvs(file_paths)
    if len(all_cities_kpi) == 0:
        raise FileNotFoundError("No CSV files loaded.")
    net_mig_clean = all_cities_kpi[0]

    net_mig_clean = clean_data(net_mig_clean)
    net_mig_clean = preprocess_data(net_mig_clean)

    df_pca, explained_variance = apply_pca(net_mig_clean)
    
    return net_mig_clean, df_pca, explained_variance

if __name__ == "__main__":
    # Ensure the data_pool directory exists
    output_dir = 'data_pool'
    os.makedirs(output_dir, exist_ok=True)

    net_mig_clean, df_pca, explained_variance = get_cleaned_data()
    
    # Save processed data to the data_pool directory
    net_mig_clean.to_csv(os.path.join(output_dir, 'processed_net_mig_clean.csv'), index=False)
    np.save(os.path.join(output_dir, 'explained_variance.npy'), explained_variance)
    np.save(os.path.join(output_dir, 'df_pca.npy'), df_pca)
