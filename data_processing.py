import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def preprocess_data(all_cities_kpi):
    kpis_clean = all_cities_kpi[0]
    kpis_initial = all_cities_kpi[1]
    cities_missing_data = all_cities_kpi[2]

    columns_to_drop_first_set = [
        'Rank_x', 'Cost of Living Plus Rent Index',
        'Restaurant Price Index', 'Rank_y',
        'Rank', 'key', 'Cost of Living Index_y'
    ]

    kpis_clean.drop(columns=columns_to_drop_first_set, inplace=True)

    columns_to_drop_second_set = [
        'Price To Income Ratio', 'Price To Rent Ratio City Centre',
        'Price To Rent Ratio Outside Of City Centre',
        'Property Price to Income Ratio'
    ]

    kpis_clean.drop(columns=columns_to_drop_second_set, inplace=True)

    repeated_cities = list(kpis_initial['City'].unique())

    filtered_df_01 = kpis_clean[kpis_clean['City'].isin(repeated_cities)].copy()
    filtered_df_02 = kpis_clean[~kpis_clean['City'].isin(repeated_cities)].copy()

    filtered_df_02_sorted = filtered_df_02.sort_values(by=['Country_x', 'Population'], ascending=[True, False])
    top_6_populations = filtered_df_02_sorted.groupby('Country_x').head(6)

    clean_appended_cities_kpi = top_6_populations
    initial_cities_kpi = filtered_df_01

    combined_df = pd.concat([clean_appended_cities_kpi, initial_cities_kpi])
    migration_database = combined_df.sort_values(by=['Country_x', 'Population'], ascending=[True, False])

    cities_missing_data.drop(columns=["City"], inplace=True)
    migration_database.reset_index(drop=True, inplace=True)
    cities_missing_data.reset_index(drop=True, inplace=True)
    migration_database = pd.concat([migration_database, cities_missing_data], axis=1)
    migration_database.drop(columns=["Net migration (thousands)"], inplace=True)

    categorical_cols = [col for col in migration_database.columns if migration_database[col].dtype == 'object']
    numerical_cols = [col for col in migration_database.columns if migration_database[col].dtype in ['int64', 'float64']]

    return migration_database, numerical_cols, categorical_cols

def prepare_data_for_modeling(migration_database):
    numerical_columns = [
        'Population', 'Cost of Living Index_x', 'Rent Index', 'Groceries Index',
        'Local Purchasing Power Index', 'Gross Rental Yield City Centre',
        'Gross Rental Yield Outside of Centre', 'Mortgage As A Percentage Of Income',
        'Affordability Index', 'Quality of Life Index', 'Purchasing Power Index',
        'Safety Index', 'Health Care Index', 'Traffic Commute Time Index',
        'Pollution Index', 'Climate Index'
    ]
    numerical_data = migration_database[numerical_columns].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numerical_data)

    final_numerical = pd.DataFrame(data_scaled, columns=numerical_columns).reset_index(drop=True)
    final_categorical = migration_database[['First_language', 'Second_language']].copy().reset_index(drop=True)

    final_modelling = pd.concat([final_categorical, final_numerical, migration_database['Net_migration'].reset_index(drop=True)], axis=1)
    modelling_db = final_modelling[~final_modelling['Net_migration'].isna()].copy()

    modelling_db['rand'] = np.random.choice(['train', 'test'], size=len(modelling_db), p=[0.8, 0.2])
    modelling_db = pd.get_dummies(modelling_db, columns=['First_language', 'Second_language'])

    features_list = [col for col in modelling_db.columns if col not in ['Net_migration', 'rand']]
    X_train = modelling_db[features_list][modelling_db['rand'] == 'train'].copy()
    X_test = modelling_db[features_list][modelling_db['rand'] == 'test'].copy()
    y_train = modelling_db['Net_migration'][modelling_db['rand'] == 'train'].copy()
    y_test = modelling_db['Net_migration'][modelling_db['rand'] == 'test'].copy()

    return modelling_db, X_train, X_test, y_train, y_test
