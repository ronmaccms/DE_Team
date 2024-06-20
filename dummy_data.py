import pandas as pd

data = {
    'population_density': [2000, 3000, 1500, 2500, 1800],
    'median_income': [55000, 60000, 50000, 62000, 57000],
    'employment_rate': [0.95, 0.90, 0.92, 0.93, 0.94],
    'climate_index': [5, 4, 6, 5, 4],
    'cost_of_living_index': [3, 4, 3, 4, 3],
    'health_care_index': [4, 5, 3, 4, 5]
}

df = pd.DataFrame(data)
df.to_csv('dummy_data.csv', index=False)
print("Dummy data CSV file created.")
