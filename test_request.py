import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'population_density': 2000,
    'median_income': 55000,
    'employment_rate': 0.95,
    'climate_index': 5,
    'cost_of_living_index': 3,
    'health_care_index': 4
}

response = requests.post(url, json=data)
print(response.json())




#conda activate DE-Test1