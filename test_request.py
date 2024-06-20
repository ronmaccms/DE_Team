# conda activate DE-Test1

import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'Net_Migration': 3,
    'country': 5,
    'city': 6,
}

response = requests.post(url, json=data)
print(response.json())




#conda activate DE-Test1