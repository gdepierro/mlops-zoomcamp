import requests

ride = {
    'PULocationID': 10,
    'DOLocationID': 50,
    'trip_distance': 40
}

response = requests.post('http://localhost:9696/predict', json=ride)
print(response.json())