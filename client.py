import requests
url = 'http://127.0.0.1:5001/predict'

example_features = [
    -1.223, 0.511, -0.332, 0.744, 0.110, -1.012,
    0.543, -0.765, 0.234, -1.012, 0.654, -0.321,
    0.432, -0.876, 0.987, -0.654, 0.543, -1.001,
    0.432, -0.543, 0.765, -1.432, 0.654, -0.321,
    0.543, -0.765, 0.234, -1.012, 0.654, -0.321
]
response = requests.post(url, json={'features': example_features})
print(response.json())
