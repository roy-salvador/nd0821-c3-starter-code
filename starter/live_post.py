import requests
import json

# Get url from user and perform a POSR
url = input("Enter live api url: ")
data = {"age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
        }
response = requests.post(url, data=json.dumps(data))

# Prompt informational message
print("Using dummy data: " + str(data))
print("================")
print("RESPONSE")
print("================")
print(response.status_code)
print(response.json())
