from fastapi.testclient import TestClient
import sys

sys.path.append('starter/')
sys.path.append('starter/starter/')
if True:
    from main import app

# Instantate app client
client = TestClient(app)


def test_greet():
    """
    Test GET method on root endpoint returns expected welcome message
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to Salary Predictor! " +
                        "At this time, we can only predict whether " +
                        "a person has >50K or <=50K salary."}


def test_salary_predict_negative():
    """
    Test POST method on /predict endpoint returns expected negative prediction
    """
    r = client.post("/predict/",
                    json={"age": 39,
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
                    )
    assert r.status_code == 200
    assert r.json() == {"salary": "<=50K"}


def test_salary_predict_positive():
    """
    Test POST method on /predict endpoint returns expected positive prediction
    """
    r = client.post("/predict/",
                    json={"age": 52,
                          "workclass": "Self-emp-inc",
                          "fnlgt": 287927,
                          "education": "HS-grad",
                          "education-num": 9,
                          "marital-status": "Married-civ-spouse",
                          "occupation": "Exec-managerial",
                          "relationship": "Wife",
                          "race": "White",
                          "sex": "Female",
                          "capital-gain": 15024,
                          "capital-loss": 0,
                          "hours-per-week": 40,
                          "native-country": "United-States"
                          }
                    )
    assert r.status_code == 200
    assert r.json() == {"salary": ">50K"}
