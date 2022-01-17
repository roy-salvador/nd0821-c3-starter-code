# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from joblib import load
import sys

# Set path to so we can import our code
sys.path.append('starter')
if True:
    from ml.data import process_data
    from ml.model import inference


# Data object defining needed components for model input
class ModelInputItem(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example='State-gov')
    fnlgt: int = Field(example=77516)
    education: str = Field(example='Bachelors')
    education_num: int = Field(alias='education-num', example=13)
    marital_status: str = Field(
        alias='marital-status',
        example='Never-married')
    occupation: str = Field(example='Adm-clerical')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(alias='capital-gain', example=2174)
    capital_loss: int = Field(alias='capital-loss', example=0)
    hours_per_week: int = Field(alias='hours-per-week', example=40)
    native_country: str = Field(
        alias='native-country',
        example='United-States')


# Load model and encoder
model = load('model/rf_model.joblib')
encoder = load('model/one_hot_encoder.joblib')
lb = load('model/label_binarizer.joblib')

# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Instantiate the app.
app = FastAPI()

# GET method for greeting on the root endpoint.


@app.get("/")
async def greet():
    return {"greeting": "Welcome to Salary Predictor! " +
            "At this time, we can only predict whether " +
            "a person has >50K or <=50K salary."}

# POST method for model inference


@app.post("/predict/")
async def salary_predict(item: ModelInputItem):
    test = pd.DataFrame([item.dict(by_alias=True)])
    X, _, _, _ = process_data(test,
                              categorical_features=cat_features,
                              training=False,
                              encoder=encoder
                              )
    preds = inference(model, X)
    salary_class = lb.inverse_transform(preds)
    return {"salary": salary_class[0]}
