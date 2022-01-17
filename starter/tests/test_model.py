import sys
import pandas as pd
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import os

# Set path to so we can import our code
sys.path.append('starter/starter')
if True:
    from ml.model import train_model
    from ml.model import compute_model_metrics
    from ml.model import inference
    from ml.data import process_data

file_dir = os.path.dirname(__file__)
model_dir = os.path.join(file_dir, '..', 'model')
data_dir = os.path.join(file_dir, '..', 'data')


@pytest.fixture
def model():
    """
    Load trained Random Forest model
    """
    return load(os.path.join(model_dir, 'rf_model.joblib'))


@pytest.fixture
def encoder():
    """
    Load one hot encoder
    """
    return load(os.path.join(model_dir, 'one_hot_encoder.joblib'))


@pytest.fixture
def lb():
    """
    Load label binarizer
    """
    return load(os.path.join(model_dir, 'label_binarizer.joblib'))


@pytest.fixture
def num_samples():
    """
    The number of data samples to include for testing
    picked randomly from 5 to 10
    """
    return int(np.random.randint(5, 11, size=1))


@pytest.fixture
def data(encoder, lb, num_samples):
    """
    Sample few records for testing.
    """
    df = pd.read_csv(os.path.join(data_dir, 'census_cleaned.csv'))
    df = df.sample(num_samples)

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
    data = {}
    data['X'], data['y'], _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return data


@pytest.fixture
def X(data):
    """
    Sample X features
    """
    return data['X']


@pytest.fixture
def y(data):
    """
    Sample X features
    """
    return data['y']


@pytest.fixture
def preds(num_samples):
    """
    Dummy predictions
    """
    preds = np.random.randint(2, size=num_samples)
    return preds


def test_train_model(X, y):
    """
    Test we have successfully trained a Random Classifier model
    """
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(y, preds):
    """
    Test computed metrics are probabilities
    """
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_inference(model, X):
    """
    Test model can ouput predictions and each data sample
    has corresponding prediction
    """
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(X) == len(preds)
