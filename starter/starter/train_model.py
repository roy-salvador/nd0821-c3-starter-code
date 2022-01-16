# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model
import pandas as pd
from joblib import dump

# Add code to load in the data.
data = pd.read_csv("../data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Process train data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save model and one hot encoder
model = train_model(X_train, y_train)
print("Completed training model.")
print("Saving to model and encoder to `model` directory")
dump(model, "../model/rf_model.joblib")
dump(encoder, "../model/one_hot_encoder.joblib")
