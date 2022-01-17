# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference
from ml.model import compute_model_metrics, compute_slice_metrics
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

# Process the test data with the process_data function.
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
print("Saving model, encoder and label binarizer  to `model` directory")
dump(model, "../model/rf_model.joblib")
dump(encoder, "../model/one_hot_encoder.joblib")
dump(lb, "../model/label_binarizer.joblib")

# Compute performance on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
with open('slice_output.txt', 'w') as f:
    f.write("====================================\n")
    f.write("Overall Performance on Test Set:\n")
    f.write("====================================\n")
    f.write("Precision: {:.2f}".format(precision) + "\n")
    f.write("Recall: {:.2f}".format(recall) + "\n")
    f.write("FBeta: {:.2f}".format(fbeta) + "\n")

    # Compute performance on each slice of each categorical field
    for feature in cat_features:
        f.write("-----------------------------------------\n")
        f.write("Performance By " + feature + "\n")
        f.write("-----------------------------------------\n")
        df = compute_slice_metrics(test,
                                   feature,
                                   categorical_features=cat_features,
                                   label="salary",
                                   model=model,
                                   encoder=encoder,
                                   lb=lb
                                   )
        f.write(df.to_string() + "\n")


print("Overall and by slice performance written to slice_output.txt")
