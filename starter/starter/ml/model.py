from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
import pandas as pd

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # create and train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
        data,
        feature,
        categorical_features,
        label,
        model,
        encoder,
        lb):
    """
    Validates the trained machine learning model using precision,
    recall, and F1 for each slice of data in the given feature

    Inputs
    ------
    data : pd.Dataframe
        Dataframe containing the features and label.
        The dataset to perform the test on
    feature : str
        The categorical feature to use for slicing
    categorical_features: list[str]
        List containing the names of the categorical features
    label : str
        Name of the label column in `X`.
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer

    Returns
    -------
    result : pd.Dataframe
        Dataframe containing the performance results of each slice (row).
        Columns include the following:
            precision : float
            recall : float
            fbeta : float
    """
    result = pd.DataFrame(columns=['slice', 'precision', 'recall', 'fbeta'])

    for slice_val in data[feature].unique():

        # Filter the slice and encode
        X, y, encoder, lb = process_data(
            data[data[feature] == slice_val],
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Get predictions for the slice
        preds = inference(model, X)

        # Calculate performance metrics for the slice and append to dataframe
        result.loc[len(result)] = [slice_val] + \
            list(compute_model_metrics(y, preds))

    return result
