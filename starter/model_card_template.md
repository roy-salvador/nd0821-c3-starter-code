# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* The model is developed by Roy Salvador for Udacity Machine Learning DevOps Nanodegree Project, Jan 2022, v0.0.1 
* A Random Forest Classifier trained with [default sklearn parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for determining whether a person makes over 50K a year.

## Intended Use
* This is only intended for proof-of-concept applications demonstrating the ability to automatically detect salary bracket of a person. A useful application of this would be to use the prediction as decision factor for targetting customers in sales environment.

## Training Data
* 80% Training data split taken from [UCI Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data
* 20% Test data split also taken from [UCI Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)

## Metrics
* As there are only two salary brackets in the dataset, hence a binary classification, the model is evaluated with Precision, Recall and FBeta scores. Overall, it achieved Precision: 0.73, Recall: 0.61, and FBeta: 0.67. For detailed performance per categorical feature, check [slice_output.txt](starter/slice_output.txt).

## Ethical Considerations
* Although the data used to train is anonymized , the model uses personal information like race. No new information is inferred or annotated.

## Caveats and Recommendations
* There are missing values in the data. 
* As the dataset used was curated in 1996, some features like sex may not reflect a more current categorization, limiting inclusivity. 
