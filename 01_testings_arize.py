from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments, Schema, Metrics


# importing the api key to the python environment
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
API_KEY = os.environ["API_KEY"]
SPACE_ID = os.environ["SPACE_ID"]

# API_KEY = ''
# SPACE_ID = ''
arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)

# Step 2: Download Dataset
from sklearn.datasets import load_breast_cancer
breast_cancer_dataset = load_breast_cancer()


"""
Breakdown of breast_cancer_dataset contents
breast_cancer_dataset.data → Feature matrix (numeric values).

breast_cancer_dataset.feature_names → List of feature names.

breast_cancer_dataset.target → Labels (0 = malignant, 1 = benign).

breast_cancer_dataset.target_names → Label names.

breast_cancer_dataset.DESCR → Full description of dataset."""


# Step 3: Extract Features, Predictions, and Actuals
breast_cancer_features = breast_cancer_dataset['data'] # feature data
breast_cancer_feature_names = breast_cancer_dataset['feature_names'] # feature names
breast_cancer_targets = breast_cancer_dataset['target'] # actual data
breast_cancer_target_names = breast_cancer_dataset['target_names'] # actual labels


# Assign breast_cancer_target_names to their corresponding breast_cancer_targets to use as a human-comprehensible list of actual labels.
target_name_transcription = [] # this will become our list of actuals

for i in breast_cancer_targets: 
    target_name_transcription.append(breast_cancer_target_names[i])
    
# Create a Pandas dataframe to use the Arize Python Pandas logger with our predefined features and actuals(target_name_transcription).
import pandas as pd

df = pd.DataFrame(breast_cancer_features, columns=breast_cancer_feature_names)
df['actual_label'] = target_name_transcription
df['prediction_label'] = target_name_transcription

# this is optional, but makes this example more interesting in the platform
df['prediction_label'] = df['prediction_label'].iloc[::-1].reset_index(drop=True)

# Step 4: Log Data to Arize
# Define the Schema so Arize knows what your columns correspond to. Log the model data.
schema = Schema(
    actual_label_column_name="actual_label",
    prediction_label_column_name="prediction_label",
    feature_column_names=[
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error',
        'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points',
        'worst symmetry', 'worst fractal dimension'
       ]
)

response = arize_client.log(
    dataframe=df,
    schema=schema,
    model_id='breast_cancer_dataset', 
    model_version='v1',
    model_type=ModelTypes.BINARY_CLASSIFICATION,
    metrics_validation=[Metrics.CLASSIFICATION],
    environment=Environments.PRODUCTION
)

