# main.py
import os
import pandas as pd
from dotenv import load_dotenv
from joblib import load
from kaggle.api.kaggle_api_extended import KaggleApi
from baseline_models import train_and_evaluate_baselines
from process_text import preprocess_text_data
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

# Load environment variables
load_dotenv()
dataset_name = os.getenv("KAGGLE_DATASET")
# train_data_pth = os.getenv("TRAIN_DATA_PATH")
# test_data_pth = os.getenv("TEST_DATA_PATH")
# test_labels_pth = os.getenv("TEST_LABELS_PATH")
train_data_pth = "data/train.csv"
test_data_pth = "data/test.csv"


# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# # Download dataset if not already
# if not os.path.exists(train_data_pth):
#     api.competition_download_file(dataset_name, "train.csv", path=os.path.dirname(train_data_pth))
#     api.competition_download_file(dataset_name, "test.csv", path=os.path.dirname(test_data_pth))
#     api.competition_download_file(dataset_name, "test_labels.csv", path=os.path.dirname(test_labels_pth))
#     print("✅ Downloaded dataset from Kaggle")

## Load and preprocess training data
train_df = pd.read_csv(train_data_pth)
train_df = preprocess_text_data(train_df)

# Prepare inputs
X_text = train_df['cleaned']
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Train and evaluate baselines
print("🔍 Training and evaluating baseline models...")
best_model_name = train_and_evaluate_baselines(X_text, y)

# Load test set and preprocess
print
test_df = pd.read_csv(test_data_pth)
test_df = preprocess_text_data(test_df)
X_test_text = test_df['cleaned']

# Load best model and vectorizer
print(f"🔍 Loading best model: {best_model_name}")
model = load(f"results/models/{best_model_name}.joblib")
vectorizer = load("results/models/vectorizer.joblib")
X_test = vectorizer.transform(X_test_text)

# Predict on test set
print("🔍 Making predictions on test set...")
y_test_pred = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame(y_test_pred, columns=y.columns)
submission.insert(0, 'id', test_df['id'])
submission.to_csv("submission.csv", index=False)
print("📤 submission.csv created")

# Submit to Kaggle
api.competition_submit(
    file_name="submission.csv",
    message=f"Baseline submission: {best_model_name}",
    competition=dataset_name
)
print("🚀 Submitted to Kaggle")
