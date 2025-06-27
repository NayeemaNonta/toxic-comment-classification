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
from baseline_influence import estimate_influence_leave_one_out

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
print("train_df shape:", train_df.shape)
train_df = preprocess_text_data(train_df)

# Prepare inputs
X_text = train_df['cleaned']
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Train and evaluate baselines
print("🔍 Training and evaluating baseline models...")
best_model_name = train_and_evaluate_baselines(X_text, y)

# influences = estimate_influence_leave_one_out(X_text, y, sample_frac=0.1, max_examples=500)

# # Print top-10 most harmful examples
# print("\n🔍 Top harmful samples (most negative influence):")
# for idx, delta in influences[:10]:
#     print(f"\nSample index: {idx} | ΔF1: {delta:.4f}")
#     print("Comment:", X_text.iloc[idx])
#     print("Labels :", y.iloc[idx].values)

# # Ensure results folder exists
# os.makedirs("results", exist_ok=True)

# # Save top-10 most harmful examples to a file
# output_path = "results/top_10_influential_examples.txt"
# with open(output_path, "w") as f:
#     f.write("🔍 Top 10 Harmful Training Examples (Leave-One-Out Influence)\n")
#     f.write("=" * 60 + "\n")
#     for idx, delta in influences[:10]:
#         f.write(f"\nSample index: {idx} | ΔF1: {delta:.4f}\n")
#         f.write(f"Comment: {X_text.iloc[idx]}\n")
#         f.write(f"Labels : {y.iloc[idx].values.tolist()}\n")
#         f.write("-" * 60 + "\n")

# print(f"\n✅ Saved top-10 harmful samples to: {output_path}")

# top_k = 10
# harmful_indices = [idx for idx, _ in influences[:top_k]]

# print(f"\n🧹 Removing top {top_k} most harmful training examples...")
# X_text_cleaned = X_text.drop(index=harmful_indices).reset_index(drop=True)
# y_cleaned = y.drop(index=harmful_indices).reset_index(drop=True)

# print("🔁 Retraining on cleaned dataset...")
# best_model_after_cleaning = train_and_evaluate_baselines(X_text_cleaned, y_cleaned, output_dir="results_cleaned")

# # Save info
# with open("results/cleaning_summary.txt", "w") as f:
#     f.write(f"Removed {top_k} harmful training examples.\n")
#     f.write(f"Best model after cleaning: {best_model_after_cleaning}\n")

# print("✅ Finished retraining on cleaned dataset")

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
