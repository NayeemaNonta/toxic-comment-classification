import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from joblib import dump
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_and_evaluate_baselines(X_text, y, test_split=0.2, output_dir="results", SEED=42):
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Split data: 80% train, 20% test
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=test_split, random_state=SEED)

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    os.makedirs(f"{output_dir}/models", exist_ok=True)
    dump(vectorizer, f"{output_dir}/models/vectorizer.joblib")

    # Define baselines
    baselines = {
        "logistic_liblinear": OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='liblinear', random_state=SEED)),
        "naive_bayes": OneVsRestClassifier(MultinomialNB())
    }

    scores = []
    os.makedirs(f"{output_dir}/reports", exist_ok=True)

    # Train and Evaluate
    for name, model in baselines.items():
        print(f"\n Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Predicted probabilities (for AUC)
        y_proba = model.predict_proba(X_test)
        if isinstance(y_proba, list):
            y_proba = np.array([p[:, 1] for p in y_proba]).T

        # Compute AUC safely per column
        aucs = []
        for i in range(y_test.shape[1]):
            try:
                auc = roc_auc_score(y_test.iloc[:, i], y_proba[:, i])
            except ValueError:
                auc = np.nan
            aucs.append(auc)

        mean_auc = np.nanmean(aucs)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

        # Save report
        report = classification_report(y_test, y_pred, target_names=y.columns, zero_division=0)
        report_path = f"{output_dir}/reports/classification_report_{name}.txt"
        with open(report_path, "w") as f:
            f.write(f"=== Baseline: {name} ===\n\n")
            f.write(report)
            f.write("\n\n=== Metrics on 20% Test Set ===\n")
            f.write(f"Macro F1: {f1:.4f}\n")
            f.write(f"Macro Precision: {precision:.4f}\n")
            f.write(f"Macro Recall: {recall:.4f}\n")
            f.write(f"Mean ROC AUC: {mean_auc:.4f}\n")
        print(f"✅ Saved report: {report_path}")

        # Save model
        dump(model, f"{output_dir}/models/{name}.joblib")

        # Save metrics
        scores.append({
            "model": name,
            "macro_f1": f1,
            "macro_precision": precision,
            "macro_recall": recall,
            "mean_roc_auc": mean_auc
        })

    # Save all scores
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f"{output_dir}/baseline_scores.csv", index=False)

    # Select best model by AUC
    best_model = scores_df.sort_values(by="mean_roc_auc", ascending=False).iloc[0]['model']
    with open(f"{output_dir}/best_model.txt", "w") as f:
        f.write(best_model)

    print(f"\n🥇 Best model by AUC: {best_model}")
    return best_model