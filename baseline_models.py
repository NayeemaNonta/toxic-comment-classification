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
from joblib import dump
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # For multi-label
        )

    def forward(self, x):
        return self.fc(x)

def train_nn_baseline(X_text, y, output_dir="results", seed=42, batch_size=64, epochs=10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Vectorize
    # vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    # X = vectorizer.fit_transform(X_text).astype(np.float32)
    # y = y.values.astype(np.float32)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X_text)
    dump(vectorizer, f"{output_dir}/models/vectorizer_nn.joblib")

    # Convert to tensors safely
    X = torch.from_numpy(X_vec.toarray()).float()
    y = torch.from_numpy(y.values).float()

    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)

    dump(vectorizer, f"{output_dir}/models/vectorizer_nn.joblib")

    # Dataset
    dataset = TensorDataset(torch.tensor(X.toarray()), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = SimpleNN(X.shape[1], y.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train 
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X.toarray())).numpy()

    y_pred_binary = (y_pred > 0.5).astype(int)
    report = classification_report(y, y_pred_binary, target_names=[str(col) for col in range(y.shape[1])], zero_division=0)

    # Metrics
    f1 = f1_score(y, y_pred_binary, average="macro", zero_division=0)
    aucs = []
    for i in range(y.shape[1]):
        try:
            auc = roc_auc_score(y[:, i], y_pred[:, i])
        except:
            auc = np.nan
        aucs.append(auc)
    mean_auc = np.nanmean(aucs)

    # Save
    torch.save(model.state_dict(), f"{output_dir}/models/nn_baseline.pt")
    with open(f"{output_dir}/reports/classification_report_nn.txt", "w") as f:
        f.write("=== Baseline: NN ===\n\n")
        f.write(report)
        f.write(f"\n\nMacro F1: {f1:.4f}\nMean ROC AUC: {mean_auc:.4f}\n")

    return {
        "model": "nn_baseline",
        "macro_f1": f1,
        "mean_roc_auc": mean_auc
    }
    
def train_and_evaluate_baselines(X_text, y, output_dir="results", SEED=42):
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

  
    # Split data: 80% train, 20% test
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=SEED)

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

        # Save report
        report = classification_report(y_test, y_pred, target_names=y.columns, zero_division=0)
        report_path = f"{output_dir}/reports/classification_report_{name}.txt"
        with open(report_path, "w") as f:
            f.write(f"=== Baseline: {name} ===\n\n")
            f.write(report)
            f.write("\n\n=== Metrics on 20% Test Set ===\n")
            f.write(f"Macro F1: {f1:.4f}\n")
            f.write(f"Mean ROC AUC: {mean_auc:.4f}\n")
        print(f"✅ Saved report: {report_path}")

        # Save model
        dump(model, f"{output_dir}/models/{name}.joblib")

        # Save metrics
        scores.append({
            "model": name,
            "macro_f1": f1,
            "mean_roc_auc": mean_auc
        })

        
    # # Train Neural Network Baseline
    # print("\n Training neural network baseline")
    # nn_metrics = train_nn_baseline(X_text, y, output_dir=output_dir, seed=SEED)
    # scores.append(nn_metrics)

    # Save all scores
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f"{output_dir}/baseline_scores.csv", index=False)

    # Select best model by AUC
    best_model = scores_df.sort_values(by="mean_roc_auc", ascending=False).iloc[0]['model']
    with open(f"{output_dir}/best_model.txt", "w") as f:
        f.write(best_model)

    print(f"\n🥇 Best model by AUC: {best_model}")
    return best_model

