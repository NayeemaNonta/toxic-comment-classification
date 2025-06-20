import os
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump
from dotenv import load_dotenv
from process_text import preprocess_text_data

# Setup
load_dotenv()
os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

train_data_pth = os.getenv("TRAIN_DATA_PATH")
train_df = pd.read_csv(train_data_pth)
train_df = preprocess_text_data(train_df)

# Data
X_text = train_df['cleaned']
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(X_text)

# Define baselines
baselines = {
    "logistic_liblinear": OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='liblinear')),
    "logistic_saga": OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='saga')),
    "naive_bayes": OneVsRestClassifier(MultinomialNB()),
    "dummy_most_frequent": OneVsRestClassifier(DummyClassifier(strategy='most_frequent')),
}

# Run all models
scores = []

for name, model in baselines.items():
    print(f"\n🧪 Running baseline: {name}")
    model.fit(X, y)
    y_pred = model.predict(X)

    # Classification report
    report = classification_report(y, y_pred, target_names=y.columns, zero_division=0)
    report_path = f"results/reports/classification_report_{name}.txt"
    with open(report_path, "w") as f:
        f.write(f"=== Baseline: {name} ===\n\n")
        f.write(report)
    print(f"✅ Saved: {report_path}")

    # Save F1 scores for overview table
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    scores.append({"model": name, "macro_f1": f1})

    # Optional: save model
    model_path = f"results/models/{name}.joblib"
    dump(model, model_path)

# Save overview CSV
scores_df = pd.DataFrame(scores)
scores_df.to_csv("results/baseline_scores.csv", index=False)
print("\n📊 All macro F1 scores saved to results/baseline_scores.csv")
