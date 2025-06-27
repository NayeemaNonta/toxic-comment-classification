import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def estimate_influence_leave_one_out(X_text, y, sample_frac=0.2, random_state=42, max_examples=None):
    """
    Estimate influence of training samples via leave-one-out retraining on a random subset.
    
    Args:
        X_text: pd.Series of raw text
        y: pd.DataFrame of binary labels
        sample_frac: Fraction of training data to subsample for faster influence analysis
        random_state: Random seed for reproducibility
        max_examples: Optional cap on how many leave-one-out samples to run
    
    Returns:
        List of (index, delta_f1) tuples sorted by most negative delta
    """

    # Split into train-val
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=random_state)

    # Subsample training set
    train_subset = X_train_text.sample(frac=sample_frac, random_state=random_state)
    y_subset = y_train.loc[train_subset.index]
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(train_subset)
    X_val_vec = vectorizer.transform(X_val_text)

    # Train base model
    base_model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
    base_model.fit(X_train_vec, y_subset)
    base_preds = base_model.predict(X_val_vec)
    base_f1 = f1_score(y_val, base_preds, average='macro', zero_division=0)
    print(f"📌 Base macro F1: {base_f1:.4f}")

    # Estimate influence via leave-one-out
    influences = []
    indices = train_subset.index.tolist()
    texts = train_subset.tolist()

    print(f"\n🧪 Running leave-one-out influence on {len(indices)} samples...\n")

    if max_examples:
        indices = indices[:max_examples]
        texts = texts[:max_examples]

    for idx, text in tqdm(zip(indices, texts), total=len(indices)):
        mask = train_subset.index != idx
        X_loo = train_subset[mask]
        y_loo = y_subset[mask]

        X_loo_vec = vectorizer.fit_transform(X_loo)
        X_val_vec = vectorizer.transform(X_val_text)

        model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
        model.fit(X_loo_vec, y_loo)

        preds = model.predict(X_val_vec)
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)

        delta = f1 - base_f1
        influences.append((idx, delta))

    # Sort: most harmful first
    influences.sort(key=lambda x: x[1])
    return influences
