"""
Model Comparison Script
Compares different models side-by-side
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 70)
print("MODEL COMPARISON FOR HEART DISEASE PREDICTION")
print("=" * 70)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('data/train.csv')
X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data prepared: {len(X_train)} training, {len(X_val)} validation samples")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, 
                                           random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                    random_state=42)
}

print("\n" + "=" * 70)
print("TRAINING AND EVALUATING MODELS...")
print("=" * 70)

results = []

for name, model in models.items():
    print(f"\n[{name}]")
    print("-" * 70)
    
    # Train
    print("  Training...", end=" ")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Done in {train_time:.2f} seconds")
    
    # Predict
    print("  Predicting...", end=" ")
    y_pred = model.predict_proba(X_val)[:, 1]
    print("Done")
    
    # Evaluate
    roc_auc = roc_auc_score(y_val, y_pred)
    print(f"  ROC AUC Score: {roc_auc:.4f}")
    
    results.append({
        'Model': name,
        'ROC AUC': roc_auc,
        'Training Time (s)': train_time
    })

# Display comparison
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

best_model = results_df.iloc[0]
fastest_model = results_df.sort_values('Training Time (s)').iloc[0]

print(f"\nBest Accuracy: {best_model['Model']}")
print(f"  - ROC AUC: {best_model['ROC AUC']:.4f}")
print(f"  - Training Time: {best_model['Training Time (s)']:.2f}s")

print(f"\nFastest Model: {fastest_model['Model']}")
print(f"  - ROC AUC: {fastest_model['ROC AUC']:.4f}")
print(f"  - Training Time: {fastest_model['Training Time (s)']:.2f}s")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("\n1. Use the best model for your submission")
print("2. Try hyperparameter tuning to improve further")
print("3. Consider ensemble methods (combining models)")
print("4. Add feature engineering for even better results")
print("\n" + "=" * 70)
