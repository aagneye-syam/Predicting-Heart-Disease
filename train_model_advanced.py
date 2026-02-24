"""
Heart Disease Prediction - Advanced Model
This script uses Random Forest for better predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HEART DISEASE PREDICTION - ADVANCED MODEL")
print("=" * 60)

# Step 1: Load the data
print("\n[Step 1] Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"[OK] Training data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"[OK] Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Step 2: Prepare the data
print("\n[Step 2] Preparing data for training...")

# Separate features and target
X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']

# Convert target to binary
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(f"[OK] Target encoded: {label_encoder.classes_}")

# Keep test IDs for submission
test_ids = test_df['id']
X_test = test_df.drop(['id'], axis=1)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Data split: {len(X_train)} training, {len(X_val)} validation samples")

# Step 3: Train Random Forest model
print("\n[Step 3] Training Random Forest model...")
print("Random Forest uses multiple decision trees for better accuracy")
print("This may take a few minutes...")

# Create Random Forest model with good parameters
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=20,            # Maximum depth of each tree
    min_samples_split=10,    # Minimum samples to split a node
    min_samples_leaf=5,      # Minimum samples in a leaf
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbose=1                # Show progress
)

model.fit(X_train, y_train)
print("[OK] Model training complete!")

# Step 4: Evaluate the model
print("\n[Step 4] Evaluating model performance...")

# Make predictions on validation set
y_val_pred = model.predict_proba(X_val)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"[OK] Validation ROC AUC Score: {roc_auc:.4f}")
print("  (Higher is better, max is 1.0)")

# Show feature importance
print("\n[Step 5] Feature Importance (Top 5):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Step 6: Make predictions on test data
print("\n[Step 6] Making predictions on test data...")
test_predictions = model.predict_proba(X_test)[:, 1]
print(f"[OK] Generated {len(test_predictions)} predictions")

# Step 7: Create submission file
print("\n[Step 7] Creating submission file...")
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': test_predictions
})
submission.to_csv('submission_rf.csv', index=False)
print("[OK] Submission file saved as 'submission_rf.csv'")

# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))

print("\n" + "=" * 60)
print("SUCCESS! Advanced model is ready!")
print("=" * 60)
print("\nModel Performance:")
print(f"- Validation ROC AUC: {roc_auc:.4f}")
print(f"- Model Type: Random Forest with {model.n_estimators} trees")
print("\nSubmission file: submission_rf.csv")
print("\nCompare this with the basic model to see improvement!")
print("=" * 60)
