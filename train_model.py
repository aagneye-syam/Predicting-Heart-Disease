"""
Heart Disease Prediction - Beginner-Friendly Script
This script trains a machine learning model to predict heart disease
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HEART DISEASE PREDICTION MODEL")
print("=" * 60)

# Step 1: Load the data
print("\n[Step 1] Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"[OK] Training data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"[OK] Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Step 2: Explore the data
print("\n[Step 2] Exploring data...")
print(f"Target variable distribution:")
print(train_df['Heart Disease'].value_counts())
print(f"\nMissing values in training data: {train_df.isnull().sum().sum()}")
print(f"Missing values in test data: {test_df.isnull().sum().sum()}")

# Step 3: Prepare the data
print("\n[Step 3] Preparing data for training...")

# Separate features (X) and target (y)
X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']

# Convert target to binary (0 = Absence, 1 = Presence)
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

# Step 4: Train the model
print("\n[Step 4] Training the model...")
print("Using Logistic Regression (a simple but effective model)")

# Create and train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("[OK] Model training complete!")

# Step 5: Evaluate the model
print("\n[Step 5] Evaluating model performance...")

# Make predictions on validation set
y_val_pred = model.predict_proba(X_val)[:, 1]  # Get probability of class 1

# Calculate ROC AUC score (the competition metric)
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"[OK] Validation ROC AUC Score: {roc_auc:.4f}")
print("  (Higher is better, max is 1.0)")

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
submission.to_csv('submission.csv', index=False)
print("[OK] Submission file saved as 'submission.csv'")

# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))

print("\n" + "=" * 60)
print("SUCCESS! Your model is ready for submission!")
print("=" * 60)
print("\nNext steps:")
print("1. Check the 'submission.csv' file")
print("2. Upload it to Kaggle competition page")
print("3. Your validation score was: {:.4f}".format(roc_auc))
print("\nTo improve your model, you can:")
print("- Try different models (Random Forest, XGBoost)")
print("- Add feature engineering")
print("- Tune hyperparameters")
print("=" * 60)
