"""
Heart Disease Prediction - XGBoost Model
XGBoost is another powerful gradient boosting framework
Known for winning many Kaggle competitions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HEART DISEASE PREDICTION - XGBOOST MODEL")
print("=" * 70)

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

# Step 3: Configure XGBoost
print("\n[Step 3] Configuring XGBoost model...")
print("XGBoost Parameters:")

params = {
    # Objective function
    'objective': 'binary:logistic',  # Binary classification with logistic output
    'eval_metric': 'auc',            # Track ROC AUC during training
    
    # Booster parameters
    'booster': 'gbtree',             # Tree-based model
    'tree_method': 'hist',           # Histogram-based algorithm (faster)
    
    # Model complexity
    'max_depth': 6,                  # Maximum tree depth
    'learning_rate': 0.05,           # Step size (eta)
    'n_estimators': 500,             # Number of boosting rounds
    
    # Regularization
    'min_child_weight': 5,           # Minimum sum of weights in a leaf
    'subsample': 0.8,                # Fraction of samples per tree
    'colsample_bytree': 0.8,         # Fraction of features per tree
    'gamma': 0.1,                    # Minimum loss reduction for split
    'reg_alpha': 0.1,                # L1 regularization
    'reg_lambda': 1.0,               # L2 regularization
    
    # Training settings
    'random_state': 42,
    'n_jobs': -1,                    # Use all CPU cores
    'verbosity': 1                   # Show progress
}

print(f"  - Objective: {params['objective']}")
print(f"  - Eval Metric: {params['eval_metric']}")
print(f"  - Learning Rate: {params['learning_rate']}")
print(f"  - Max Depth: {params['max_depth']}")
print(f"  - Number of Trees: {params['n_estimators']}")

# Step 4: Train the model
print("\n[Step 4] Training XGBoost model...")
print("This may take 1-3 minutes...")

# Create XGBoost model
model = xgb.XGBClassifier(**params)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100  # Print every 100 rounds
)

print(f"[OK] Model training complete!")
print(f"[OK] Best iteration: {model.best_iteration}")

# Step 5: Evaluate the model
print("\n[Step 5] Evaluating model performance...")

# Make predictions on validation set
y_val_pred = model.predict_proba(X_val)[:, 1]

# Calculate metrics
roc_auc = roc_auc_score(y_val, y_val_pred)
logloss = log_loss(y_val, y_val_pred)

print(f"[OK] Validation ROC AUC Score: {roc_auc:.4f}")
print(f"     (Competition metric - higher is better)")
print(f"[OK] Validation Log Loss: {logloss:.4f}")
print(f"     (Training objective - lower is better)")

# Step 6: Feature Importance
print("\n[Step 6] Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {idx+1}. {row['feature']:25} - Importance: {row['importance']:.4f}")

# Step 7: Make predictions on test data
print("\n[Step 7] Making predictions on test data...")
test_predictions = model.predict_proba(X_test)[:, 1]
print(f"[OK] Generated {len(test_predictions)} predictions")

# Show prediction distribution
print(f"\nPrediction Statistics:")
print(f"  - Mean probability: {test_predictions.mean():.4f}")
print(f"  - Min probability: {test_predictions.min():.4f}")
print(f"  - Max probability: {test_predictions.max():.4f}")

# Step 8: Create submission file
print("\n[Step 8] Creating submission file...")
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': test_predictions
})
submission.to_csv('submission_xgboost.csv', index=False)
print("[OK] Submission file saved as 'submission_xgboost.csv'")

# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))

# Step 9: Summary
print("\n" + "=" * 70)
print("SUCCESS! XGBOOST MODEL READY!")
print("=" * 70)
print("\nModel Performance Summary:")
print(f"  - Validation ROC AUC: {roc_auc:.4f}")
print(f"  - Validation Log Loss: {logloss:.4f}")
print(f"  - Best Iteration: {model.best_iteration}")
print(f"  - Trees Used: {model.best_iteration + 1}")
print("\nSubmission file: submission_xgboost.csv")
print("\nXGBoost vs LightGBM:")
print("  - XGBoost: More stable, better regularization")
print("  - LightGBM: Faster training, similar accuracy")
print("  - Best approach: Ensemble both!")
print("=" * 70)
