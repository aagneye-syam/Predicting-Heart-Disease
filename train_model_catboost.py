"""
Heart Disease Prediction - CatBoost Model
CatBoost is a gradient boosting library developed by Yandex
Known for handling categorical features well and robust performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HEART DISEASE PREDICTION - CATBOOST MODEL")
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

# Step 3: Configure CatBoost
print("\n[Step 3] Configuring CatBoost model...")
print("CatBoost Parameters:")

params = {
    # Objective function
    'loss_function': 'Logloss',      # Binary log loss
    'eval_metric': 'AUC',            # Track ROC AUC during training
    
    # Model complexity
    'iterations': 500,               # Number of boosting rounds
    'learning_rate': 0.05,           # Step size
    'depth': 6,                      # Tree depth
    
    # Regularization
    'l2_leaf_reg': 3.0,              # L2 regularization
    'bagging_temperature': 1.0,      # Bayesian bootstrap intensity
    'random_strength': 1.0,          # Randomness for scoring splits
    
    # Sampling
    'subsample': 0.8,                # Fraction of data per tree
    'rsm': 0.8,                      # Random subspace method (feature sampling)
    
    # Training settings
    'random_seed': 42,
    'thread_count': -1,              # Use all CPU cores
    'verbose': 100,                  # Print every 100 rounds
    'early_stopping_rounds': 50,     # Stop if no improvement
    'use_best_model': True           # Use best iteration
}

print(f"  - Loss Function: {params['loss_function']}")
print(f"  - Eval Metric: {params['eval_metric']}")
print(f"  - Learning Rate: {params['learning_rate']}")
print(f"  - Depth: {params['depth']}")
print(f"  - Iterations: {params['iterations']}")

# Step 4: Train the model
print("\n[Step 4] Training CatBoost model...")
print("This may take 1-3 minutes...")

# Create CatBoost model
model = CatBoostClassifier(**params)

# Create Pool objects for better performance
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

# Train with validation
model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=100
)

print(f"[OK] Model training complete!")
print(f"[OK] Best iteration: {model.best_iteration_}")

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
submission.to_csv('submission_catboost.csv', index=False)
print("[OK] Submission file saved as 'submission_catboost.csv'")

# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))

# Step 9: Summary
print("\n" + "=" * 70)
print("SUCCESS! CATBOOST MODEL READY!")
print("=" * 70)
print("\nModel Performance Summary:")
print(f"  - Validation ROC AUC: {roc_auc:.4f}")
print(f"  - Validation Log Loss: {logloss:.4f}")
print(f"  - Best Iteration: {model.best_iteration_}")
print(f"  - Trees Used: {model.best_iteration_ + 1}")
print("\nSubmission file: submission_catboost.csv")
print("\nCatBoost Advantages:")
print("  - Robust to overfitting")
print("  - Handles categorical features automatically")
print("  - Ordered boosting (reduces overfitting)")
print("  - Great for ensembling with LightGBM and XGBoost")
print("=" * 70)
