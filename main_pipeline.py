"""
Main Pipeline - Complete ML Workflow
Trains LightGBM, XGBoost, and CatBoost models
Then creates ensemble predictions
Shows all scores and comparisons
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import time
import warnings
warnings.filterwarnings('ignore')

# Import models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

print("=" * 80)
print("COMPLETE ML PIPELINE - HEART DISEASE PREDICTION")
print("=" * 80)
print("\nThis pipeline will:")
print("  1. Train LightGBM model")
print("  2. Train XGBoost model")
print("  3. Train CatBoost model")
print("  4. Create ensemble predictions")
print("  5. Compare all models")
print("\nEstimated time: 3-5 minutes")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING AND PREPARING DATA")
print("=" * 80)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"[OK] Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"[OK] Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Prepare features and target
X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Keep test IDs
test_ids = test_df['id']
X_test = test_df.drop(['id'], axis=1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ============================================================================
# STEP 2: TRAIN LIGHTGBM
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: TRAINING LIGHTGBM MODEL")
print("=" * 80)

start_time = time.time()

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[val_data],
    valid_names=['valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)  # Silent
    ]
)

lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

lgb_auc = roc_auc_score(y_val, lgb_val_pred)
lgb_logloss = log_loss(y_val, lgb_val_pred)
lgb_time = time.time() - start_time

print(f"[OK] LightGBM trained in {lgb_time:.1f} seconds")
print(f"     Validation ROC AUC: {lgb_auc:.4f}")
print(f"     Validation Log Loss: {lgb_logloss:.4f}")
print(f"     Best iteration: {lgb_model.best_iteration}")

# ============================================================================
# STEP 3: TRAIN XGBOOST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAINING XGBOOST MODEL")
print("=" * 80)

start_time = time.time()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_val, xgb_val_pred)
xgb_logloss = log_loss(y_val, xgb_val_pred)
xgb_time = time.time() - start_time

print(f"[OK] XGBoost trained in {xgb_time:.1f} seconds")
print(f"     Validation ROC AUC: {xgb_auc:.4f}")
print(f"     Validation Log Loss: {xgb_logloss:.4f}")
print(f"     Trees used: {xgb_model.n_estimators}")

# ============================================================================
# STEP 4: TRAIN CATBOOST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAINING CATBOOST MODEL")
print("=" * 80)

start_time = time.time()

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'bagging_temperature': 1.0,
    'random_strength': 1.0,
    'subsample': 0.8,
    'rsm': 0.8,
    'random_seed': 42,
    'thread_count': -1,
    'verbose': 0,
    'early_stopping_rounds': 50,
    'use_best_model': True
}

cat_model = CatBoostClassifier(**cat_params)
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

cat_model.fit(train_pool, eval_set=val_pool, verbose=False)

cat_val_pred = cat_model.predict_proba(X_val)[:, 1]
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]

cat_auc = roc_auc_score(y_val, cat_val_pred)
cat_logloss = log_loss(y_val, cat_val_pred)
cat_time = time.time() - start_time

print(f"[OK] CatBoost trained in {cat_time:.1f} seconds")
print(f"     Validation ROC AUC: {cat_auc:.4f}")
print(f"     Validation Log Loss: {cat_logloss:.4f}")
print(f"     Best iteration: {cat_model.best_iteration_}")

# ============================================================================
# STEP 5: CREATE ENSEMBLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CREATING ENSEMBLE PREDICTIONS")
print("=" * 80)

# Simple average
ensemble_val_simple = (lgb_val_pred + xgb_val_pred + cat_val_pred) / 3
ensemble_test_simple = (lgb_test_pred + xgb_test_pred + cat_test_pred) / 3

# Weighted average (based on validation performance)
weights = np.array([lgb_auc, xgb_auc, cat_auc])
weights = weights / weights.sum()

ensemble_val_weighted = (
    weights[0] * lgb_val_pred + 
    weights[1] * xgb_val_pred + 
    weights[2] * cat_val_pred
)
ensemble_test_weighted = (
    weights[0] * lgb_test_pred + 
    weights[1] * xgb_test_pred + 
    weights[2] * cat_test_pred
)

# Calculate ensemble scores
ensemble_simple_auc = roc_auc_score(y_val, ensemble_val_simple)
ensemble_simple_logloss = log_loss(y_val, ensemble_val_simple)

ensemble_weighted_auc = roc_auc_score(y_val, ensemble_val_weighted)
ensemble_weighted_logloss = log_loss(y_val, ensemble_val_weighted)

print(f"[OK] Simple Ensemble (equal weights)")
print(f"     Validation ROC AUC: {ensemble_simple_auc:.4f}")
print(f"     Validation Log Loss: {ensemble_simple_logloss:.4f}")

print(f"\n[OK] Weighted Ensemble (performance-based weights)")
print(f"     Weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
print(f"     Validation ROC AUC: {ensemble_weighted_auc:.4f}")
print(f"     Validation Log Loss: {ensemble_weighted_logloss:.4f}")

# ============================================================================
# STEP 6: SAVE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: SAVING PREDICTIONS")
print("=" * 80)

# Individual models
pd.DataFrame({'id': test_ids, 'Heart Disease': lgb_test_pred}).to_csv(
    'submission_lightgbm.csv', index=False)
print("[OK] submission_lightgbm.csv")

pd.DataFrame({'id': test_ids, 'Heart Disease': xgb_test_pred}).to_csv(
    'submission_xgboost.csv', index=False)
print("[OK] submission_xgboost.csv")

pd.DataFrame({'id': test_ids, 'Heart Disease': cat_test_pred}).to_csv(
    'submission_catboost.csv', index=False)
print("[OK] submission_catboost.csv")

# Ensemble models
pd.DataFrame({'id': test_ids, 'Heart Disease': ensemble_test_simple}).to_csv(
    'submission_ensemble_simple.csv', index=False)
print("[OK] submission_ensemble_simple.csv")

pd.DataFrame({'id': test_ids, 'Heart Disease': ensemble_test_weighted}).to_csv(
    'submission_ensemble_weighted.csv', index=False)
print("[OK] submission_ensemble_weighted.csv")

# ============================================================================
# STEP 7: FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL COMPARISON SUMMARY")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble (Simple)', 'Ensemble (Weighted)'],
    'ROC AUC': [lgb_auc, xgb_auc, cat_auc, ensemble_simple_auc, ensemble_weighted_auc],
    'Log Loss': [lgb_logloss, xgb_logloss, cat_logloss, ensemble_simple_logloss, ensemble_weighted_logloss],
    'Training Time (s)': [lgb_time, xgb_time, cat_time, 0, 0]
})

results = results.sort_values('ROC AUC', ascending=False)
print("\n" + results.to_string(index=False))

# Find best model
best_model = results.iloc[0]
print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)
print(f"Model: {best_model['Model']}")
print(f"Validation ROC AUC: {best_model['ROC AUC']:.4f}")
print(f"Validation Log Loss: {best_model['Log Loss']:.4f}")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1. SUBMIT TO KAGGLE:")
if 'Ensemble' in best_model['Model']:
    print(f"   [BEST] submission_ensemble_weighted.csv (Best: {best_model['ROC AUC']:.4f})")
    print(f"   Also try: submission_ensemble_simple.csv")
else:
    print(f"   [BEST] submission_{best_model['Model'].lower()}.csv (Best: {best_model['ROC AUC']:.4f})")
    print(f"   Also try: submission_ensemble_weighted.csv")

print("\n2. EXPECTED PERFORMANCE:")
print(f"   - Validation score: {best_model['ROC AUC']:.4f}")
print(f"   - Expected Kaggle: {best_model['ROC AUC']-0.001:.4f} - {best_model['ROC AUC']+0.001:.4f}")
print(f"   - Target (0.95332): {'BEAT!' if best_model['ROC AUC'] > 0.95332 else 'Close'}")

print("\n3. IMPROVEMENT GAINED:")
single_best = max(lgb_auc, xgb_auc, cat_auc)
ensemble_best = max(ensemble_simple_auc, ensemble_weighted_auc)
improvement = ensemble_best - single_best
print(f"   - Best single model: {single_best:.4f}")
print(f"   - Best ensemble: {ensemble_best:.4f}")
print(f"   - Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")

print("\n4. NEXT STEPS:")
print("   - Submit best model to Kaggle")
print("   - Try hyperparameter tuning (see HYPERPARAMETER_TUNING_GUIDE.md)")
print("   - Add feature engineering")
print("   - Try stacking (meta-model)")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print("\nAll submission files are ready in the current directory.")
print("Upload the recommended file to Kaggle and check your score!")
print("=" * 80)
