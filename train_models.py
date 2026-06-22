import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import datetime

# Paths
clean_path = os.path.join('data', 'processed', 'cleaned_data.csv')
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
paths = {
    'scaler': os.path.join(models_dir, 'scaler.pkl'),
    'rf_orig': os.path.join(models_dir, 'random_forest.pkl'),
    'xgb_orig': os.path.join(models_dir, 'xgboost.pkl'),
    'tuned_rf': os.path.join(models_dir, 'tuned_random_forest.pkl'),
    'tuned_xgb': os.path.join(models_dir, 'tuned_xgboost.pkl'),
    'best': os.path.join(models_dir, 'best_model.pkl')
}

# Load data
if not os.path.exists(clean_path):
    raise FileNotFoundError(f'Cleaned dataset not found at {clean_path}. Run preprocessing first.')
df = pd.read_csv(clean_path)
print('Loaded cleaned data shape:', df.shape)

# Ensure pulse_pressure
if 'pulse_pressure' not in df.columns and 'ap_hi' in df.columns and 'ap_lo' in df.columns:
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Features and target
feature_columns = [
    'age_years', 'gender', 'height', 'weight', 'BMI',
    'ap_hi', 'ap_lo', 'pulse_pressure', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]
missing = [c for c in feature_columns + ['cardio'] if c not in df.columns]
if missing:
    raise KeyError(f'Missing required columns: {missing}')
X = df[feature_columns].copy()
y = df['cardio'].copy()

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train/test split:', X_train.shape, X_test.shape)

# Scale
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, paths['scaler'])
print('Saved scaler to', paths['scaler'])

# Helper eval
def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} - Acc:{acc:.4f} Prec:{pre:.4f} Rec:{rec:.4f} F1:{f1:.4f}")
    print('Confusion Matrix:\n', cm)
    return {'Model': name, 'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Confusion Matrix': cm}

results = []

# Baseline Random Forest
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_orig.fit(X_train_scaled, y_train)
rf_orig_pred = rf_orig.predict(X_test_scaled)
results.append(evaluate('Random Forest (Original)', y_test, rf_orig_pred))
joblib.dump(rf_orig, paths['rf_orig'])
print('Saved original RF to', paths['rf_orig'])

# Baseline XGBoost
xgb_orig = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
xgb_orig.fit(X_train_scaled, y_train)
xgb_orig_pred = xgb_orig.predict(X_test_scaled)
results.append(evaluate('XGBoost (Original)', y_test, xgb_orig_pred))
joblib.dump(xgb_orig, paths['xgb_orig'])
print('Saved original XGB to', paths['xgb_orig'])

# CV splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV for RF
print('\nStarting RandomizedSearchCV for Random Forest...')
rf_param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=15,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_search.fit(X_train_scaled, y_train)
print('RF best params:', rf_search.best_params_, 'best_score:', rf_search.best_score_)
rf_tuned = rf_search.best_estimator_
rf_tuned_pred = rf_tuned.predict(X_test_scaled)
rf_tuned_results = evaluate('Random Forest (Tuned)', y_test, rf_tuned_pred)
results.append(rf_tuned_results)
joblib.dump(rf_tuned, paths['tuned_rf'])
print('Saved tuned RF to', paths['tuned_rf'])

# RandomizedSearchCV for XGB
print('\nStarting RandomizedSearchCV for XGBoost...')
xgb_param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3,4,5,6,7,8,10],
    'learning_rate': [0.01,0.05,0.1,0.15,0.2],
    'subsample':[0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree':[0.6,0.7,0.8,0.9,1.0],
    'min_child_weight':[1,3,5,7]
}
xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0),
    param_distributions=xgb_param_dist,
    n_iter=15,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_search.fit(X_train_scaled, y_train)
print('XGB best params:', xgb_search.best_params_, 'best_score:', xgb_search.best_score_)
xgb_tuned = xgb_search.best_estimator_
xgb_tuned_pred = xgb_tuned.predict(X_test_scaled)
xgb_tuned_results = evaluate('XGBoost (Tuned)', y_test, xgb_tuned_pred)
results.append(xgb_tuned_results)
joblib.dump(xgb_tuned, paths['tuned_xgb'])
print('Saved tuned XGB to', paths['tuned_xgb'])

# Comparison and best model selection
import pandas as pd
comparison_df = pd.DataFrame(results)
print('\nComparison:')
print(comparison_df[['Model','Accuracy','Precision','Recall','F1 Score']])
comparison_csv_path = os.path.join(results_dir, 'model_comparison.csv')
comparison_df.to_csv(comparison_csv_path, index=False)
print('Saved comparison CSV to', comparison_csv_path)
best_idx = comparison_df['F1 Score'].idxmax()
best_name = comparison_df.loc[best_idx,'Model']
# map name to object
if 'Random Forest (Tuned)' in comparison_df['Model'].values:
    best_obj = rf_tuned if comparison_df.loc[comparison_df['Model']=='Random Forest (Tuned)','F1 Score'].values[0] >= comparison_df.loc[comparison_df['Model']=='XGBoost (Tuned)','F1 Score'].values[0] else xgb_tuned
else:
    best_obj = xgb_orig if comparison_df.loc[comparison_df['Model']=='XGBoost (Original)','Accuracy'].values[0] >= comparison_df.loc[comparison_df['Model']=='Random Forest (Original)','Accuracy'].values[0] else rf_orig
joblib.dump(best_obj, paths['best'])
print('Saved best model to', paths['best'])

# Final summary prints required by user
orig_rf_acc = comparison_df[comparison_df['Model']=='Random Forest (Original)']['Accuracy'].values[0]
orig_xgb_acc = comparison_df[comparison_df['Model']=='XGBoost (Original)']['Accuracy'].values[0]
tuned_rf_acc = comparison_df[comparison_df['Model']=='Random Forest (Tuned)']['Accuracy'].values[0]
tuned_xgb_acc = comparison_df[comparison_df['Model']=='XGBoost (Tuned)']['Accuracy'].values[0]

print('\nOriginal Model Results:')
print('- Random Forest Accuracy', f'{orig_rf_acc:.4f}')
print('- XGBoost Accuracy', f'{orig_xgb_acc:.4f}')

print('\nTuned Model Results:')
print('- Tuned Random Forest Accuracy', f'{tuned_rf_acc:.4f}')
print('- Tuned XGBoost Accuracy', f'{tuned_xgb_acc:.4f}')

print('\nImprovement:')
print('- Accuracy Difference', f'{tuned_rf_acc - orig_rf_acc:+.4f}', '(RF),', f'{tuned_xgb_acc - orig_xgb_acc:+.4f}', '(XGB)')
print('- F1 Difference', f"{comparison_df.loc[comparison_df['Model']=='Random Forest (Tuned)','F1 Score'].values[0] - comparison_df.loc[comparison_df['Model']=='Random Forest (Original)','F1 Score'].values[0]:+.4f}", '(RF)')

print('\nBest Model:')
print('- Model Name', best_name)
best_acc = comparison_df.loc[best_idx,'Accuracy']
best_prec = comparison_df.loc[best_idx,'Precision']
best_rec = comparison_df.loc[best_idx,'Recall']
best_f1 = comparison_df.loc[best_idx,'F1 Score']
print('- Accuracy', f'{best_acc:.4f}')
print('- Precision', f'{best_prec:.4f}')
print('- Recall', f'{best_rec:.4f}')
print('- F1 Score', f'{best_f1:.4f}')

# Top 10 important features
if hasattr(best_obj, 'feature_importances_'):
    fi = best_obj.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': fi}).sort_values('Importance', ascending=False)
    print('\nTop 10 Important Features')
    print(imp_df.head(10).to_string(index=False))
    imp_csv_path = os.path.join(results_dir, 'feature_importance.csv')
    imp_df.to_csv(imp_csv_path, index=False)
    print('Saved feature importance CSV to', imp_csv_path)
else:
    print('\nBest model does not provide feature_importances_.')

print('\nDone at', datetime.datetime.now().isoformat())
