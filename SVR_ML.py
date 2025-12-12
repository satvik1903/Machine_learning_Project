import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')
from sklearn.model_selection import (train_test_split, cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import uniform, loguniform
import joblib
import time

BASE_PATH = '/Users/satviktajne/Desktop/Sem 1/ML and PR/ML Project/CODE/dataset/Final Datasets'
TRAIN_PATH = f'{BASE_PATH}/train_features_v2.csv'
OUTPUT_DIR = f'{BASE_PATH}/SVR_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

PERFORM_GRID_SEARCH = True     
USE_RANDOMIZED_SEARCH = True   
USE_QUICK_GRID = True          
CV_FOLDS_GRID = 3             
N_ITER_RANDOM = 30             
USE_SUBSET_FOR_GRIDSEARCH = True
SUBSET_SIZE = 10000            
USE_LOG_TRANSFORM = True       
REMOVE_OUTLIERS = True         
OUTLIER_PERCENTILE = 99
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

numerical_features = [
    'unit_size', 'size_in_oz', 'pack_size', 'total_size',
    'has_premium_word', 'premium_word_count',
    'has_health_word', 'health_word_count',
    'has_ingredients', 'ingredient_word_count'
]

param_distributions = {
    'C': loguniform(1e-1, 1e2),           
    'epsilon': uniform(0.05, 0.5),       
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
}

SVR_HYPERPARAMETERS = {
    'C': 50, 'epsilon': 0.1, 'kernel': 'rbf', 'gamma': 'scale',
    'cache_size': 1000, 'max_iter': 100000, 'tol': 1e-3,                
}

def evaluate_regression(y_true, y_pred, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    valid_mask = denominator != 0
    smape = np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]) / denominator[valid_mask]) * 100
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Adj_R2': adj_r2, 'MAPE': mape, 'SMAPE': smape}

print("Loading data...")
df_train = pd.read_csv(TRAIN_PATH)

encoders = {}
if 'category' in df_train.columns:
    encoders['category'] = LabelEncoder()
    df_train['category_encoded'] = encoders['category'].fit_transform(df_train['category'].fillna('Unknown'))
if 'unit_type' in df_train.columns:
    encoders['unit_type'] = LabelEncoder()
    df_train['unit_type_encoded'] = encoders['unit_type'].fit_transform(df_train['unit_type'].fillna('Unknown'))
if 'brand_popularity' in df_train.columns:
    popularity_map = {'low': 0, 'mid': 1, 'high': 2}
    df_train['brand_popularity_encoded'] = df_train['brand_popularity'].map(popularity_map).fillna(0)

all_features = numerical_features + ['category_encoded', 'unit_type_encoded', 'brand_popularity_encoded']

X = df_train[all_features].copy().fillna(0)
y = df_train['item_price'].copy().fillna(df_train['item_price'].median())
valid_mask = y > 0
X, y = X[valid_mask], y[valid_mask]

if REMOVE_OUTLIERS:
    price_threshold = y.quantile(OUTLIER_PERCENTILE / 100)
    outlier_mask = y <= price_threshold
    X, y = X[outlier_mask], y[outlier_mask]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

if USE_LOG_TRANSFORM:
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_log.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val_log.values.reshape(-1, 1)).ravel()
else:
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

if PERFORM_GRID_SEARCH:
    print("Running hyperparameter search...")
    if USE_SUBSET_FOR_GRIDSEARCH and len(X_train_scaled) > SUBSET_SIZE:
        np.random.seed(RANDOM_STATE)
        subset_idx = np.random.choice(len(X_train_scaled), SUBSET_SIZE, replace=False)
        X_grid, y_grid = X_train_scaled[subset_idx], y_train_scaled[subset_idx]
    else:
        X_grid, y_grid = X_train_scaled, y_train_scaled
    
    start_time_grid = time.time()
    svr_base = SVR(cache_size=1000, max_iter=100000, tol=1e-3)
    
    grid_search = RandomizedSearchCV(
        estimator=svr_base, param_distributions=param_distributions,
        n_iter=N_ITER_RANDOM, cv=CV_FOLDS_GRID, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0, return_train_score=True
    )
    grid_search.fit(X_grid, y_grid)
    grid_time = time.time() - start_time_grid
    
    SVR_HYPERPARAMETERS.update(grid_search.best_params_)
    SVR_HYPERPARAMETERS['cache_size'] = 1000
    SVR_HYPERPARAMETERS['max_iter'] = 100000
    SVR_HYPERPARAMETERS['tol'] = 1e-3
    
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['rmse'] = np.sqrt(-cv_results_df['mean_test_score'])
    cv_results_df.sort_values('rank_test_score').to_csv(f'{OUTPUT_DIR}/grid_search_results.csv', index=False)
else:
    grid_time = 0

print("Training SVR model...")
start_time = time.time()
svr_model = SVR(**SVR_HYPERPARAMETERS)
svr_model.fit(X_train_scaled, y_train_scaled)
train_time = time.time() - start_time
n_support = len(svr_model.support_)

y_train_pred_scaled = svr_model.predict(X_train_scaled)
y_val_pred_scaled = svr_model.predict(X_val_scaled)

if USE_LOG_TRANSFORM:
    y_train_pred = np.maximum(np.expm1(scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()), 0)
    y_val_pred = np.maximum(np.expm1(scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()), 0)
else:
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()

train_metrics = evaluate_regression(y_train.values, y_train_pred, len(all_features))
val_metrics = evaluate_regression(y_val.values, y_val_pred, len(all_features))

r2_gap = train_metrics['R2'] - val_metrics['R2']
if r2_gap > 0.15: fit_status = "OVERFITTING"
elif r2_gap > 0.05: fit_status = "MODERATE_OVERFITTING"
elif val_metrics['R2'] < 0.3: fit_status = "UNDERFITTING"
else: fit_status = "GOOD_FIT"

print("Running cross-validation...")
if len(X_train_scaled) > 20000:
    cv_idx = np.random.choice(len(X_train_scaled), 20000, replace=False)
    X_cv, y_cv = X_train_scaled[cv_idx], y_train_scaled[cv_idx]
else:
    X_cv, y_cv = X_train_scaled, y_train_scaled

cv_r2_scores = cross_val_score(SVR(**SVR_HYPERPARAMETERS), X_cv, y_cv, cv=CV_FOLDS, scoring='r2', n_jobs=-1)

print("Generating visualizations...")
residuals = y_val.values - y_val_pred
price_limit = y_val.quantile(0.95)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=10, c='steelblue', label='Predictions')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($)'); axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title(f'Training Set - R2 = {train_metrics["R2"]:.4f}'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].scatter(y_val, y_val_pred, alpha=0.3, s=10, c='coral', label='Predictions')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price ($)'); axes[1].set_ylabel('Predicted Price ($)')
axes[1].set_title(f'Validation Set - R2 = {val_metrics["R2"]:.4f}'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/01_predicted_vs_actual.png', dpi=300, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].scatter(y_val_pred, residuals, alpha=0.3, s=10, c='steelblue')
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Price ($)'); axes[0, 0].set_ylabel('Residual ($)')
axes[0, 0].set_title('Residuals vs Predicted'); axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Residual ($)'); axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residuals Distribution')
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot'); axes[1, 0].grid(True, alpha=0.3)
axes[1, 1].scatter(y_val, residuals, alpha=0.3, s=10, c='coral')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Actual Price ($)'); axes[1, 1].set_ylabel('Residual ($)')
axes[1, 1].set_title('Residuals vs Actual'); axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/02_residuals_analysis.png', dpi=300, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics_names = ['MAE', 'RMSE', 'R2']
train_vals = [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['R2']]
val_vals = [val_metrics['MAE'], val_metrics['RMSE'], val_metrics['R2']]
x = np.arange(len(metrics_names)); width = 0.35
axes[0].bar(x - width/2, train_vals, width, label='Training', color='steelblue')
axes[0].bar(x + width/2, val_vals, width, label='Validation', color='coral')
axes[0].set_ylabel('Value'); axes[0].set_title('Training vs Validation Metrics')
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics_names); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')
fold_nums = np.arange(1, CV_FOLDS + 1)
axes[1].bar(fold_nums, cv_r2_scores, color='forestgreen', edgecolor='black', alpha=0.7)
axes[1].axhline(y=cv_r2_scores.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {cv_r2_scores.mean():.4f}')
axes[1].set_xlabel('Fold'); axes[1].set_ylabel('R2 Score'); axes[1].set_title('Cross-Validation R2 Scores')
axes[1].set_xticks(fold_nums); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/03_metrics_comparison.png', dpi=300, bbox_inches='tight'); plt.close()

feature_correlations = [(f, abs(np.corrcoef(X_val.iloc[:, i], y_val_pred)[0, 1])) for i, f in enumerate(all_features)]
feature_correlations.sort(key=lambda x: x[1], reverse=True)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh([f[0] for f in feature_correlations], [f[1] for f in feature_correlations], 
        color=['green' if c[1] > 0.1 else 'gray' for c in feature_correlations], edgecolor='black')
ax.set_xlabel('Absolute Correlation with Predictions'); ax.set_ylabel('Feature')
ax.set_title('Feature Importance'); ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/04_feature_importance.png', dpi=300, bbox_inches='tight'); plt.close()

print("Saving model and results...")
joblib.dump(svr_model, f'{OUTPUT_DIR}/svr_model.pkl')
joblib.dump(scaler_X, f'{OUTPUT_DIR}/scaler_X.pkl')
joblib.dump(scaler_y, f'{OUTPUT_DIR}/scaler_y.pkl')
joblib.dump(encoders, f'{OUTPUT_DIR}/encoders.pkl')
joblib.dump({
    'USE_LOG_TRANSFORM': USE_LOG_TRANSFORM, 'REMOVE_OUTLIERS': REMOVE_OUTLIERS,
    'OUTLIER_PERCENTILE': OUTLIER_PERCENTILE, 'features': all_features
}, f'{OUTPUT_DIR}/model_config.pkl')

pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2', 'Adj_R2', 'MAPE', 'SMAPE'],
    'Training': [train_metrics['MAE'], train_metrics['MSE'], train_metrics['RMSE'],
                 train_metrics['R2'], train_metrics['Adj_R2'], train_metrics['MAPE'], train_metrics['SMAPE']],
    'Validation': [val_metrics['MAE'], val_metrics['MSE'], val_metrics['RMSE'],
                   val_metrics['R2'], val_metrics['Adj_R2'], val_metrics['MAPE'], val_metrics['SMAPE']]
}).to_csv(f'{OUTPUT_DIR}/svr_metrics.csv', index=False)

pd.DataFrame({
    'Parameter': list(SVR_HYPERPARAMETERS.keys()) + ['TEST_SIZE', 'RANDOM_STATE', 'CV_FOLDS', 
                 'GRID_SEARCH_PERFORMED', 'USE_LOG_TRANSFORM', 'REMOVE_OUTLIERS'],
    'Value': [str(v) for v in SVR_HYPERPARAMETERS.values()] + [str(TEST_SIZE), str(RANDOM_STATE), 
              str(CV_FOLDS), str(PERFORM_GRID_SEARCH), str(USE_LOG_TRANSFORM), str(REMOVE_OUTLIERS)]
}).to_csv(f'{OUTPUT_DIR}/svr_hyperparameters.csv', index=False)

pd.DataFrame({'Feature': all_features}).to_csv(f'{OUTPUT_DIR}/features_used.csv', index=False)

print(f"""
SVR MODEL TRAINING COMPLETE
---------------------------
Validation Metrics:
  MAE:   ${val_metrics['MAE']:.2f}
  RMSE:  ${val_metrics['RMSE']:.2f}
  R2:    {val_metrics['R2']:.4f} ({val_metrics['R2']*100:.1f}% variance explained)
  MAPE:  {val_metrics['MAPE']:.2f}%
  SMAPE: {val_metrics['SMAPE']:.2f}%

Cross-Validation: Mean R2 = {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std()*2:.4f})
Fit Status: {fit_status}
Support Vectors: {n_support} ({n_support/len(X_train)*100:.1f}% of training data)
Training Time: {train_time:.2f}s | Grid Search Time: {grid_time:.1f}s

Output saved to: {OUTPUT_DIR}
""")