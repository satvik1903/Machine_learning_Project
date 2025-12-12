import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

input_path = '/Users/ayalabnine/Downloads/EECE5644/Datasets/Final Dataset/train_features_v2.csv'
output_dir = '/Users/ayalabnine/Downloads/EECE5644/SRC_EECE5644/XGBoost_Results/'
os.makedirs(output_dir, exist_ok=True)

def calculate_smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0.01
    if np.sum(mask) > 0:
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = 0.0
    return smape

def evaluate_model(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.maximum(np.array(y_pred), 0)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true > 0.01
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else 0.0
    smape = calculate_smape(y_true, y_pred)
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape, 'smape': smape}

print('xgboost regression - amazon price prediction')
print('loading data...')
df = pd.read_csv(input_path)
print(f'loaded {len(df)} rows')

target_col = 'item_price'
exclude_cols = ['item_price', 'item_name']
ordinal_col = 'brand_popularity'
onehot_cols = ['category', 'unit_type']

df_processed = df.copy()
brand_mapping = {'low': 0, 'mid': 1, 'high': 2}
df_processed['brand_popularity'] = df_processed['brand_popularity'].map(brand_mapping)
df_encoded = pd.get_dummies(df_processed, columns=onehot_cols, drop_first=True, dtype=int)

feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
X = df_encoded[feature_cols]
y = df_encoded[target_col]

print(f'features: {len(feature_cols)}, samples: {len(X)}')

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f'train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}')

baseline_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.3,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

print('\ntraining baseline model...')
start = time.time()
xgb_baseline = XGBRegressor(**baseline_params)
xgb_baseline.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
baseline_time = time.time() - start

y_val_pred_bl = xgb_baseline.predict(X_val)
bl_val_metrics = evaluate_model(y_val, y_val_pred_bl)

print(f'baseline validation - r2: {bl_val_metrics["r2"]:.4f}, rmse: {bl_val_metrics["rmse"]:.2f}, smape: {bl_val_metrics["smape"]:.2f}%')

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

total_combos = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
                len(param_grid['learning_rate']) * len(param_grid['subsample']) * 
                len(param_grid['colsample_bytree']))

print(f'\ntuning hyperparameters ({total_combos} combinations, 5-fold cv)...')
print('this may take 10-20 minutes...')

start = time.time()
grid_search = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)
tune_time = time.time() - start

best_params = grid_search.best_params_

print(f'tuning complete in {tune_time/60:.1f} minutes')
print(f'best params - trees: {best_params["n_estimators"]}, depth: {best_params["max_depth"]}, lr: {best_params["learning_rate"]}')
print(f'best cv rmse: {np.sqrt(-grid_search.best_score_):.2f}')

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['rmse'] = np.sqrt(-cv_results['mean_test_score'])
top10 = cv_results.sort_values('rmse').head(10)

print('\ntop 10 cross-validation results:')
for idx, row in top10.iterrows():
    print(f"rank {int(row['rank_test_score'])} - trees: {row['param_n_estimators']}, "
          f"depth: {row['param_max_depth']}, lr: {row['param_learning_rate']}, "
          f"rmse: {row['rmse']:.2f}")

print('\ntraining final model with best params...')
start = time.time()
xgb_tuned = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
xgb_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_time = time.time() - start

y_train_pred = xgb_tuned.predict(X_train)
y_val_pred = xgb_tuned.predict(X_val)
y_test_pred = xgb_tuned.predict(X_test)

train_metrics = evaluate_model(y_train, y_train_pred)
val_metrics = evaluate_model(y_val, y_val_pred)
test_metrics = evaluate_model(y_test, y_test_pred)

print(f'\ntrain metrics - r2: {train_metrics["r2"]:.4f}, rmse: {train_metrics["rmse"]:.2f}')
print(f'val metrics - r2: {val_metrics["r2"]:.4f}, rmse: {val_metrics["rmse"]:.2f}, smape: {val_metrics["smape"]:.2f}%')
print(f'test metrics - r2: {test_metrics["r2"]:.4f}, rmse: {test_metrics["rmse"]:.2f}, smape: {test_metrics["smape"]:.2f}%')

y_test_pred_bl = xgb_baseline.predict(X_test)
bl_test_metrics = evaluate_model(y_test, y_test_pred_bl)

y_train_pred_bl = xgb_baseline.predict(X_train)
bl_train_metrics = evaluate_model(y_train, y_train_pred_bl)

print(f'\ncomparison (validation set)')
print(f'baseline - r2: {bl_val_metrics["r2"]:.4f}, rmse: {bl_val_metrics["rmse"]:.2f}, smape: {bl_val_metrics["smape"]:.2f}%')
print(f'tuned - r2: {val_metrics["r2"]:.4f}, rmse: {val_metrics["rmse"]:.2f}, smape: {val_metrics["smape"]:.2f}%')
print(f'improvement - rmse: {bl_val_metrics["rmse"] - val_metrics["rmse"]:.2f} better')

print(f'\ncomparison (test set)')
print(f'baseline - r2: {bl_test_metrics["r2"]:.4f}, rmse: {bl_test_metrics["rmse"]:.2f}, smape: {bl_test_metrics["smape"]:.2f}%')
print(f'tuned - r2: {test_metrics["r2"]:.4f}, rmse: {test_metrics["rmse"]:.2f}, smape: {test_metrics["smape"]:.2f}%')
print(f'improvement - rmse: {bl_test_metrics["rmse"] - test_metrics["rmse"]:.2f} better')

print(f'\noverfitting check (train r2 - val r2)')
print(f'baseline: {bl_train_metrics["r2"] - bl_val_metrics["r2"]:.4f}')
print(f'tuned: {train_metrics["r2"] - val_metrics["r2"]:.4f}')

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_tuned.feature_importances_
}).sort_values('importance', ascending=False)

print('\ntop 15 feature importance:')
for i, row in feature_importance.head(15).iterrows():
    rank = feature_importance.index.get_loc(i) + 1
    print(f'{rank}. {row["feature"]}: {row["importance"]:.4f}')

print('\ncreating visualizations...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'xgboost regression results - baseline vs tuned\n'
             f'baseline: depth={baseline_params["max_depth"]}, trees={baseline_params["n_estimators"]} | '
             f'tuned: depth={best_params["max_depth"]}, trees={best_params["n_estimators"]}, lr={best_params["learning_rate"]}',
             fontsize=14)

cap = np.percentile(y_val, 95)

axes[0, 0].scatter(y_val, y_val_pred_bl, alpha=0.3, s=10, c='steelblue')
axes[0, 0].plot([0, cap], [0, cap], 'r--', lw=2)
axes[0, 0].set_xlim(0, cap)
axes[0, 0].set_ylim(0, cap)
axes[0, 0].set_xlabel('actual price')
axes[0, 0].set_ylabel('predicted price')
axes[0, 0].set_title(f'baseline (r2={bl_val_metrics["r2"]:.4f})')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_val, y_val_pred, alpha=0.3, s=10, c='coral')
axes[0, 1].plot([0, cap], [0, cap], 'r--', lw=2)
axes[0, 1].set_xlim(0, cap)
axes[0, 1].set_ylim(0, cap)
axes[0, 1].set_xlabel('actual price')
axes[0, 1].set_ylabel('predicted price')
axes[0, 1].set_title(f'tuned (r2={val_metrics["r2"]:.4f})')
axes[0, 1].grid(True, alpha=0.3)

metrics_names = ['r2', 'rmse/10', 'mae/10', 'mape/100', 'smape/100']
bl_vals = [bl_val_metrics['r2'], bl_val_metrics['rmse']/10, bl_val_metrics['mae']/10, 
           bl_val_metrics['mape']/100, bl_val_metrics['smape']/100]
tu_vals = [val_metrics['r2'], val_metrics['rmse']/10, val_metrics['mae']/10, 
           val_metrics['mape']/100, val_metrics['smape']/100]

x = np.arange(5)
width = 0.35
axes[0, 2].bar(x - width/2, bl_vals, width, label='baseline', color='steelblue', edgecolor='black')
axes[0, 2].bar(x + width/2, tu_vals, width, label='tuned', color='coral', edgecolor='black')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(metrics_names, fontsize=9)
axes[0, 2].set_ylabel('value (scaled)')
axes[0, 2].set_title('metrics comparison')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

res_bl = y_val - y_val_pred_bl
axes[1, 0].hist(res_bl, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 0].set_xlim(-50, 50)
axes[1, 0].set_xlabel('residual')
axes[1, 0].set_title(f'baseline residuals (mean={res_bl.mean():.2f})')

res_tu = y_val - y_val_pred
axes[1, 1].hist(res_tu, bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 1].set_xlim(-50, 50)
axes[1, 1].set_xlabel('residual')
axes[1, 1].set_title(f'tuned residuals (mean={res_tu.mean():.2f})')

top10_features = feature_importance.head(10)
axes[1, 2].barh(range(10), top10_features['importance'].values, color='teal', edgecolor='black')
axes[1, 2].set_yticks(range(10))
axes[1, 2].set_yticklabels(top10_features['feature'].values, fontsize=9)
axes[1, 2].invert_yaxis()
axes[1, 2].set_xlabel('importance')
axes[1, 2].set_title('top 10 feature importance')
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output_file = output_dir + 'xgboost_results.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'saved visualization: {output_file}')
plt.close()

print('\nfinal summary')
print(f'total time: {(baseline_time + tune_time + train_time)/60:.1f} minutes')
print(f'test set final results - r2: {test_metrics["r2"]:.4f}, rmse: {test_metrics["rmse"]:.2f}, mae: {test_metrics["mae"]:.2f}')
print(f'mape: {test_metrics["mape"]:.2f}%, smape: {test_metrics["smape"]:.2f}%')
print(f'improvement over baseline: {bl_test_metrics["rmse"] - test_metrics["rmse"]:.2f} rmse reduction')
print(f'\ntop 3 features:')
print(f'1. {feature_importance.iloc[0]["feature"]}: {feature_importance.iloc[0]["importance"]:.4f}')
print(f'2. {feature_importance.iloc[1]["feature"]}: {feature_importance.iloc[1]["importance"]:.4f}')
print(f'3. {feature_importance.iloc[2]["feature"]}: {feature_importance.iloc[2]["importance"]:.4f}')
print('done')
