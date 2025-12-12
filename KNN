import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

input_path = '/Users/ayalabnine/Downloads/EECE5644/Datasets/Final Dataset/train_features_v2.csv'
output_dir = '/Users/ayalabnine/Downloads/EECE5644/SRC_EECE5644/KNN_Results/'
os.makedirs(output_dir, exist_ok=True)

tuning_mode = 'fast'

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

def get_tuning_config(mode):
    configs = {
        'ultra_fast': {
            'n_iter': 15,
            'cv': 3,
            'sample_frac': 0.15,
            'param_dist': {
                'n_neighbors': [5, 10, 15, 20, 30],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'fast': {
            'n_iter': 25,
            'cv': 3,
            'sample_frac': 0.25,
            'param_dist': {
                'n_neighbors': [3, 5, 7, 10, 15, 20, 30, 40],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'balanced': {
            'n_iter': 40,
            'cv': 5,
            'sample_frac': 0.40,
            'param_dist': {
                'n_neighbors': [3, 5, 7, 10, 15, 20, 30, 40, 50],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'thorough': {
            'n_iter': 60,
            'cv': 5,
            'sample_frac': 0.60,
            'param_dist': {
                'n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
    }
    return configs[mode]

print('knn regression - amazon price prediction')
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

config = get_tuning_config(tuning_mode)
sample_size = int(len(X_train_scaled) * config['sample_frac'])
sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_tune = X_train_scaled[sample_indices]
y_tune = y_train.iloc[sample_indices]

print(f'using {sample_size} samples for tuning ({config["sample_frac"]*100:.0f}% of training data)')

baseline_params = {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'}
print('\ntraining baseline model (k=5, uniform, euclidean)...')

start = time.time()
knn_baseline = KNeighborsRegressor(**baseline_params, n_jobs=-1)
knn_baseline.fit(X_train_scaled, y_train)
baseline_time = time.time() - start

y_val_pred_bl = knn_baseline.predict(X_val_scaled)
bl_val_metrics = evaluate_model(y_val, y_val_pred_bl)

print(f'baseline validation - r2: {bl_val_metrics["r2"]:.4f}, rmse: {bl_val_metrics["rmse"]:.2f}, smape: {bl_val_metrics["smape"]:.2f}%')

print(f'\ntuning hyperparameters ({config["n_iter"]} combinations, {config["cv"]}-fold cv)...')

start = time.time()
random_search = RandomizedSearchCV(
    KNeighborsRegressor(n_jobs=-1),
    param_distributions=config['param_dist'],
    n_iter=config['n_iter'],
    cv=config['cv'],
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    random_state=42
)
random_search.fit(X_tune, y_tune)
tune_time = time.time() - start

best_params = random_search.best_params_
best_score = random_search.best_score_
search_results = random_search.cv_results_

print(f'tuning complete in {tune_time/60:.1f} minutes')
print(f'best params - k: {best_params["n_neighbors"]}, weights: {best_params["weights"]}, metric: {best_params["metric"]}')

print('\ntraining final model with best params...')
start = time.time()
knn_tuned = KNeighborsRegressor(**best_params, n_jobs=-1)
knn_tuned.fit(X_train_scaled, y_train)
train_time = time.time() - start

y_train_pred = knn_tuned.predict(X_train_scaled)
y_val_pred = knn_tuned.predict(X_val_scaled)
y_test_pred = knn_tuned.predict(X_test_scaled)

train_metrics = evaluate_model(y_train, y_train_pred)
val_metrics = evaluate_model(y_val, y_val_pred)
test_metrics = evaluate_model(y_test, y_test_pred)

print(f'\ntrain metrics - r2: {train_metrics["r2"]:.4f}, rmse: {train_metrics["rmse"]:.2f}')
print(f'val metrics - r2: {val_metrics["r2"]:.4f}, rmse: {val_metrics["rmse"]:.2f}, smape: {val_metrics["smape"]:.2f}%')
print(f'test metrics - r2: {test_metrics["r2"]:.4f}, rmse: {test_metrics["rmse"]:.2f}, smape: {test_metrics["smape"]:.2f}%')

y_test_pred_bl = knn_baseline.predict(X_test_scaled)
bl_test_metrics = evaluate_model(y_test, y_test_pred_bl)

print(f'\ncomparison (validation set)')
print(f'baseline - r2: {bl_val_metrics["r2"]:.4f}, rmse: {bl_val_metrics["rmse"]:.2f}, smape: {bl_val_metrics["smape"]:.2f}%')
print(f'tuned - r2: {val_metrics["r2"]:.4f}, rmse: {val_metrics["rmse"]:.2f}, smape: {val_metrics["smape"]:.2f}%')
print(f'improvement - rmse: {bl_val_metrics["rmse"] - val_metrics["rmse"]:.2f} better')

print(f'\ncomparison (test set)')
print(f'baseline - r2: {bl_test_metrics["r2"]:.4f}, rmse: {bl_test_metrics["rmse"]:.2f}, smape: {bl_test_metrics["smape"]:.2f}%')
print(f'tuned - r2: {test_metrics["r2"]:.4f}, rmse: {test_metrics["rmse"]:.2f}, smape: {test_metrics["smape"]:.2f}%')
print(f'improvement - rmse: {bl_test_metrics["rmse"] - test_metrics["rmse"]:.2f} better')

print('\ncreating visualizations...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'knn regression results - baseline vs tuned\n'
             f'baseline: k={baseline_params["n_neighbors"]} | '
             f'tuned: k={best_params["n_neighbors"]}, {best_params["weights"]}, {best_params["metric"]}',
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

cv_results_df = pd.DataFrame(search_results)
cv_results_df['rmse'] = np.sqrt(-cv_results_df['mean_test_score'])

for weight in ['uniform', 'distance']:
    mask = cv_results_df['param_weights'] == weight
    if mask.sum() > 0:
        subset = cv_results_df[mask].groupby('param_n_neighbors')['rmse'].mean()
        axes[1, 2].plot(subset.index, subset.values, marker='o', lw=2, label=weight, markersize=6)

axes[1, 2].axvline(best_params['n_neighbors'], color='green', linestyle='--', lw=2, 
                   label=f'best k={best_params["n_neighbors"]}')
axes[1, 2].set_xlabel('number of neighbors (k)')
axes[1, 2].set_ylabel('cv rmse')
axes[1, 2].set_title('effect of k on performance')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
output_file = output_dir + f'knn_results_{tuning_mode}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'saved visualization: {output_file}')
plt.close()

print('\nfinal summary')
print(f'total time: {(baseline_time + tune_time + train_time)/60:.1f} minutes')
print(f'test set final results - r2: {test_metrics["r2"]:.4f}, rmse: {test_metrics["rmse"]:.2f}, mae: {test_metrics["mae"]:.2f}')
print(f'mape: {test_metrics["mape"]:.2f}%, smape: {test_metrics["smape"]:.2f}%')
print(f'improvement over baseline: {bl_test_metrics["rmse"] - test_metrics["rmse"]:.2f} rmse reduction')
print('done')
