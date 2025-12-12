import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

BASE_PATH = '/Users/satviktajne/Desktop/Sem 1/ML and PR/ML Project/CODE/dataset/Final Datasets'
TRAIN_PATH = f'{BASE_PATH}/train_features_v2.csv'
OUTPUT_DIR = f'{BASE_PATH}/RF_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

N_ITER = 10
CV_FOLDS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("RANDOM FOREST - QUICK MODE (~1-2 min)")
print(f"N_ITER={N_ITER}, CV={CV_FOLDS}, Fits={N_ITER*CV_FOLDS}")

df_train = pd.read_csv(TRAIN_PATH)
print(f"\nLoaded: {len(df_train)} rows")

numerical_features = [
    'unit_size', 'size_in_oz', 'pack_size', 'total_size',
    'has_premium_word', 'premium_word_count',
    'has_health_word', 'health_word_count',
    'has_ingredients', 'ingredient_word_count'
]

encoders = {}
if 'category' in df_train.columns:
    encoders['category'] = LabelEncoder()
    df_train['category_encoded'] = encoders['category'].fit_transform(df_train['category'].fillna('Unknown'))
if 'unit_type' in df_train.columns:
    encoders['unit_type'] = LabelEncoder()
    df_train['unit_type_encoded'] = encoders['unit_type'].fit_transform(df_train['unit_type'].fillna('Unknown'))
if 'brand_popularity' in df_train.columns:
    df_train['brand_popularity_encoded'] = df_train['brand_popularity'].map({'low': 0, 'mid': 1, 'high': 2}).fillna(0)

all_features = numerical_features + ['category_encoded', 'unit_type_encoded', 'brand_popularity_encoded']

X = df_train[all_features].fillna(0)
y = df_train['item_price'].fillna(df_train['item_price'].median())
X, y = X[y > 0], y[y > 0]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

print("\nQUICK RANDOMIZEDSEARCHCV")

param_distributions = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

print(f"Fits: {N_ITER * CV_FOLDS} | Est. time: ~1-2 min")
print("\nRunning...")
start_time = time.time()

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions, n_iter=N_ITER, cv=CV_FOLDS,
    scoring='neg_mean_squared_error', n_jobs=-1,
    random_state=RANDOM_STATE, verbose=1
)
random_search.fit(X_train, y_train)
search_time = time.time() - start_time

print(f"\nDone in {search_time:.1f}s ({search_time/60:.1f} min)")

print("\nBEST PARAMS:")
for p, v in random_search.best_params_.items():
    print(f"   {p}: {v}")
print(f"\nBest CV RMSE: ${np.sqrt(-random_search.best_score_):.2f}")

RF_PARAMS = random_search.best_params_.copy()
RF_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'oob_score': True})

rf_model = RandomForestRegressor(**RF_PARAMS)
rf_model.fit(X_train, y_train)
print(f"\nModel trained | OOB R2: {rf_model.oob_score_:.4f}")

y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)

def smape(y_true, y_pred):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(num[den != 0] / den[den != 0]) * 100

def metrics(y_true, y_pred, n_feat):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2,
        'Adj_R2': 1 - (1 - r2) * (n - 1) / (n - n_feat - 1),
        'MAPE': np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100,
        'SMAPE': smape(y_true, y_pred)
    }

train_m = metrics(y_train.values, y_train_pred, len(all_features))
val_m = metrics(y_val.values, y_val_pred, len(all_features))

print("\nRESULTS")
print("\nREGRESSION METRICS")
print(f"{'TRAINING':<30} {'VALIDATION':<30}")

for m in ['MAE', 'RMSE', 'R2', 'Adj_R2', 'MAPE', 'SMAPE']:
    t, v = train_m[m], val_m[m]
    if m in ['MAE', 'RMSE']:
        ts, vs = f"{m}: ${t:.2f}", f"{m}: ${v:.2f}"
    elif m in ['MAPE', 'SMAPE']:
        ts, vs = f"{m}: {t:.2f}%", f"{m}: {v:.2f}%"
    else:
        ts, vs = f"{m}: {t:.4f}", f"{m}: {v:.4f}"
    print(f"  {ts:<26}   {vs:<25}")

r2_gap = train_m['R2'] - val_m['R2']
if r2_gap > 0.15: fit_status = "OVERFITTING"
elif r2_gap > 0.05: fit_status = "MODERATE"
elif val_m['R2'] < 0.3: fit_status = "UNDERFITTING"
else: fit_status = "GOOD FIT"
print(f"\nR2 Gap: {r2_gap:.4f} -> {fit_status}")

cv_r2 = cross_val_score(RandomForestRegressor(**{k:v for k,v in RF_PARAMS.items() if k!='oob_score'}),
                        X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
print(f"CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std()*2:.4f})")

print("\nFEATURE IMPORTANCE:")
feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': rf_model.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=False)
for _, r in feat_imp.iterrows():
    print(f"   {r['Feature']:<28} {r['Importance']:.4f}")

joblib.dump(rf_model, f'{OUTPUT_DIR}/rf_model.pkl')
joblib.dump(scaler_X, f'{OUTPUT_DIR}/scaler_X.pkl')
joblib.dump(encoders, f'{OUTPUT_DIR}/encoders.pkl')

pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2', 'Adj_R2', 'MAPE', 'SMAPE', 'OOB_R2'],
    'Training': [train_m['MAE'], train_m['RMSE'], train_m['R2'], train_m['Adj_R2'], train_m['MAPE'], train_m['SMAPE'], '-'],
    'Validation': [val_m['MAE'], val_m['RMSE'], val_m['R2'], val_m['Adj_R2'], val_m['MAPE'], val_m['SMAPE'], rf_model.oob_score_]
}).to_csv(f'{OUTPUT_DIR}/rf_metrics.csv', index=False)

feat_imp.to_csv(f'{OUTPUT_DIR}/rf_feature_importance.csv', index=False)
print(f"\nSaved to {OUTPUT_DIR}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=10, c='forestgreen')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual ($)'); axes[0].set_ylabel('Predicted ($)')
axes[0].set_title(f'Train: R2={train_m["R2"]:.3f}, SMAPE={train_m["SMAPE"]:.1f}%')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_val, y_val_pred, alpha=0.3, s=10, c='darkorange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual ($)'); axes[1].set_ylabel('Predicted ($)')
axes[1].set_title(f'Val: R2={val_m["R2"]:.3f}, SMAPE={val_m["SMAPE"]:.1f}%')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_predictions.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=plt.cm.Greens(np.linspace(0.4, 0.9, len(feat_imp)))[::-1])
ax.set_xlabel('Importance'); ax.set_title('Feature Importance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_feature_importance.png', dpi=150)
plt.close()
print("Plots saved")

print("\nFINAL SUMMARY")
print(f"""
RANDOM FOREST - QUICK RESULTS
Time: {search_time:.1f}s | Fits: {N_ITER*CV_FOLDS}

VALIDATION:
  MAE:   ${val_m['MAE']:<8.2f}  RMSE:  ${val_m['RMSE']:<8.2f}
  R2:    {val_m['R2']:<8.4f}  SMAPE: {val_m['SMAPE']:<8.2f}%
  MAPE:  {val_m['MAPE']:<8.2f}%  OOB:   {rf_model.oob_score_:<8.4f}

BEST PARAMS:
  n_estimators: {RF_PARAMS.get('n_estimators'):<6} max_depth: {str(RF_PARAMS.get('max_depth')):<10}
  min_samples_split: {RF_PARAMS.get('min_samples_split'):<4} min_samples_leaf: {RF_PARAMS.get('min_samples_leaf'):<4}
  max_features: {str(RF_PARAMS.get('max_features')):<10}

Status: {fit_status}
""")
print("DONE!")