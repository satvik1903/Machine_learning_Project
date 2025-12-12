import pandas as pd
import numpy as np
import os
import warnings
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

BASE_PATH = '/Users/satviktajne/Desktop/Sem 1/ML and PR/ML Project/CODE/dataset/Final Datasets'
TEST_PATH = f'{BASE_PATH}/test_features_v2.csv'
MODEL_DIR = f'{BASE_PATH}/CNN_SVR_Results'
OUTPUT_PATH = f'{BASE_PATH}/CNN_SVR_Results/predictions.csv'

USE_LOG_TRANSFORM = True

numerical_features = [
    'unit_size', 'size_in_oz', 'pack_size', 'total_size',
    'has_premium_word', 'premium_word_count',
    'has_health_word', 'health_word_count',
    'has_ingredients', 'ingredient_word_count'
]

print("Loading model artifacts...")
feature_extractor = load_model(f'{MODEL_DIR}/feature_extractor.h5', compile=False)
svr_model = joblib.load(f'{MODEL_DIR}/svr_model.pkl')
scaler_X = joblib.load(f'{MODEL_DIR}/scaler_X.pkl')
scaler_y = joblib.load(f'{MODEL_DIR}/scaler_y.pkl')
encoders = joblib.load(f'{MODEL_DIR}/encoders.pkl')

print("Loading test data...")
df_test = pd.read_csv(TEST_PATH)
print(f"Loaded: {len(df_test)} rows")

if 'sample_id' in df_test.columns:
    test_ids = df_test['sample_id'].values
elif 'id' in df_test.columns:
    test_ids = df_test['id'].values
elif 'item_id' in df_test.columns:
    test_ids = df_test['item_id'].values
else:
    test_ids = np.arange(len(df_test))

print("Preprocessing test data...")

if 'category' in df_test.columns and 'category' in encoders:
    known_categories = set(encoders['category'].classes_)
    df_test['category_clean'] = df_test['category'].fillna('Unknown').apply(
        lambda x: x if x in known_categories else 'Unknown'
    )
    if 'Unknown' not in known_categories:
        df_test['category_encoded'] = df_test['category_clean'].map(
            {cat: idx for idx, cat in enumerate(encoders['category'].classes_)}
        ).fillna(-1).astype(int)
    else:
        df_test['category_encoded'] = encoders['category'].transform(df_test['category_clean'])

if 'unit_type' in df_test.columns and 'unit_type' in encoders:
    known_unit_types = set(encoders['unit_type'].classes_)
    df_test['unit_type_clean'] = df_test['unit_type'].fillna('Unknown').apply(
        lambda x: x if x in known_unit_types else 'Unknown'
    )
    if 'Unknown' not in known_unit_types:
        df_test['unit_type_encoded'] = df_test['unit_type_clean'].map(
            {ut: idx for idx, ut in enumerate(encoders['unit_type'].classes_)}
        ).fillna(-1).astype(int)
    else:
        df_test['unit_type_encoded'] = encoders['unit_type'].transform(df_test['unit_type_clean'])

if 'brand_popularity' in df_test.columns:
    df_test['brand_popularity_encoded'] = df_test['brand_popularity'].map(
        {'low': 0, 'mid': 1, 'high': 2}
    ).fillna(0)

all_features = numerical_features + ['category_encoded', 'unit_type_encoded', 'brand_popularity_encoded']
X_test = df_test[all_features].fillna(0)

print("Applying transformations...")
X_test_scaled = scaler_X.transform(X_test)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

print("Generating predictions...")
X_test_features = feature_extractor.predict(X_test_cnn, verbose=0)
predictions_scaled = svr_model.predict(X_test_features)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

if USE_LOG_TRANSFORM:
    predictions = np.expm1(predictions)

predictions = np.maximum(predictions, 0.01)

print(f"Predictions: {len(predictions)}")
print(f"Min: ${predictions.min():.2f}, Max: ${predictions.max():.2f}")
print(f"Mean: ${predictions.mean():.2f}, Median: ${np.median(predictions):.2f}")

output_df = pd.DataFrame({
    'sample_id': test_ids,
    'item_price': predictions
})
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")