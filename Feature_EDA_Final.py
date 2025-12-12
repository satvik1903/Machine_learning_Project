import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
warnings.filterwarnings('ignore')

remove_outliers = True
price_upper_limit = 2000
price_lower_limit = 0

unit_to_oz = {
    'oz': 1.0, 'ounce': 1.0, 'ounces': 1.0,
    'lb': 16.0, 'lbs': 16.0, 'pound': 16.0, 'pounds': 16.0,
    'g': 0.035274, 'gram': 0.035274, 'grams': 0.035274,
    'kg': 35.274, 'kilogram': 35.274,
    'fl oz': 1.0, 'fl': 1.0, 'fluid ounce': 1.0,
    'ml': 0.033814, 'milliliter': 0.033814,
    'l': 33.814, 'liter': 33.814, 'litre': 33.814,
    'gallon': 128.0, 'gal': 128.0,
    'quart': 32.0, 'qt': 32.0,
    'pint': 16.0, 'pt': 16.0,
    'cup': 8.0,
    'count': 1.0, 'ct': 1.0, 'piece': 1.0, 'pieces': 1.0,
    'each': 1.0, 'pack': 1.0, 'bag': 1.0, 'box': 1.0,
    'can': 1.0, 'bottle': 1.0, 'jar': 1.0,
}

def convert_to_oz(value, unit):
    if pd.isna(value) or pd.isna(unit) or value == 0:
        return 0.0
    unit_lower = str(unit).lower().strip()
    if unit_lower in unit_to_oz:
        return float(value) * unit_to_oz[unit_lower]
    for key, multiplier in unit_to_oz.items():
        if key in unit_lower:
            return float(value) * multiplier
    return float(value)

def remove_price_outliers(df, price_column='price', lower_limit=0, upper_limit=2000):
    if price_column not in df.columns:
        return df
    df_clean = df[(df[price_column] > lower_limit) & (df[price_column] <= upper_limit)].copy()
    return df_clean

def parse_catalog_content(catalog_text):
    if pd.isna(catalog_text):
        return {'Item Name': None, 'Bullet Points': None, 'Product Description': None, 'Value': None, 'Unit': None}
    
    components = {}
    
    item_match = re.search(r'(?m)^Item Name:\s*(?P<item>.*?)(?=\n\s*Bullet Point|\n\s*Product Description|\n\s*Value:|\Z)', catalog_text, re.DOTALL | re.MULTILINE)
    components['Item Name'] = item_match.group('item').strip() if item_match else None
    
    bullet_points = re.findall(r'Bullet Point \d+:\s*(.*?)(?=\n\s*Bullet Point \d+:|\n\s*Product Description:|\n\s*Value:|\Z)', catalog_text, re.DOTALL)
    components['Bullet Points'] = ' '.join(bullet_points).strip() if bullet_points else None
    
    desc_match = re.search(r'(?m)^Product Description:\s*(?P<desc>.*?)(?=\n\s*Value:|\Z)', catalog_text, re.DOTALL | re.MULTILINE)
    components['Product Description'] = desc_match.group('desc').strip() if desc_match else None
    
    value_match = re.search(r'Value:\s*(?P<value>[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)', catalog_text)
    if value_match:
        try:
            components['Value'] = float(value_match.group('value').replace(',', ''))
        except ValueError:
            components['Value'] = None
    else:
        components['Value'] = None
    
    unit_match = re.search(r'(?m)^Unit:\s*(?P<unit>.*?)$', catalog_text, re.MULTILINE)
    components['Unit'] = unit_match.group('unit').strip() if unit_match else None
    
    return components

def extract_features(df, dataset_type='train'):
    df_features = df.copy()
    
    if 'catalog_content' in df_features.columns:
        parsed_data = df_features['catalog_content'].apply(parse_catalog_content)
        parsed_df = pd.DataFrame(parsed_data.tolist())
        for col in parsed_df.columns:
            df_features[col] = parsed_df[col]
    
    def get_combined_text(row):
        texts = []
        for col in ['Item Name', 'Bullet Points', 'Product Description']:
            if col in row and pd.notna(row[col]):
                texts.append(str(row[col]))
        return ' '.join(texts)
    
    df_features['_combined_text'] = df_features.apply(get_combined_text, axis=1)
    df_features['item_name'] = df_features['Item Name'].fillna('Unknown')
    
    if dataset_type == 'train':
        df_features['item_price'] = df_features['price'] if 'price' in df_features.columns else np.nan
    
    def extract_category(row):
        text = str(row.get('Item Name', '')) + ' ' + str(row.get('Bullet Points', ''))
        if pd.isna(text) or text == 'nan nan':
            return 'Other'
        text_lower = text.lower()
        categories = {
            'sauce': ['sauce', 'salsa', 'dressing', 'marinade', 'gravy', 'ketchup', 'mayo'],
            'snack': ['chips', 'popcorn', 'crackers', 'pretzels', 'nuts', 'trail mix', 'snack'],
            'candy': ['candy', 'chocolate', 'gummy', 'taffy', 'caramel', 'lollipop', 'sweet'],
            'beverage': ['juice', 'tea', 'coffee', 'water', 'soda', 'drink', 'cola'],
            'soup': ['soup', 'broth', 'stew', 'chowder'],
            'cereal': ['cereal', 'granola', 'oatmeal', 'muesli'],
            'pasta': ['pasta', 'noodle', 'macaroni', 'spaghetti', 'ramen'],
            'seasoning': ['spice', 'seasoning', 'salt', 'pepper', 'herb'],
            'baking': ['flour', 'sugar', 'baking', 'yeast', 'cake'],
            'protein': ['meat', 'chicken', 'beef', 'pork', 'fish', 'jerky'],
            'dairy': ['cheese', 'milk', 'yogurt', 'butter', 'cream'],
            'bread': ['bread', 'bagel', 'muffin', 'toast', 'roll'],
            'oil': ['oil', 'vinegar', 'cooking spray'],
        }
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'Other'
    
    df_features['category'] = df_features.apply(extract_category, axis=1)
    df_features['unit_size'] = pd.to_numeric(df_features['Value'], errors='coerce').fillna(0)
    df_features['unit_type'] = df_features['Unit'].fillna('Unknown')
    df_features['size_in_oz'] = df_features.apply(lambda row: convert_to_oz(row['unit_size'], row['unit_type']), axis=1)
    
    def extract_pack_size(name):
        if pd.isna(name):
            return 1
        name_str = str(name).lower()
        patterns = [r'pack of (\d+)', r'(\d+)[\s-]pack', r'(\d+)[\s-]ct', r'(\d+)[\s-]count', r'(\d+) count', r'(\d+)[\s-]pk']
        for pattern in patterns:
            match = re.search(pattern, name_str)
            if match:
                return int(match.group(1))
        return 1
    
    df_features['pack_size'] = df_features['Item Name'].apply(extract_pack_size)
    df_features['total_size'] = df_features['size_in_oz'] * df_features['pack_size']
    
    if dataset_type == 'train':
        df_features['price_per_ounce'] = df_features.apply(
            lambda row: row['item_price'] / row['total_size'] if row['total_size'] > 0 and pd.notna(row['item_price']) else 0, axis=1)
        df_features['price_per_unit'] = df_features.apply(
            lambda row: row['item_price'] / row['pack_size'] if row['pack_size'] > 0 and pd.notna(row['item_price']) else 0, axis=1)
    
    def extract_brand(name):
        if pd.isna(name):
            return 'Unknown'
        words = str(name).split()[:3]
        non_brand_words = ['The', 'A', 'An', 'Premium', 'Organic', 'Natural']
        brand_words = [w for w in words if w not in non_brand_words]
        return ' '.join(brand_words[:2]) if brand_words else 'Unknown'
    
    df_features['_brand'] = df_features['Item Name'].apply(extract_brand)
    brand_counts = df_features['_brand'].value_counts().to_dict()
    df_features['_brand_count'] = df_features['_brand'].map(brand_counts)
    low_threshold = df_features['_brand_count'].quantile(0.33)
    high_threshold = df_features['_brand_count'].quantile(0.66)
    conditions = [df_features['_brand_count'] <= low_threshold, df_features['_brand_count'] <= high_threshold, df_features['_brand_count'] > high_threshold]
    df_features['brand_popularity'] = np.select(conditions, ['low', 'mid', 'high'], default='low')
    
    premium_words = ['gourmet', 'premium', 'artisan', 'imported', 'luxury', 'finest', 'superior', 'deluxe', 'exclusive', 'specialty', 'authentic', 'handcrafted']
    df_features['has_premium_word'] = df_features['_combined_text'].apply(lambda t: int(any(w in str(t).lower() for w in premium_words)) if pd.notna(t) else 0)
    df_features['premium_word_count'] = df_features['_combined_text'].apply(lambda t: sum(1 for w in premium_words if w in str(t).lower()) if pd.notna(t) else 0)
    
    health_words = ['organic', 'natural', 'gluten-free', 'vegan', 'sugar-free', 'low-fat', 'non-gmo', 'keto', 'paleo', 'whole grain', 'low sodium', 'kosher', 'dairy-free', 'plant-based', 'healthy', 'nutritious']
    df_features['has_health_word'] = df_features['_combined_text'].apply(lambda t: int(any(w in str(t).lower() for w in health_words)) if pd.notna(t) else 0)
    df_features['health_word_count'] = df_features['_combined_text'].apply(lambda t: sum(1 for w in health_words if w in str(t).lower()) if pd.notna(t) else 0)
    
    ingredient_words = ['chicken', 'beef', 'pork', 'fish', 'vegetable', 'fruit', 'cheese', 'butter', 'chocolate', 'vanilla', 'berry', 'nut', 'honey', 'rice', 'wheat', 'tomato', 'onion', 'garlic', 'pepper', 'corn', 'potato', 'milk', 'egg', 'cream', 'sugar', 'flour', 'olive', 'lemon', 'lime']
    df_features['has_ingredients'] = df_features['_combined_text'].apply(lambda t: int(any(w in str(t).lower() for w in ingredient_words)) if pd.notna(t) else 0)
    df_features['ingredient_word_count'] = df_features['_combined_text'].apply(lambda t: sum(1 for w in ingredient_words if w in str(t).lower()) if pd.notna(t) else 0)
    
    if dataset_type == 'train':
        final_columns = ['item_name', 'item_price', 'category', 'unit_size', 'unit_type', 'price_per_ounce', 'price_per_unit', 'size_in_oz', 'pack_size', 'total_size', 'brand_popularity', 'has_premium_word', 'premium_word_count', 'has_health_word', 'health_word_count', 'has_ingredients', 'ingredient_word_count']
    else:
        final_columns = ['item_name', 'category', 'unit_size', 'unit_type', 'size_in_oz', 'pack_size', 'total_size', 'brand_popularity', 'has_premium_word', 'premium_word_count', 'has_health_word', 'health_word_count', 'has_ingredients', 'ingredient_word_count']
    
    existing_columns = [col for col in final_columns if col in df_features.columns]
    df_final = df_features[existing_columns].copy()
    
    numeric_cols = ['unit_size', 'size_in_oz', 'pack_size', 'total_size', 'has_premium_word', 'premium_word_count', 'has_health_word', 'health_word_count', 'has_ingredients', 'ingredient_word_count']
    if dataset_type == 'train':
        numeric_cols.extend(['price_per_ounce', 'price_per_unit'])
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
    
    numeric_to_round = ['size_in_oz', 'total_size', 'unit_size']
    if dataset_type == 'train':
        numeric_to_round.extend(['price_per_ounce', 'price_per_unit', 'item_price'])
    for col in numeric_to_round:
        if col in df_final.columns:
            df_final[col] = df_final[col].round(2)
    
    return df_final

def run_eda(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    df = pd.read_csv(input_path)
    
    price_limit = df['item_price'].quantile(0.95)
    price_zoomed = df[df['item_price'] <= price_limit]['item_price']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(price_zoomed, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(price_zoomed.mean(), color='red', linestyle='--', lw=2, label=f'mean: ${price_zoomed.mean():.2f}')
    axes[0, 0].axvline(price_zoomed.median(), color='green', linestyle='--', lw=2, label=f'median: ${price_zoomed.median():.2f}')
    axes[0, 0].set_xlabel('item price ($)', fontsize=12)
    axes[0, 0].set_ylabel('frequency', fontsize=12)
    axes[0, 0].set_title('distribution of item price', fontsize=14)
    axes[0, 0].legend()
    
    bp1 = axes[0, 1].boxplot(price_zoomed, vert=True, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
    axes[0, 1].set_ylabel('item price ($)', fontsize=12)
    axes[0, 1].set_title('item price box plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(np.log1p(df['item_price']), bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].set_xlabel('log(item price + 1)', fontsize=12)
    axes[1, 0].set_ylabel('frequency', fontsize=12)
    axes[1, 0].set_title('log-transformed price distribution', fontsize=14)
    
    sns.kdeplot(data=price_zoomed, ax=axes[1, 1], fill=True, color='steelblue')
    axes[1, 1].set_xlabel('item price ($)', fontsize=12)
    axes[1, 1].set_ylabel('density', fontsize=12)
    axes[1, 1].set_title('price density', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_target_item_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    price_features = ['price_per_ounce', 'price_per_unit']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ppo_limit = df['price_per_ounce'].quantile(0.95)
    ppo_data = df[df['price_per_ounce'] <= ppo_limit]['price_per_ounce']
    axes[0, 0].hist(ppo_data, bins=50, edgecolor='black', alpha=0.7, color='teal')
    axes[0, 0].axvline(ppo_data.mean(), color='red', linestyle='--', lw=2, label=f'mean: ${ppo_data.mean():.2f}')
    axes[0, 0].axvline(ppo_data.median(), color='green', linestyle='--', lw=2, label=f'median: ${ppo_data.median():.2f}')
    axes[0, 0].set_xlabel('price per ounce ($/oz)', fontsize=12)
    axes[0, 0].set_ylabel('frequency', fontsize=12)
    axes[0, 0].set_title('distribution of price per ounce', fontsize=14)
    axes[0, 0].legend()
    
    bp2 = axes[0, 1].boxplot(ppo_data, vert=True, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('teal')
    axes[0, 1].set_ylabel('price per ounce ($/oz)', fontsize=12)
    axes[0, 1].set_title('price per ounce box plot', fontsize=14)
    
    ppu_limit = df['price_per_unit'].quantile(0.95)
    ppu_data = df[df['price_per_unit'] <= ppu_limit]['price_per_unit']
    axes[1, 0].hist(ppu_data, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 0].axvline(ppu_data.mean(), color='red', linestyle='--', lw=2, label=f'mean: ${ppu_data.mean():.2f}')
    axes[1, 0].axvline(ppu_data.median(), color='green', linestyle='--', lw=2, label=f'median: ${ppu_data.median():.2f}')
    axes[1, 0].set_xlabel('price per unit ($)', fontsize=12)
    axes[1, 0].set_ylabel('frequency', fontsize=12)
    axes[1, 0].set_title('distribution of price per unit', fontsize=14)
    axes[1, 0].legend()
    
    bp3 = axes[1, 1].boxplot(ppu_data, vert=True, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('purple')
    axes[1, 1].set_ylabel('price per unit ($)', fontsize=12)
    axes[1, 1].set_title('price per unit box plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_price_normalization_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    size_features = ['unit_size', 'size_in_oz', 'pack_size', 'total_size']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, col in enumerate(size_features):
        row, c = idx // 2, idx % 2
        upper_limit = df[col].quantile(0.95)
        zoomed_data = df[df[col] <= upper_limit][col]
        axes[row, c].hist(zoomed_data, bins=50, edgecolor='black', alpha=0.7, color=['steelblue', 'coral', 'teal', 'purple'][idx])
        axes[row, c].axvline(zoomed_data.mean(), color='red', linestyle='--', lw=2, label=f'mean: {zoomed_data.mean():.2f}')
        axes[row, c].axvline(zoomed_data.median(), color='green', linestyle='--', lw=2, label=f'median: {zoomed_data.median():.2f}')
        axes[row, c].set_xlabel(col.replace('_', ' '), fontsize=12)
        axes[row, c].set_ylabel('frequency', fontsize=12)
        axes[row, c].set_title(f'distribution of {col.replace("_", " ")}', fontsize=14)
        axes[row, c].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_size_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, col in enumerate(size_features):
        row, c = idx // 2, idx % 2
        data = df[col][df[col] > 0]
        axes[row, c].hist(data, bins=50, edgecolor='black', alpha=0.7, color=['steelblue', 'coral', 'teal', 'purple'][idx])
        axes[row, c].set_xscale('log')
        axes[row, c].set_xlabel(f'{col.replace("_", " ")} (log scale)', fontsize=12)
        axes[row, c].set_ylabel('frequency', fontsize=12)
        axes[row, c].set_title(f'distribution of {col.replace("_", " ")} (log scale)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03c_size_features_logscale.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 5))
    pack_counts = df['pack_size'].value_counts().head(15)
    pack_counts.plot(kind='bar', color='teal', edgecolor='black')
    plt.xlabel('pack size', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.title('distribution of pack size (top 15)', fontsize=14)
    plt.xticks(rotation=0)
    for i, v in enumerate(pack_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03b_pack_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    categorical_features = ['category', 'unit_type', 'brand_popularity']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cat_counts = df['category'].value_counts()
    cat_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_xlabel('category', fontsize=12)
    axes[0].set_ylabel('frequency', fontsize=12)
    axes[0].set_title('distribution of category', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    unit_counts = df['unit_type'].value_counts().head(10)
    unit_counts.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_xlabel('unit type', fontsize=12)
    axes[1].set_ylabel('frequency', fontsize=12)
    axes[1].set_title('distribution of unit type (top 10)', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    pop_order = ['low', 'mid', 'high']
    pop_counts = df['brand_popularity'].value_counts().reindex(pop_order)
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    pop_counts.plot(kind='bar', ax=axes[2], color=colors, edgecolor='black')
    axes[2].set_xlabel('brand popularity', fontsize=12)
    axes[2].set_ylabel('frequency', fontsize=12)
    axes[2].set_title('distribution of brand popularity', fontsize=14)
    axes[2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_categorical_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    binary_features = ['has_premium_word', 'has_health_word', 'has_ingredients']
    count_features = ['premium_word_count', 'health_word_count', 'ingredient_word_count']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(binary_features):
        counts = df[col].value_counts()
        labels = ['no', 'yes']
        colors = ['#ff6b6b', '#6bcb77']
        axes[idx].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, explode=(0, 0.05), startangle=90)
        axes[idx].set_title(f'{col.replace("_", " ")}', fontsize=12)
    plt.suptitle('binary feature distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05a_binary_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(count_features):
        counts = df[col].value_counts().sort_index()
        counts.plot(kind='bar', ax=axes[idx], color=['steelblue', 'coral', 'teal'][idx], edgecolor='black')
        axes[idx].set_xlabel(col.replace('_', ' '), fontsize=11)
        axes[idx].set_ylabel('frequency', fontsize=11)
        axes[idx].set_title(f'distribution of {col.replace("_", " ")}', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05b_count_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    numerical_cols = ['item_price', 'price_per_ounce', 'price_per_unit', 
                      'unit_size', 'size_in_oz', 'pack_size', 'total_size',
                      'has_premium_word', 'premium_word_count',
                      'has_health_word', 'health_word_count',
                      'has_ingredients', 'ingredient_word_count']
    
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    plt.title('correlation matrix - numerical features', fontsize=16, pad=20)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    price_corr = corr_matrix['item_price'].drop('item_price').sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in price_corr.values]
    price_corr.plot(kind='barh', color=colors, edgecolor='black')
    plt.xlabel('correlation with item price', fontsize=12)
    plt.ylabel('feature', fontsize=12)
    plt.title('feature correlations with target (item price)', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06b_correlation_with_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    price_limit = df['item_price'].quantile(0.95)
    df_zoomed = df[df['item_price'] <= price_limit]
    
    plt.figure(figsize=(14, 6))
    category_order = df_zoomed.groupby('category')['item_price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_zoomed, x='category', y='item_price', order=category_order, palette='viridis')
    plt.xlabel('category', fontsize=12)
    plt.ylabel('item price ($)', fontsize=12)
    plt.title('item price distribution by category', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07a_price_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df_zoomed, x='brand_popularity', y='item_price', order=['low', 'mid', 'high'], 
                palette=['#ff6b6b', '#ffd93d', '#6bcb77'], ax=axes[0])
    axes[0].set_xlabel('brand popularity', fontsize=12)
    axes[0].set_ylabel('item price ($)', fontsize=12)
    axes[0].set_title('item price by brand popularity', fontsize=14)
    
    pop_mean = df.groupby('brand_popularity')['item_price'].mean().reindex(['low', 'mid', 'high'])
    pop_mean.plot(kind='bar', ax=axes[1], color=['#ff6b6b', '#ffd93d', '#6bcb77'], edgecolor='black')
    axes[1].set_xlabel('brand popularity', fontsize=12)
    axes[1].set_ylabel('average item price ($)', fontsize=12)
    axes[1].set_title('average price by brand popularity', fontsize=14)
    axes[1].tick_params(axis='x', rotation=0)
    for i, v in enumerate(pop_mean.values):
        axes[1].text(i, v + 0.5, f'${v:.2f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07b_price_by_brand_popularity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, col in enumerate(binary_features):
        feature_price = df.groupby(col)['item_price'].mean()
        feature_price.index = ['no', 'yes']
        feature_price.plot(kind='bar', ax=axes[idx], color=['#ff6b6b', '#6bcb77'], edgecolor='black')
        axes[idx].set_xlabel(col.replace('_', ' '), fontsize=11)
        axes[idx].set_ylabel('average price ($)', fontsize=11)
        axes[idx].set_title(f'avg price by {col.replace("_", " ")}', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=0)
        for i, v in enumerate(feature_price.values):
            axes[idx].text(i, v + 0.3, f'${v:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07c_price_by_text_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, col in enumerate(count_features):
        word_price = df.groupby(col)['item_price'].mean()
        word_price.plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
        axes[idx].set_xlabel(col.replace('_', ' '), fontsize=11)
        axes[idx].set_ylabel('average price ($)', fontsize=11)
        axes[idx].set_title(f'avg price by {col.replace("_", " ")}', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=0)
        axes[idx].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07d_price_by_word_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    scatter_features = ['unit_size', 'size_in_oz', 'pack_size', 'total_size']
    colors = ['steelblue', 'coral', 'teal', 'purple']
    
    for idx, col in enumerate(scatter_features):
        row, c = idx // 2, idx % 2
        x_limit = df[col].quantile(0.95)
        y_limit = df['item_price'].quantile(0.95)
        mask = (df[col] <= x_limit) & (df['item_price'] <= y_limit)
        plot_df = df[mask]
        
        axes[row, c].scatter(plot_df[col], plot_df['item_price'], alpha=0.3, s=15, c=colors[idx])
        axes[row, c].set_xlabel(col.replace('_', ' '), fontsize=12)
        axes[row, c].set_ylabel('item price ($)', fontsize=12)
        axes[row, c].set_title(f'item price vs {col.replace("_", " ")}', fontsize=14)
        axes[row, c].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_scatter_size_vs_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    outlier_cols = ['item_price', 'price_per_ounce', 'price_per_unit', 'total_size']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for idx, col in enumerate(outlier_cols):
        limit = df[col].quantile(0.95)
        zoomed_data = df[df[col] <= limit][col]
        bp = axes[idx].boxplot(zoomed_data.dropna(), vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[idx].set_ylabel(col.replace('_', ' '), fontsize=10)
        axes[idx].set_title(f'{col.replace("_", " ")}', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle('outlier detection - box plots', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    pair_features = ['item_price', 'total_size', 'pack_size', 'brand_popularity']
    price_limit = df['item_price'].quantile(0.95)
    total_limit = df['total_size'].quantile(0.95)
    pack_limit = df['pack_size'].quantile(0.95)
    
    df_pair = df[(df['item_price'] <= price_limit) & (df['total_size'] <= total_limit) & (df['pack_size'] <= pack_limit)][pair_features]
    sample_df = df_pair.sample(min(1000, len(df_pair)), random_state=42)
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(sample_df, hue='brand_popularity', palette={'low': '#ff6b6b', 'mid': '#ffd93d', 'high': '#6bcb77'},
                 diag_kind='kde', plot_kws={'alpha': 0.5, 's': 30})
    plt.suptitle('pair plot - key features by brand popularity', fontsize=14, y=1.02)
    plt.savefig(f'{output_dir}/10_pair_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    train_path = '/Users/ayalabnine/Downloads/EECE5644/Datasets/Main dataset/train.csv'
    test_path = '/Users/ayalabnine/Downloads/EECE5644/Datasets/Main dataset/test.csv'
    output_dir = '/Users/ayalabnine/Downloads/EECE5644/Datasets/Final Dataset/'
    eda_output_dir = '/Users/ayalabnine/Downloads/EECE5644/Datasets/Eda dataset'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print('processing train data...')
    df_train = pd.read_csv(train_path)
    
    if remove_outliers and 'price' in df_train.columns:
        df_train = remove_price_outliers(df_train, price_column='price', lower_limit=price_lower_limit, upper_limit=price_upper_limit)
    
    df_train_features = extract_features(df_train, dataset_type='train')
    output_path = output_dir + 'train_features_v2.csv'
    df_train_features.to_csv(output_path, index=False)
    print(f'saved train features: {len(df_train_features)} rows')
    
    print('processing test data...')
    df_test = pd.read_csv(test_path)
    df_test_features = extract_features(df_test, dataset_type='test')
    output_path = output_dir + 'test_features_v2.csv'
    df_test_features.to_csv(output_path, index=False)
    print(f'saved test features: {len(df_test_features)} rows')
    
    print('running eda...')
    run_eda(output_dir + 'train_features_v2.csv', eda_output_dir)
    print('done')

if __name__ == "__main__":
    main()
