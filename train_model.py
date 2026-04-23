"""
Dubai Property Rental Price Prediction - Model Training Script
================================================================
Trains a robust ML model to predict rental prices of properties in UAE.

Features used:
  - Beds (number of bedrooms)
  - Baths (number of bathrooms)
  - Type (Apartment, Villa, Townhouse, etc.)
  - Area_in_sqft (property area)
  - Furnishing (Furnished, Unfurnished, etc.)
  - Location (specific area)
  - City (Abu Dhabi, Dubai, etc.)

Pipeline:
  1. Load & clean data
  2. Feature engineering
  3. Encode categorical variables
  4. Scale numerical features
  5. Train & compare models (Linear, KNN, RandomForest, GradientBoosting)
  6. Save best model + artifacts to pickle
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv("dubai_properties.csv")
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================
# 2. Data Cleaning
# ============================================================
# Keep only relevant columns
keep_cols = ['Rent', 'Beds', 'Baths', 'Type', 'Area_in_sqft', 'Furnishing', 'Location', 'City']
df = df[keep_cols].copy()

# Drop rows with missing values in key columns
df = df.dropna(subset=keep_cols)
print(f"\nAfter dropping NaN: {df.shape[0]} rows")

# Ensure numeric types
df['Rent'] = pd.to_numeric(df['Rent'], errors='coerce')
df['Beds'] = pd.to_numeric(df['Beds'], errors='coerce')
df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce')
df['Area_in_sqft'] = pd.to_numeric(df['Area_in_sqft'], errors='coerce')

df = df.dropna()
print(f"After numeric conversion cleanup: {df.shape[0]} rows")

# ============================================================
# 3. Outlier Removal
# ============================================================
# Remove extreme rent outliers (top 1% and bottom 0.5%)
q_low = df['Rent'].quantile(0.005)
q_high = df['Rent'].quantile(0.99)
df = df[(df['Rent'] >= q_low) & (df['Rent'] <= q_high)]
print(f"After rent outlier removal ({q_low:.0f} - {q_high:.0f}): {df.shape[0]} rows")

# Remove extreme area outliers
q_area_high = df['Area_in_sqft'].quantile(0.99)
df = df[(df['Area_in_sqft'] > 0) & (df['Area_in_sqft'] <= q_area_high)]
print(f"After area outlier removal: {df.shape[0]} rows")

# Cap beds and baths at reasonable values
df = df[df['Beds'] <= 12]
df = df[df['Baths'] <= 15]
print(f"After beds/baths cap: {df.shape[0]} rows")

# ============================================================
# 4. Feature Engineering
# ============================================================
# Clean Type - group rare types
type_counts = df['Type'].value_counts()
common_types = type_counts[type_counts >= 50].index.tolist()
df['Type_clean'] = df['Type'].apply(lambda x: x if x in common_types else 'Other')

# Clean Furnishing
furnishing_map = {
    'Unfurnished': 'Unfurnished',
    'Furnished': 'Furnished',
    'Partly Furnished': 'Partly Furnished'
}
df['Furnishing_clean'] = df['Furnishing'].map(furnishing_map).fillna('Unfurnished')

# Clean Location - keep top N locations, rest as 'Other'
location_counts = df['Location'].value_counts()
top_locations = location_counts[location_counts >= 100].index.tolist()
df['Location_clean'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')

# Clean City
city_counts = df['City'].value_counts()
top_cities = city_counts[city_counts >= 50].index.tolist()
df['City_clean'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')

print(f"\nUnique property types (cleaned): {df['Type_clean'].nunique()}")
print(f"Unique furnishing types: {df['Furnishing_clean'].nunique()}")
print(f"Unique locations (cleaned): {df['Location_clean'].nunique()}")
print(f"Unique cities (cleaned): {df['City_clean'].nunique()}")

# ============================================================
# 5. Encode Categorical Variables
# ============================================================
# Type encoding
type_encoder = LabelEncoder()
df['type_encoded'] = type_encoder.fit_transform(df['Type_clean'])
type_list = list(type_encoder.classes_)

# Furnishing encoding
furnishing_encoder = LabelEncoder()
df['furnishing_encoded'] = furnishing_encoder.fit_transform(df['Furnishing_clean'])
furnishing_list = list(furnishing_encoder.classes_)

# Location encoding
location_encoder = LabelEncoder()
df['location_encoded'] = location_encoder.fit_transform(df['Location_clean'])
location_list = list(location_encoder.classes_)

# City encoding
city_encoder = LabelEncoder()
df['city_encoded'] = city_encoder.fit_transform(df['City_clean'])
city_list = list(city_encoder.classes_)

print(f"\nType classes: {type_list}")
print(f"Furnishing classes: {furnishing_list}")
print(f"Location classes ({len(location_list)}): {location_list[:10]}...")
print(f"City classes: {city_list}")

# ============================================================
# 6. Prepare Feature Matrix
# ============================================================
feature_columns = [
    'Beds',
    'Baths',
    'type_encoded',
    'Area_in_sqft',
    'furnishing_encoded',
    'location_encoded',
    'city_encoded',
]

X = df[feature_columns].copy()
Y = df['Rent'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nRent statistics (AED/year):")
print(Y.describe())

# ============================================================
# 7. Scale Features
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 8. Train/Test Split
# ============================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ============================================================
# 9. Train and Compare Models
# ============================================================
models = {
    'LinearRegression': LinearRegression(),
    'KNN-5': KNeighborsRegressor(n_neighbors=5),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, min_samples_leaf=3, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, min_samples_leaf=5
    ),
}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

best_model = None
best_score = -999
best_name = ""

for name, model in models.items():
    model.fit(X_train, Y_train)
    
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    
    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    # Cross validation
    cv_scores = cross_val_score(model, X_scaled, Y, cv=5, scoring='r2')
    
    print(f"\n{name}:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²:  {test_score:.4f}")
    print(f"  CV R²:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  MAE:      {mae:,.0f} AED")
    print(f"  RMSE:     {rmse:,.0f} AED")
    
    if test_score > best_score:
        best_score = test_score
        best_model = model
        best_name = name

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_name} (Test R² = {best_score:.4f})")
print(f"{'=' * 60}")

# ============================================================
# 10. Retrain Best Model on Full Data
# ============================================================
best_model.fit(X_scaled, Y)
final_train_score = best_model.score(X_scaled, Y)
print(f"\nFinal model trained on full data - R²: {final_train_score:.4f}")

# ============================================================
# 11. Save Model Artifacts to Pickle
# ============================================================
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'type_list': type_list,
    'furnishing_list': furnishing_list,
    'location_list': location_list,
    'city_list': city_list,
    'type_encoder': type_encoder,
    'furnishing_encoder': furnishing_encoder,
    'location_encoder': location_encoder,
    'city_encoder': city_encoder,
    'model_name': best_name,
    'r2_score': best_score,
    'num_samples': len(df),
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\n✅ Model saved to model.pkl")
print(f"   Model: {best_name}")
print(f"   R² Score: {best_score:.4f}")
print(f"   Features: {feature_columns}")
print(f"   Types: {type_list}")
print(f"   Furnishing: {furnishing_list}")
print(f"   Locations: {len(location_list)} areas")
print(f"   Cities: {city_list}")

# ============================================================
# 12. Sanity Check Predictions
# ============================================================
print("\n" + "=" * 60)
print("SANITY CHECK - Sample Predictions")
print("=" * 60)

test_cases = [
    {
        'desc': '2-Bed Apartment, 1200 sqft, Unfurnished, Dubai Marina, Dubai',
        'beds': 2, 'baths': 3, 'type': 'Apartment', 'area': 1200,
        'furnishing': 'Unfurnished', 'location': 'Dubai Marina', 'city': 'Dubai'
    },
    {
        'desc': '3-Bed Villa, 3500 sqft, Furnished, Jumeirah, Dubai',
        'beds': 3, 'baths': 4, 'type': 'Villa', 'area': 3500,
        'furnishing': 'Furnished', 'location': 'Jumeirah Village Circle (JVC)', 'city': 'Dubai'
    },
    {
        'desc': '1-Bed Apartment, 800 sqft, Furnished, Al Reem Island, Abu Dhabi',
        'beds': 1, 'baths': 2, 'type': 'Apartment', 'area': 800,
        'furnishing': 'Furnished', 'location': 'Al Reem Island', 'city': 'Abu Dhabi'
    },
    {
        'desc': 'Studio Apartment, 450 sqft, Unfurnished, Business Bay, Dubai',
        'beds': 0, 'baths': 1, 'type': 'Apartment', 'area': 450,
        'furnishing': 'Unfurnished', 'location': 'Business Bay', 'city': 'Dubai'
    },
]

for tc in test_cases:
    # Encode
    type_enc = type_list.index(tc['type']) if tc['type'] in type_list else type_list.index('Other')
    furn_enc = furnishing_list.index(tc['furnishing']) if tc['furnishing'] in furnishing_list else 0
    loc_enc = location_list.index(tc['location']) if tc['location'] in location_list else location_list.index('Other')
    city_enc = city_list.index(tc['city']) if tc['city'] in city_list else city_list.index('Other')
    
    features = [[
        tc['beds'],
        tc['baths'],
        type_enc,
        tc['area'],
        furn_enc,
        loc_enc,
        city_enc,
    ]]
    features_scaled = scaler.transform(features)
    pred = best_model.predict(features_scaled)[0]
    pred = max(pred, 0)
    print(f"\n{tc['desc']}")
    print(f"  Predicted Rent: AED {pred:,.0f}/year  |  AED {pred/12:,.0f}/month")
