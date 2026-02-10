"""
Sample Data Generator for Cloud Job Scheduler

This script generates realistic synthetic data for testing the optimization system.
Includes:
- Historical energy data with realistic patterns
- Trained ML models (RandomForest)
- Regional variations and temporal patterns

Usage:
    python generate_sample_data.py
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configuration
REGIONS = ['US-West-1', 'US-East-1', 'EU-Central-1', 'Asia-Pacific-1', 'EU-North-1']
START_DATE = datetime(2024, 1, 1)
DAYS_HISTORICAL = 90  # 3 months
INTERVAL_MINUTES = 5

print("=" * 60)
print("Cloud Job Scheduler - Sample Data Generator")
print("=" * 60)

# Generate timestamps
n_slots = DAYS_HISTORICAL * 24 * (60 // INTERVAL_MINUTES)
timestamps = [START_DATE + timedelta(minutes=i * INTERVAL_MINUTES) for i in range(n_slots)]

print(f"\nGenerating {n_slots:,} time slots over {DAYS_HISTORICAL} days...")

# Generate data for each region
data = []

for region in REGIONS:
    print(f"Processing {region}...")
    
    # Regional characteristics
    if 'West' in region or 'Asia' in region:
        base_solar = 60  # High solar potential
        base_wind = 30
        base_carbon = 200  # Low carbon (more renewables)
    elif 'North' in region:
        base_solar = 25  # Low solar
        base_wind = 70  # High wind
        base_carbon = 150
    else:
        base_solar = 40
        base_wind = 45
        base_carbon = 300
    
    for ts in timestamps:
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday
        day_of_week = ts.weekday()
        
        # Solar pattern (peaks at noon, zero at night)
        solar_pattern = max(0, np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
        solar_seasonal = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)  # Summer peak
        solar = base_solar * solar_pattern * solar_seasonal
        solar += np.random.normal(0, 5)  # Noise
        solar = np.clip(solar, 0, 100)
        
        # Wind pattern (more consistent, but varies)
        wind_pattern = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)
        wind_seasonal = 1 + 0.2 * np.cos(2 * np.pi * day_of_year / 365)  # Winter peak
        wind = base_wind * wind_pattern * wind_seasonal
        wind += np.random.normal(0, 8)
        wind = np.clip(wind, 0, 100)
        
        # Temperature (affects energy demand)
        temp_pattern = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Seasonal
        temp_pattern += 5 * np.sin(2 * np.pi * hour / 24)  # Daily variation
        temp = temp_pattern + np.random.normal(0, 2)
        
        # Carbon intensity (inversely related to renewables, peaks during high demand)
        renewable_factor = (solar + wind) / 200
        demand_factor = 1.5 if 9 <= hour <= 21 else 0.8  # High during day
        carbon = base_carbon * (1 - 0.4 * renewable_factor) * demand_factor
        carbon += np.random.normal(0, 20)
        carbon = np.clip(carbon, 50, 800)
        
        data.append({
            'timestamp': ts,
            'region': region,
            'carbon_intensity': round(carbon, 2),
            'solar_cloud_pct': round(solar, 2),
            'wind_speed': round(wind, 2),
            'temperature': round(temp, 2)
        })

# Create DataFrame
df = pd.DataFrame(data)

# Feature engineering
print("\nEngineering features...")
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Lag features
for col in ['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']:
    df[f'{col}_lag1'] = df.groupby('region')[col].shift(1)

# Drop rows with NaN
df = df.dropna().reset_index(drop=True)

print(f"Final dataset: {len(df):,} rows with {len(df.columns)} columns")

# Save historical data
output_csv = 'energy_data_historical_sample.csv'
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved historical data: {output_csv}")

# Train models for each region and target
print("\nTraining ML models...")

feature_cols = [
    'hour', 'day_of_week', 'month', 'day_of_year',
    'carbon_intensity_lag1', 'solar_cloud_pct_lag1',
    'wind_speed_lag1', 'temperature_lag1'
]
target_cols = ['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']

models = {}

for region in REGIONS:
    print(f"  Training models for {region}...")
    region_data = df[df['region'] == region]
    
    models[region] = {}
    
    for target in target_cols:
        X = region_data[feature_cols]
        y = region_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        models[region][target] = model
        
        print(f"    {target:20s}: R² train={train_score:.3f}, test={test_score:.3f}")

# Save models
output_pkl = 'trained_models_sample.pkl'
with open(output_pkl, 'wb') as f:
    pickle.dump(models, f)

print(f"\n✅ Saved trained models: {output_pkl}")

# Generate summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

summary = df.groupby('region')[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']].agg(['mean', 'std'])
print("\nMean values by region:")
print(summary['carbon_intensity']['mean'].to_string())

print("\n" + "=" * 60)
print("✨ Sample data generation complete!")
print("=" * 60)
print(f"\nGenerated files:")
print(f"  1. {output_csv} ({len(df):,} rows)")
print(f"  2. {output_pkl} ({len(REGIONS)} regions × {len(target_cols)} targets)")
print(f"\nYou can now use these files in the Streamlit app:")
print(f"  1. Upload '{output_csv}' as historical data")
print(f"  2. Upload '{output_pkl}' as trained models")
print(f"  3. Configure parameters and run optimization")
print("\n🚀 Ready to optimize!")
