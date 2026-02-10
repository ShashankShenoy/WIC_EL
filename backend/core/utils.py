"""Utility functions for data validation and preprocessing"""
import pandas as pd
import numpy as np

def validate_dataframe(df):
    """Validate historical dataframe has required columns"""
    required = {'timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature'}
    missing = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"

def add_confidence_intervals(df_future, df_historical, target_cols, confidence_level):
    """Add confidence intervals to forecasts"""
    z_score = 1.96 if confidence_level == 95 else 2.576  # 95% or 99%
    
    for target in target_cols:
        std = df_historical[target].std()
        df_future[f'{target}_lower'] = df_future[target] - z_score * std
        df_future[f'{target}_upper'] = df_future[target] + z_score * std
    
    return df_future

def generate_cost_data(df):
    """Generate electricity cost estimates based on time and carbon"""
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # Base cost varies by region
    region_base_costs = {
        'IN-WE': 0.08,
        'IN-SO': 0.07,
        'NL': 0.12,
    }
    df['base_cost'] = df['region'].map(region_base_costs).fillna(0.10)
    
    # Time-of-day multiplier
    peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    df['time_multiplier'] = df['hour'].apply(lambda h: 1.5 if h in peak_hours else 1.0)
    
    # Calculate final cost ($/kWh)
    df['electricity_cost'] = df['base_cost'] * df['time_multiplier'] * (1 + df['carbon_intensity']/1000)
    
    return df
