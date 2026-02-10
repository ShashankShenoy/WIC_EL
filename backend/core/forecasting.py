"""Forecasting module for carbon intensity and renewable energy"""
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

def forecast_future_slots(df_historical, models, forecast_days, confidence_level):
    """
    Generate future carbon intensity forecasts
    
    Args:
        df_historical: Historical data with timestamp, region, and features
        models: Dict of trained models per region and target
        forecast_days: Number of days to forecast
        confidence_level: Confidence interval percentage
        
    Returns:
        DataFrame with forecasted slots
    """
    # Preprocess historical data
    df_historical = df_historical.sort_values(['region', 'timestamp']).reset_index(drop=True)
    df_historical[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']] = (
        df_historical.groupby('region')[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']].ffill()
    )
    
    # Feature engineering
    df_historical['hour'] = df_historical['timestamp'].dt.hour
    df_historical['day_of_week'] = df_historical['timestamp'].dt.dayofweek
    df_historical['month'] = df_historical['timestamp'].dt.month
    df_historical['day_of_year'] = df_historical['timestamp'].dt.dayofyear
    
    for col in ['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']:
        df_historical[f'{col}_lag1'] = df_historical.groupby('region')[col].shift(1)
    
    df_historical = df_historical.dropna().reset_index(drop=True)
    
    feature_cols = [
        'hour', 'day_of_week', 'month', 'day_of_year',
        'carbon_intensity_lag1', 'solar_cloud_pct_lag1',
        'wind_speed_lag1', 'temperature_lag1'
    ]
    target_cols = ['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']
    
    # Generate future timestamps
    regions = df_historical['region'].unique()
    last_timestamp = df_historical['timestamp'].max()
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(minutes=5),
        periods=forecast_days * 288,  # 5-min slots per day
        freq='5min'
    )
    
    future_data = []
    for region in regions:
        if region not in models:
            continue
        
        last_values = df_historical[df_historical['region'] == region].iloc[-1]
        
        for ts in future_timestamps:
            features = {
                'timestamp': ts,
                'region': region,
                'hour': ts.hour,
                'day_of_week': ts.dayofweek,
                'month': ts.month,
                'day_of_year': ts.dayofyear,
                'carbon_intensity_lag1': last_values['carbon_intensity'],
                'solar_cloud_pct_lag1': last_values['solar_cloud_pct'],
                'wind_speed_lag1': last_values['wind_speed'],
                'temperature_lag1': last_values['temperature']
            }
            
            # Predict each target
            for target in target_cols:
                X_pred = pd.DataFrame([features])[feature_cols]
                prediction = models[region][target].predict(X_pred)[0]
                features[target] = float(prediction)
            
            future_data.append(features)
            last_values = pd.Series(features)
    
    df_future = pd.DataFrame(future_data)
    
    # Normalize predictions
    scaler = MinMaxScaler()
    scaler.fit(df_historical[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']])
    df_future[['carbon_norm', 'solar_norm', 'wind_norm', 'temp_norm']] = scaler.transform(
        df_future[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']]
    )
    df_future['renewable_norm'] = (df_future['solar_norm'] + df_future['wind_norm']) / 2
    
    df_future['slot_index'] = df_future.index
    
    return df_future
