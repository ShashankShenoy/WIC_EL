"""Analytics and metrics calculation"""
import pandas as pd
import numpy as np

def calculate_baseline_schedule(df_future, n_jobs):
    """Generate naive baseline schedule (first n slots)"""
    baseline_slots = df_future.head(n_jobs)[['timestamp', 'region', 'carbon_intensity', 
                                              'solar_cloud_pct', 'wind_speed', 'temperature']].copy()
    baseline_slots['job_id'] = range(1, n_jobs + 1)
    return baseline_slots

def calculate_metrics(optimized, baseline, job_power_kw, job_duration_min):
    """
    Calculate comprehensive metrics comparing optimized vs baseline
    
    Args:
        optimized: Optimized schedule DataFrame
        baseline: Baseline schedule DataFrame
        job_power_kw: Power consumption per job
        job_duration_min: Duration per job
        
    Returns:
        dict with all metrics
    """
    job_energy_kwh = job_power_kw * (job_duration_min / 60.0)
    
    # Carbon metrics
    baseline_carbon_avg = baseline['carbon_intensity'].mean()
    optimized_carbon_avg = optimized['carbon_intensity'].mean()
    carbon_reduction_pct = ((baseline_carbon_avg - optimized_carbon_avg) / baseline_carbon_avg) * 100 if baseline_carbon_avg != 0 else 0.0
    
    # Total emissions
    total_carbon_baseline_g = (baseline['carbon_intensity'] * job_energy_kwh).sum()
    total_carbon_optimized_g = (optimized['carbon_intensity'] * job_energy_kwh).sum()
    carbon_saved_kg = (total_carbon_baseline_g - total_carbon_optimized_g) / 1000
    
    # Renewable metrics
    baseline_solar = baseline['solar_cloud_pct'].mean()
    optimized_solar = optimized['solar_cloud_pct'].mean()
    solar_improvement_pct = ((optimized_solar - baseline_solar) / baseline_solar) * 100 if baseline_solar != 0 else 0.0
    
    baseline_wind = baseline['wind_speed'].mean()
    optimized_wind = optimized['wind_speed'].mean()
    wind_improvement_pct = ((optimized_wind - baseline_wind) / baseline_wind) * 100 if baseline_wind != 0 else 0.0
    
    # Regional distribution
    optimized_regions = optimized['region'].nunique()
    baseline_regions = baseline['region'].nunique()
    
    # Variability
    baseline_carbon_std = baseline['carbon_intensity'].std()
    optimized_carbon_std = optimized['carbon_intensity'].std()
    consistency_improvement_pct = ((baseline_carbon_std - optimized_carbon_std) / baseline_carbon_std) * 100 if baseline_carbon_std != 0 else 0.0
    
    return {
        'carbon': {
            'baseline_avg': float(baseline_carbon_avg),
            'optimized_avg': float(optimized_carbon_avg),
            'reduction_pct': float(carbon_reduction_pct),
            'carbon_saved_kg': float(carbon_saved_kg),
            'baseline_total_kg': float(total_carbon_baseline_g / 1000),
            'optimized_total_kg': float(total_carbon_optimized_g / 1000)
        },
        'renewables': {
            'baseline_solar': float(baseline_solar),
            'optimized_solar': float(optimized_solar),
            'solar_improvement_pct': float(solar_improvement_pct),
            'baseline_wind': float(baseline_wind),
            'optimized_wind': float(optimized_wind),
            'wind_improvement_pct': float(wind_improvement_pct)
        },
        'distribution': {
            'optimized_regions': int(optimized_regions),
            'baseline_regions': int(baseline_regions),
            'regional_diversity_change': int(optimized_regions - baseline_regions)
        },
        'consistency': {
            'baseline_std': float(baseline_carbon_std),
            'optimized_std': float(optimized_carbon_std),
            'improvement_pct': float(consistency_improvement_pct)
        },
        'environmental_impact': {
            'trees_equivalent': float(carbon_saved_kg * 0.00022),  # Trees to absorb this CO2/year
            'cost_savings_usd': float(carbon_saved_kg * 0.05)  # At $50/ton carbon price
        }
    }
