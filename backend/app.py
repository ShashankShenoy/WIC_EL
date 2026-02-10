"""
FastAPI Backend for Carbon-Aware Cloud Scheduler
Professional REST API with CORS support
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import json
from io import BytesIO
import uuid

from core.forecasting import forecast_future_slots
from core.optimization import run_optimization, CloudSchedulingProblem
from core.analytics import calculate_metrics, calculate_baseline_schedule
from core.utils import validate_dataframe, add_confidence_intervals, generate_cost_data

app = FastAPI(
    title="Carbon-Aware Cloud Scheduler API",
    description="Multi-objective optimization for sustainable cloud computing",
    version="2.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with Redis/Database in production)
sessions = {}

# Pydantic models for request/response
class OptimizationConfig(BaseModel):
    n_jobs: int = 20
    forecast_days: int = 3
    confidence_level: int = 95
    weight_carbon: float = 0.5
    weight_renewable: float = 0.3
    weight_cost: float = 0.2
    weight_load: float = 0.15
    pop_size: int = 100
    n_gen: int = 200
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    avg_job_power_kw: float = 0.5
    avg_job_duration_min: int = 5

class ForecastResponse(BaseModel):
    status: str
    num_slots: int
    time_range: Dict[str, str]
    regions: List[str]
    avg_carbon: float
    forecast_data: List[Dict[str, Any]]

class OptimizationResponse(BaseModel):
    status: str
    best_solution: Dict[str, Any]
    pareto_solutions: List[Dict[str, Any]]
    convergence: List[Dict[str, float]]
    schedule: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Carbon-Aware Cloud Scheduler",
        "version": "2.0.0"
    }

@app.post("/api/upload")
async def upload_data(
    historical_csv: UploadFile = File(...),
    trained_model: UploadFile = File(...)
):
    """
    Upload historical data and trained models
    Returns session_id for subsequent API calls
    """
    try:
        # Read CSV
        csv_content = await historical_csv.read()
        df_historical = pd.read_csv(BytesIO(csv_content), parse_dates=['timestamp'])
        
        # Validate
        ok, msg = validate_dataframe(df_historical)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        
        # Read model
        pkl_content = await trained_model.read()
        models = pickle.loads(pkl_content)
        
        # Generate unique session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        sessions[session_id] = {
            "df_historical": df_historical,
            "models": models,
            "df_future": None,
            "results": None
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "records": len(df_historical),
            "regions": df_historical['region'].unique().tolist(),
            "time_range": {
                "start": df_historical['timestamp'].min().isoformat(),
                "end": df_historical['timestamp'].max().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/forecast/{session_id}", response_model=ForecastResponse)
async def generate_forecast(session_id: str, config: OptimizationConfig):
    """Generate future carbon intensity forecasts"""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404, 
            detail=f"Session not found. Server may have restarted. Please upload your files again."
        )
    
    try:
        session = sessions[session_id]
        df_historical = session["df_historical"]
        models = session["models"]
        
        # Generate forecast
        df_future = forecast_future_slots(
            df_historical=df_historical,
            models=models,
            forecast_days=config.forecast_days,
            confidence_level=config.confidence_level
        )
        
        # Add cost data
        df_future = generate_cost_data(df_future)
        
        session["df_future"] = df_future
        
        return ForecastResponse(
            status="success",
            num_slots=len(df_future),
            time_range={
                "start": df_future['timestamp'].min().isoformat(),
                "end": df_future['timestamp'].max().isoformat()
            },
            regions=df_future['region'].unique().tolist(),
            avg_carbon=float(df_future['carbon_intensity'].mean()),
            forecast_data=df_future.head(100).to_dict('records')  # Limit for performance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/api/optimize/{session_id}", response_model=OptimizationResponse)
async def optimize_schedule(session_id: str, config: OptimizationConfig):
    """Run multi-objective optimization"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if session["df_future"] is None:
        raise HTTPException(status_code=400, detail="Generate forecast first")
    
    try:
        df_future = session["df_future"]
        
        # Run optimization
        results = run_optimization(
            df_future=df_future,
            n_jobs=config.n_jobs,
            weight_carbon=config.weight_carbon,
            weight_renewable=config.weight_renewable,
            weight_cost=config.weight_cost,
            weight_load=config.weight_load,
            pop_size=config.pop_size,
            n_gen=config.n_gen,
            crossover_prob=config.crossover_prob,
            mutation_prob=config.mutation_prob,
            crossover_eta=config.crossover_eta,
            mutation_eta=config.mutation_eta,
            avg_job_power_kw=config.avg_job_power_kw,
            avg_job_duration_min=config.avg_job_duration_min
        )
        
        session["results"] = results
        
        # Calculate metrics
        baseline = calculate_baseline_schedule(df_future, config.n_jobs)
        metrics = calculate_metrics(
            optimized=results["schedule"],
            baseline=baseline,
            job_power_kw=config.avg_job_power_kw,
            job_duration_min=config.avg_job_duration_min
        )
        
        return OptimizationResponse(
            status="success",
            best_solution=results["best_solution"],
            pareto_solutions=results["pareto_solutions"][:50],  # Limit for performance
            convergence=results["convergence"],
            schedule=results["schedule"].to_dict('records'),
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/api/analytics/{session_id}")
async def get_analytics(session_id: str):
    """Get detailed analytics and insights"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if session["results"] is None:
        raise HTTPException(status_code=400, detail="Run optimization first")
    
    try:
        schedule = session["results"]["schedule"]
        df_future = session["df_future"]
        
        # Regional analysis
        regional_agg = schedule.groupby('region').agg({
            'carbon_intensity': ['mean', 'std', 'min', 'max'],
            'job_id': 'count'
        })
        regional_agg.columns = ['_'.join(col).strip() for col in regional_agg.columns.values]
        regional_stats = regional_agg.reset_index().to_dict('records')
        
        # Temporal analysis
        schedule_copy = schedule.copy()
        schedule_copy['hour'] = pd.to_datetime(schedule_copy['timestamp']).dt.hour
        hourly_stats = schedule_copy.groupby('hour')['carbon_intensity'].mean().to_dict()
        
        # Best/worst insights
        best_region = schedule.groupby('region')['carbon_intensity'].mean().idxmin()
        worst_region = schedule.groupby('region')['carbon_intensity'].mean().idxmax()
        peak_hour = schedule_copy.groupby('hour')['carbon_intensity'].mean().idxmin()
        
        return {
            "regional_stats": regional_stats,
            "hourly_stats": {str(k): float(v) for k, v in hourly_stats.items()},
            "insights": {
                "best_region": str(best_region),
                "worst_region": str(worst_region),
                "optimal_hour": int(peak_hour)
            },
            "timeline_data": schedule[['timestamp', 'region', 'carbon_intensity', 'job_id']].head(100).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.get("/api/export/{session_id}")
async def export_schedule(session_id: str, format: str = "json"):
    """Export optimized schedule"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if session["results"] is None:
        raise HTTPException(status_code=400, detail="Run optimization first")
    
    schedule = session["results"]["schedule"]
    
    if format == "json":
        return schedule.to_dict('records')
    elif format == "csv":
        from fastapi.responses import StreamingResponse
        buffer = BytesIO()
        schedule.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=schedule.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
