import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

st.set_page_config(
    page_title="Cloud Job Scheduler — Forecast & Optimize",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Cloud Job Scheduler — Forecasting & NSGA-II Optimization")
st.markdown("""
This app predicts future carbon intensity, solar/cloud cover, wind speed and temperature per region (using your trained `.pkl` models)
and then schedules cloud jobs across predicted time slots using a multi-objective NSGA-II optimization.
""")

# ---------------------------
# Cached functions for performance
# ---------------------------
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and preprocess historical data with caching"""
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    
    # Basic preprocessing and forward-fill
    df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
    df[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']] = (
        df.groupby('region')[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']].ffill()
    )
    
    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    for col in ['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']:
        df[f'{col}_lag1'] = df.groupby('region')[col].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def load_models(uploaded_file):
    """Load ML models with caching"""
    return pickle.load(uploaded_file)

# ---------------------------
# Sidebar: Uploads & sliders
# ---------------------------
st.sidebar.header("1) Upload data & model")
uploaded_csv = st.sidebar.file_uploader("Upload historical CSV (must have 'timestamp' & 'region')", type=["csv"])
uploaded_pkl = st.sidebar.file_uploader("Upload trained models (.pkl)", type=["pkl"])

st.sidebar.header("2) Forecast & scheduling settings")
N_JOBS = st.sidebar.slider("Number of jobs to schedule", min_value=1, max_value=500, value=10, step=1)
forecast_days = st.sidebar.slider("Forecast days ahead", min_value=1, max_value=30, value=3, step=1)

st.sidebar.header("3) Objective weights")
weight_carbon = st.sidebar.slider("Weight — Carbon Intensity (higher → minimize carbon)", 0.0, 1.0, 0.7, 0.01)
weight_renewable = 1.0 - weight_carbon
st.sidebar.markdown(f"Renewable weight auto-set to **{weight_renewable:.2f}** (1 - carbon weight)")
weight_load = st.sidebar.slider("Weight — Load Balancing Penalty", 0.0, 1.0, 0.1, 0.01)

st.sidebar.header("4) NSGA-II hyperparameters")
pop_size = st.sidebar.slider("Population size", min_value=20, max_value=500, value=100, step=10)
n_gen = st.sidebar.slider("Generations", min_value=10, max_value=500, value=200, step=10)
crossover_prob = st.sidebar.slider("Crossover probability (SBX)", 0.0, 1.0, 0.9, 0.01)
mutation_prob = st.sidebar.slider("Mutation probability (PM)", 0.0, 1.0, 0.1, 0.01)

run_button = st.sidebar.button("🚀 Forecast & Optimize")

# ---------------------------
# Helper: validate uploads
# ---------------------------
def validate_df(df):
    required = {'timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature'}
    if not required.issubset(set(df.columns)):
        return False, f"CSV missing required columns. Required: {', '.join(sorted(required))}"
    return True, None

# ---------------------------
# Main flow
# ---------------------------
if run_button:
    # Validate uploads
    if uploaded_csv is None or uploaded_pkl is None:
        st.error("Please upload both the historical CSV and the trained `.pkl` model in the sidebar.")
    else:
        try:
            with st.spinner("Loading dataset and model..."):
                # Use cached loading functions
                df_historical = load_and_process_data(uploaded_csv)
                ok, msg = validate_df(df_historical)
                if not ok:
                    st.error(msg)
                    st.stop()

                # Load models: expected structure: models[region][target] = estimator
                models = load_models(uploaded_pkl)

                feature_cols = [
                    'hour', 'day_of_week', 'month', 'day_of_year',
                    'carbon_intensity_lag1', 'solar_cloud_pct_lag1',
                    'wind_speed_lag1', 'temperature_lag1'
                ]
                target_cols = ['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']

            st.success("Loaded data & model ✔")
        except Exception as e:
            st.error(f"Failed to load files: {e}")
            st.exception(e)
            st.stop()

        # Forecasting - OPTIMIZED with vectorization
        with st.spinner("Generating future timestamps and forecasting..."):
            regions = df_historical['region'].unique()
            last_timestamp = df_historical['timestamp'].max()
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(minutes=5),
                periods=forecast_days * 288,  # 5min slots per day
                freq='5min'
            )
            
            future_data = []
            progress_bar = st.progress(0)
            total_regions = len(regions)
            
            for idx, region in enumerate(regions):
                # If model doesn't have region, skip
                if region not in models:
                    st.warning(f"Region '{region}' not present in model file — skipping.")
                    continue

                last_values = df_historical[df_historical['region'] == region].iloc[-1]
                
                # Pre-allocate arrays for this region
                n_timestamps = len(future_timestamps)
                region_predictions = np.zeros((n_timestamps, len(target_cols)))
                
                # Create feature matrix for all timestamps at once
                feature_matrix = pd.DataFrame({
                    'hour': [ts.hour for ts in future_timestamps],
                    'day_of_week': [ts.dayofweek for ts in future_timestamps],
                    'month': [ts.month for ts in future_timestamps],
                    'day_of_year': [ts.dayofyear for ts in future_timestamps],
                    'carbon_intensity_lag1': float(last_values['carbon_intensity']),
                    'solar_cloud_pct_lag1': float(last_values['solar_cloud_pct']),
                    'wind_speed_lag1': float(last_values['wind_speed']),
                    'temperature_lag1': float(last_values['temperature'])
                })
                
                # Explicitly set lag columns to float to avoid FutureWarning
                feature_matrix = feature_matrix.astype({
                    'carbon_intensity_lag1': 'float64',
                    'solar_cloud_pct_lag1': 'float64',
                    'wind_speed_lag1': 'float64',
                    'temperature_lag1': 'float64'
                })
                
                # Predict iteratively but update lag features efficiently
                for i in range(n_timestamps):
                    for j, target in enumerate(target_cols):
                        pred = models[region][target].predict(feature_matrix.iloc[[i]])[0]
                        region_predictions[i, j] = pred
                    
                    # Update lag features for next iteration
                    if i < n_timestamps - 1:
                        feature_matrix.loc[i+1, 'carbon_intensity_lag1'] = region_predictions[i, 0]
                        feature_matrix.loc[i+1, 'solar_cloud_pct_lag1'] = region_predictions[i, 1]
                        feature_matrix.loc[i+1, 'wind_speed_lag1'] = region_predictions[i, 2]
                        feature_matrix.loc[i+1, 'temperature_lag1'] = region_predictions[i, 3]
                
                # Build result records
                for i, ts in enumerate(future_timestamps):
                    future_data.append({
                        'timestamp': ts,
                        'region': region,
                        'hour': ts.hour,
                        'day_of_week': ts.dayofweek,
                        'month': ts.month,
                        'day_of_year': ts.dayofyear,
                        'carbon_intensity': region_predictions[i, 0],
                        'solar_cloud_pct': region_predictions[i, 1],
                        'wind_speed': region_predictions[i, 2],
                        'temperature': region_predictions[i, 3]
                    })
                
                progress_bar.progress((idx + 1) / total_regions)

            if len(future_data) == 0:
                st.error("No future data generated — check that your model contains the regions present in CSV.")
                st.stop()

            df_future = pd.DataFrame(future_data)
            df_future = df_future.reset_index(drop=True)
            
            # Count regions and time slots
            n_regions = len(df_future['region'].unique())
            n_timeslots = len(future_timestamps)
            
        st.success(f"Generated **{len(df_future)} future slots** across **{n_regions} regions** × **{n_timeslots} time slots** ({df_future['timestamp'].min()} → {df_future['timestamp'].max()})")

        # Normalization with SOLAR INVERSION FIX
        with st.spinner("Normalizing predictions and computing renewable availability..."):
            scaler = MinMaxScaler()
            scaler.fit(df_historical[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']])
            
            # Normalize all features
            df_future[['carbon_norm', 'cloud_norm', 'wind_norm', 'temp_norm']] = scaler.transform(
                df_future[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']]
            )
            
            # ✅ FIX: Invert cloud cover to get solar availability
            # High cloud % = low solar availability
            # Low cloud % = high solar availability
            df_future['solar_norm'] = 1.0 - df_future['cloud_norm']
            
            # Compute renewable energy availability (solar + wind)
            df_future['renewable_norm'] = (df_future['solar_norm'] + df_future['wind_norm']) / 2.0
            
            # Also create actual solar availability column for display
            df_future['solar_availability_pct'] = 100.0 - df_future['solar_cloud_pct']
            
            df_future['slot_index'] = df_future.index

            region_counts = df_future.groupby('region')['slot_index'].count().to_dict()
            n_regions = len(region_counts)
            max_jobs_per_region = N_JOBS // n_regions + (1 if N_JOBS % n_regions != 0 else 0)
            
        st.success("Normalization complete ✔ (Cloud cover inverted to solar availability)")

        # Define optimization problem - OPTIMIZED with pre-computed arrays
        class CloudSchedulingProblem(Problem):
            def __init__(self, df, n_jobs, max_jobs_per_region, weight_carbon, weight_renewable, weight_load):
                self.df = df.reset_index(drop=True)
                self.n_jobs = n_jobs
                self.max_jobs_per_region = max_jobs_per_region
                self.weight_carbon = weight_carbon
                self.weight_renewable = weight_renewable
                self.weight_load = weight_load
                
                # Pre-compute arrays for fast access
                self.carbon_array = df['carbon_norm'].values
                self.renewable_array = df['renewable_norm'].values
                self.region_array = df['region'].values

                n_var = n_jobs
                super().__init__(n_var=n_var, n_obj=2, n_constr=1, xl=0, xu=len(df)-1, type_var=int)

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                G = []
                for sol in X:
                    sol_int = np.array(np.round(sol), dtype=int)
                    # Clip indices just in case
                    sol_int = np.clip(sol_int, 0, len(self.df)-1)

                    # Use pre-computed arrays instead of DataFrame indexing
                    carbon = self.carbon_array[sol_int]
                    renewable = self.renewable_array[sol_int]

                    # Duplicate constraint
                    duplicates = len(sol_int) - len(np.unique(sol_int))

                    # Region load penalty - optimized
                    regions_selected = self.region_array[sol_int]
                    unique_regions, counts = np.unique(regions_selected, return_counts=True)
                    load_penalty = np.sum(np.maximum(0, counts - self.max_jobs_per_region))

                    # Objective 1: Minimize carbon (with load penalty)
                    obj_carbon = self.weight_carbon * carbon.mean() + self.weight_load * load_penalty
                    
                    # Objective 2: Maximize renewable (negative because we minimize)
                    obj_renewable = -self.weight_renewable * renewable.mean()

                    F.append([obj_carbon, obj_renewable])
                    G.append([duplicates])

                out["F"] = np.array(F)
                out["G"] = np.array(G)

        # Run NSGA-II
        with st.spinner("Running NSGA-II optimization — this may take some time..."):
            problem = CloudSchedulingProblem(
                df_future,
                n_jobs=N_JOBS,
                max_jobs_per_region=max_jobs_per_region,
                weight_carbon=weight_carbon,
                weight_renewable=weight_renewable,
                weight_load=weight_load
            )

            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=crossover_prob, eta=15, vtype=float),
                mutation=PM(prob=mutation_prob, eta=20, vtype=float),
                eliminate_duplicates=True
            )

            res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=False)
        st.success("Optimization finished ✅")

        # Extract one Pareto solution (first)
        with st.spinner("Extracting best schedule..."):
            if res.X is None or len(res.X) == 0:
                st.error("Optimization returned no solutions.")
                st.stop()

            best_solution_int = np.array(np.round(res.X[0]), dtype=int)
            best_solution_int = np.clip(best_solution_int, 0, len(df_future)-1)

            scheduled_slots = df_future.loc[best_solution_int,
                                ['timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 
                                 'solar_availability_pct', 'wind_speed', 'temperature']].copy()
            scheduled_slots['job_id'] = range(1, N_JOBS + 1)
            scheduled_slots = scheduled_slots.sort_values('timestamp').reset_index(drop=True)

            avg_carbon = scheduled_slots['carbon_intensity'].mean()
            avg_cloud = scheduled_slots['solar_cloud_pct'].mean()
            avg_solar = scheduled_slots['solar_availability_pct'].mean()
            avg_wind = scheduled_slots['wind_speed'].mean()
            avg_temp = scheduled_slots['temperature'].mean()
            region_distribution = scheduled_slots['region'].value_counts()

        # Show metrics and tables
        st.header("📋 Optimized Schedule Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Carbon Intensity", f"{avg_carbon:.2f} gCO₂/kWh")
        col2.metric("Avg Solar Availability", f"{avg_solar:.1f}%", 
                   help="Higher is better (inverted from cloud cover)")
        col3.metric("Avg Wind Speed", f"{avg_wind:.1f} m/s")
        col4.metric("Avg Temperature", f"{avg_temp:.1f}°C")

        st.subheader("Regional Distribution")
        dist_df = region_distribution.reset_index()
        dist_df.columns = ['region', 'jobs']
        st.table(dist_df)

        st.subheader("Sample Scheduled Jobs (first 20)")
        display_cols = ['job_id', 'timestamp', 'region', 'carbon_intensity', 
                       'solar_availability_pct', 'wind_speed', 'temperature']
        st.dataframe(scheduled_slots[display_cols].head(20), use_container_width=True)

        # Plots
        st.subheader("Forecasted Metrics (all regions)")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Carbon Intensity", "Solar Availability", "Wind Speed", "Temperature"])
        
        with tab1:
            fig1 = px.line(df_future, x='timestamp', y='carbon_intensity', color='region',
                          title="Carbon Intensity Forecast")
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            fig2 = px.line(df_future, x='timestamp', y='solar_availability_pct', color='region',
                          title="Solar Availability % Forecast (100% - cloud cover)")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            fig3 = px.line(df_future, x='timestamp', y='wind_speed', color='region',
                          title="Wind Speed Forecast")
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            fig4 = px.line(df_future, x='timestamp', y='temperature', color='region',
                          title="Temperature Forecast")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Scheduled Jobs on Timeline")
        # Mark scheduled slots on df_future timeline
        df_future_mark = df_future.copy()
        df_future_mark['scheduled'] = False
        df_future_mark.loc[best_solution_int, 'scheduled'] = True
        
        fig5 = px.scatter(df_future_mark, x='timestamp', y='region', color='scheduled',
                          title="Timeline (scheduled slots highlighted)",
                          color_discrete_map={True: 'red', False: 'lightblue'})
        st.plotly_chart(fig5, use_container_width=True)

        # Renewable vs Carbon scatter
        st.subheader("Trade-off: Carbon vs Renewable Energy")
        fig6 = px.scatter(scheduled_slots, x='carbon_intensity', y='solar_availability_pct',
                         size='wind_speed', color='region', hover_data=['timestamp', 'job_id'],
                         title="Carbon Intensity vs Solar Availability (bubble size = wind speed)")
        st.plotly_chart(fig6, use_container_width=True)

        # Download schedule
        csv = scheduled_slots.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Optimized Schedule (CSV)", csv, "cloud_schedule_future_optimal.csv", "text/csv")

        # Save to server file (optional)
        try:
            scheduled_slots.to_csv("cloud_schedule_future_optimal.csv", index=False)
            st.info("✅ Schedule also saved to server as `cloud_schedule_future_optimal.csv`")
        except Exception as e:
            st.warning(f"Could not save to server: {e}")

        st.success("✅ Schedule ready — check the table and download as needed.")
        
        # Show optimization details
        with st.expander("🔍 Optimization Details"):
            st.write(f"**Total candidate slots:** {len(df_future)}")
            st.write(f"**Jobs scheduled:** {N_JOBS}")
            st.write(f"**Regions:** {n_regions}")
            st.write(f"**Max jobs per region:** {max_jobs_per_region}")
            st.write(f"**Population size:** {pop_size}")
            st.write(f"**Generations:** {n_gen}")
            st.write(f"**Carbon weight:** {weight_carbon:.2f}")
            st.write(f"**Renewable weight:** {weight_renewable:.2f}")
            st.write(f"**Load balance weight:** {weight_load:.2f}")
            
            if hasattr(res, 'F') and res.F is not None:
                st.write("**Objective values (Pareto front):**")
                pareto_df = pd.DataFrame(res.F, columns=['Carbon Objective', 'Renewable Objective (negative)'])
                st.dataframe(pareto_df.head(10))
