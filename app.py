import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
import plotly.express as px

# ---------------------------
# Page config & header
# ---------------------------
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
                df_historical = pd.read_csv(uploaded_csv, parse_dates=['timestamp'])
                ok, msg = validate_df(df_historical)
                if not ok:
                    st.error(msg)
                    st.stop()

                # Load models: expected structure: models[region][target] = estimator
                models = pickle.load(uploaded_pkl)

                # Basic preprocessing and forward-fill
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

            st.success("Loaded data & model ✔")
        except Exception as e:
            st.exception(f"Failed to load files: {e}")
            st.stop()

        # Forecasting
        with st.spinner("Generating future timestamps and forecasting..."):
            regions = df_historical['region'].unique()
            last_timestamp = df_historical['timestamp'].max()
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(minutes=5),
                periods=forecast_days * 288,  # 5min slots per day
                freq='5min'
            )
            future_data = []
            for region in regions:
                # If model doesn't have region, try to use nearest or raise
                if region not in models:
                    st.warning(f"Region '{region}' not present in model file — skipping.")
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

                    # Predict each parameter using model for this region
                    for target in target_cols:
                        X_pred = pd.DataFrame([features])[feature_cols]
                        prediction = models[region][target].predict(X_pred)[0]
                        features[target] = float(prediction)

                    future_data.append(features)
                    last_values = pd.Series(features)

            if len(future_data) == 0:
                st.error("No future data generated — check that your model contains the regions present in CSV.")
                st.stop()

            df_future = pd.DataFrame(future_data)
            df_future = df_future.reset_index(drop=True)
        st.success(f"Generated {len(df_future)} future slots ({df_future['timestamp'].min()} → {df_future['timestamp'].max()})")

        # Normalization
        with st.spinner("Normalizing predictions..."):
            scaler = MinMaxScaler()
            scaler.fit(df_historical[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']])
            df_future[['carbon_norm', 'solar_norm', 'wind_norm', 'temp_norm']] = scaler.transform(
                df_future[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']]
            )
            df_future['renewable_norm'] = (df_future['solar_norm'] + df_future['wind_norm']) / 2
            df_future['slot_index'] = df_future.index

            region_counts = df_future.groupby('region')['slot_index'].count().to_dict()
            n_regions = len(region_counts)
            max_jobs_per_region = N_JOBS // n_regions + (1 if N_JOBS % n_regions != 0 else 0)
        st.success("Normalization complete ✔")

        # Define optimization problem
        class CloudSchedulingProblem(Problem):
            def __init__(self, df, n_jobs, max_jobs_per_region, weight_carbon, weight_renewable, weight_load):
                self.df = df.reset_index(drop=True)
                self.n_jobs = n_jobs
                self.max_jobs_per_region = max_jobs_per_region
                self.weight_carbon = weight_carbon
                self.weight_renewable = weight_renewable
                self.weight_load = weight_load

                n_var = n_jobs
                super().__init__(n_var=n_var, n_obj=2, n_constr=1, xl=0, xu=len(df)-1, type_var=int)

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                G = []
                for sol in X:
                    sol_int = np.array(np.round(sol), dtype=int)
                    # Clip indices just in case
                    sol_int = np.clip(sol_int, 0, len(self.df)-1)

                    carbon = np.array([self.df.loc[i, 'carbon_norm'] for i in sol_int])
                    renewable = np.array([self.df.loc[i, 'renewable_norm'] for i in sol_int])

                    # Duplicate constraint
                    duplicates = len(sol_int) - len(np.unique(sol_int))

                    # Region load penalty
                    region_counts = self.df.loc[sol_int, 'region'].value_counts()
                    load_penalty = 0
                    for count in region_counts:
                        if count > self.max_jobs_per_region:
                            load_penalty += (count - self.max_jobs_per_region)

                    obj_carbon = self.weight_carbon * carbon.mean() + self.weight_load * load_penalty
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
                                ['timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']].copy()
            scheduled_slots['job_id'] = range(1, N_JOBS + 1)
            scheduled_slots = scheduled_slots.sort_values('timestamp').reset_index(drop=True)

            avg_carbon = scheduled_slots['carbon_intensity'].mean()
            avg_solar = scheduled_slots['solar_cloud_pct'].mean()
            avg_wind = scheduled_slots['wind_speed'].mean()
            region_distribution = scheduled_slots['region'].value_counts()

        # Show metrics and tables
        st.header("📋 Optimized Schedule Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Carbon Intensity", f"{avg_carbon:.2f} gCO₂/kWh")
        col2.metric("Avg Solar Cloud Cover", f"{avg_solar:.1f} %")
        col3.metric("Avg Wind Speed", f"{avg_wind:.1f} m/s")

        st.subheader("Regional Distribution")
        dist_df = region_distribution.reset_index()
        dist_df.columns = ['region', 'jobs']
        st.table(dist_df)

        st.subheader("Sample Scheduled Jobs (first 20)")
        st.dataframe(scheduled_slots.head(20), use_container_width=True)

        # Plots
        st.subheader("Forecasted Carbon Intensity (all regions)")
        fig = px.line(df_future, x='timestamp', y='carbon_intensity', color='region')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scheduled Jobs on Timeline")
        # Mark scheduled slots on df_future timeline
        df_future_mark = df_future.copy()
        df_future_mark['scheduled'] = False
        df_future_mark.loc[best_solution_int, 'scheduled'] = True
        fig2 = px.scatter(df_future_mark, x='timestamp', y='region', color='scheduled',
                          title="Timeline (scheduled slots highlighted)")
        st.plotly_chart(fig2, use_container_width=True)

        # Download schedule
        csv = scheduled_slots.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Optimized Schedule (CSV)", csv, "cloud_schedule_future_optimal.csv", "text/csv")

        # Save to server file (optional)
        try:
            scheduled_slots.to_csv("cloud_schedule_future_optimal.csv", index=False)
        except Exception:
            # ignore write failures on restricted hosts
            pass

        st.success("Schedule ready — check the table and download as needed.")
