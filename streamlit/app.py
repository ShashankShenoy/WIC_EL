import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import BytesIO

# ---------------------------
# Page config & header
# ---------------------------
st.set_page_config(
    page_title="Advanced Cloud Job Scheduler — Multi-Objective Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# Initialize session state
if 'df_future' not in st.session_state:
    st.session_state.df_future = None
if 'scheduled_slots' not in st.session_state:
    st.session_state.scheduled_slots = None
if 'all_solutions' not in st.session_state:
    st.session_state.all_solutions = None
if 'convergence_history' not in st.session_state:
    st.session_state.convergence_history = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'df_historical' not in st.session_state:
    st.session_state.df_historical = None

# Custom callback to track convergence
class ConvergenceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = []
    
    def notify(self, algorithm):
        self.history.append({
            'n_gen': algorithm.n_gen,
            'min_f1': algorithm.pop.get("F")[:, 0].min(),
            'min_f2': algorithm.pop.get("F")[:, 1].min(),
            'avg_f1': algorithm.pop.get("F")[:, 0].mean(),
            'avg_f2': algorithm.pop.get("F")[:, 1].mean(),
        })

# Multi-page navigation
page = st.sidebar.radio(
    "📑 Navigation",
    ["🏠 Home", "📊 Data & Forecasting", "⚙️ Optimization", "📈 Analytics & Insights", "💾 Export & Reports"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")

# Sidebar: Uploads & settings
with st.sidebar.expander("📁 Data & Model", expanded=(page == "🏠 Home")):
    uploaded_csv = st.file_uploader("Upload historical CSV (must have 'timestamp' & 'region')", type=["csv"])
    uploaded_pkl = st.file_uploader("Upload trained models (.pkl)", type=["pkl"])

with st.sidebar.expander("📅 Forecast Settings", expanded=False):
    N_JOBS = st.slider("Number of jobs to schedule", min_value=1, max_value=500, value=20, step=1)
    avg_job_power_kw = st.slider("Average job power consumption (kW)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, help="Average power draw per job")
    avg_job_duration_min = st.slider("Average job duration (minutes)", min_value=1, max_value=120, value=5, step=1, help="How long each job runs")
    forecast_days = st.slider("Forecast days ahead", min_value=1, max_value=30, value=3, step=1)
    confidence_level = st.slider("Confidence interval (%)", min_value=80, max_value=99, value=95, step=1)

with st.sidebar.expander("🎯 Optimization Objectives", expanded=False):
    weight_carbon = st.slider("Weight — Carbon Intensity", 0.0, 1.0, 0.5, 0.01, help="Higher = prioritize low carbon")
    weight_renewable = st.slider("Weight — Renewable Energy", 0.0, 1.0, 0.3, 0.01, help="Higher = prioritize solar/wind")
    weight_cost = st.slider("Weight — Operating Cost", 0.0, 1.0, 0.2, 0.01, help="Higher = minimize electricity cost")
    st.info(f"Total weight: {weight_carbon + weight_renewable + weight_cost:.2f}")
    weight_load = st.slider("Load Balancing Penalty", 0.0, 1.0, 0.15, 0.01)

with st.sidebar.expander("🔬 NSGA-II Hyperparameters", expanded=False):
    pop_size = st.slider("Population size", min_value=20, max_value=500, value=100, step=10)
    n_gen = st.slider("Generations", min_value=10, max_value=500, value=200, step=10)
    crossover_prob = st.slider("Crossover probability (SBX)", 0.0, 1.0, 0.9, 0.01)
    mutation_prob = st.slider("Mutation probability (PM)", 0.0, 1.0, 0.1, 0.01)
    crossover_eta = st.slider("Crossover distribution index (η)", 5.0, 30.0, 15.0, 1.0)
    mutation_eta = st.slider("Mutation distribution index (η)", 5.0, 30.0, 20.0, 1.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

# ---------------------------
# Helper functions
# ---------------------------
def validate_df(df):
    required = {'timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature'}
    if not required.issubset(set(df.columns)):
        return False, f"CSV missing required columns. Required: {', '.join(sorted(required))}"
    return True, None

@st.cache_data
def calculate_model_metrics(_df_historical, _models, feature_cols, target_cols):
    """Calculate comprehensive model performance metrics"""
    metrics = {}
    for region in _models.keys():
        region_data = _df_historical[_df_historical['region'] == region]
        if len(region_data) < 100:
            continue
        
        metrics[region] = {}
        for target in target_cols:
            if target not in _models[region]:
                continue
            
            X = region_data[feature_cols]
            y_true = region_data[target]
            y_pred = _models[region][target].predict(X)
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            metrics[region][target] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
    return metrics

def add_confidence_intervals(df_future, df_historical, target_cols, confidence_level=95):
    """Add confidence intervals to predictions based on historical residuals"""
    z_score = {80: 1.28, 90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
    
    for target in target_cols:
        # Estimate std from historical data per region
        std_dict = df_historical.groupby('region')[target].std().to_dict()
        df_future[f'{target}_lower'] = df_future.apply(
            lambda row: row[target] - z_score * std_dict.get(row['region'], df_historical[target].std()),
            axis=1
        )
        df_future[f'{target}_upper'] = df_future.apply(
            lambda row: row[target] + z_score * std_dict.get(row['region'], df_historical[target].std()),
            axis=1
        )
    return df_future

def calculate_baseline_schedule(df_future, n_jobs):
    """Calculate naive baseline: schedule jobs sequentially starting from first slot"""
    baseline_slots = df_future.head(n_jobs)[['timestamp', 'region', 'carbon_intensity', 
                                              'solar_cloud_pct', 'wind_speed', 'temperature']].copy()
    baseline_slots['job_id'] = range(1, n_jobs + 1)
    return baseline_slots

def generate_cost_data(df):
    """Generate synthetic electricity cost data based on region and time"""
    # Simulate regional base prices ($/kWh)
    region_prices = {
        region: np.random.uniform(0.08, 0.15) 
        for region in df['region'].unique()
    }
    
    df['base_cost'] = df['region'].map(region_prices)
    # Add time-of-day multiplier (peak hours cost more)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['time_multiplier'] = df['hour'].apply(
        lambda h: 1.5 if 9 <= h <= 17 else 1.2 if 17 < h <= 21 else 0.8
    )
    df['electricity_cost'] = df['base_cost'] * df['time_multiplier'] * (1 + df['carbon_intensity']/1000)
    return df

# ---------------------------
# PAGE: Home
# ---------------------------
if page == "🏠 Home":
    st.title("🌍 Advanced Cloud Job Scheduler")
    st.markdown("""
    ### Multi-Objective Optimization for Sustainable Cloud Computing
    
    This advanced system uses **NSGA-II genetic algorithm** to schedule cloud workloads across multiple datacenters,
    simultaneously optimizing for multiple conflicting objectives:
    - 🌱 **Carbon Footprint** — Minimize CO₂ emissions by scheduling during clean energy periods
    - ⚡ **Renewable Energy** — Maximize solar and wind utilization across regions
    - 💰 **Operating Costs** — Reduce electricity expenses through smart timing
    - ⚖️ **Load Balancing** — Distribute jobs evenly to prevent server overload
    
    **Why this matters:** Cloud datacenters consume 2-3% of global electricity. Intelligent scheduling can reduce 
    emissions by 40%+ while cutting costs—no hardware changes needed, just smarter timing.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", "Multi-Page", "Professional", help="5 specialized pages for different tasks")
    with col2:
        st.metric("Algorithm", "NSGA-II", "Genetic", help="State-of-the-art multi-objective optimization")
    with col3:
        st.metric("Objectives", "3 Primary", "Multi-Objective", help="Carbon, Renewable, Cost optimized simultaneously")
    
    st.markdown("---")
    st.subheader("📋 How It Works")
    st.markdown("""
    **Step-by-step process:**
    
    1. **Upload Data** 📁
       - Historical energy/carbon data (CSV with 5-minute intervals)
       - Trained ML models (pickle file with predictive models per region/target)
    
    2. **Forecast Future Conditions** 🔮
       - ML models predict carbon intensity, solar, wind, temperature
       - Generates forecasts for 1-30 days ahead
       - Includes confidence intervals to quantify uncertainty
    
    3. **Run Multi-Objective Optimization** ⚙️
       - NSGA-II genetic algorithm explores solution space
       - Evolves 100-500 candidate schedules over 100-300 generations
       - Finds Pareto front: all non-dominated optimal solutions
    
    4. **Analyze Results** 📊
       - 3D Pareto front shows trade-offs between objectives
       - Convergence plots demonstrate algorithm learning
       - Compare optimized schedule vs naive baseline
       - Calculate carbon savings, cost reduction, ROI
    
    5. **Export & Deploy** 💾
       - Download schedules in CSV, JSON, or Excel format
       - Generate comprehensive reports
       - Integrate with Kubernetes, AWS, Azure, GCP schedulers
    """)
    
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.info("""
    **👈 Quick Start Guide:**
    1. Click **"Data & Model"** in the sidebar to upload your files
    2. Configure optimization parameters (or use defaults)
    3. Click **"🚀 Run Full Pipeline"**
    4. Navigate through pages to explore results
    
    **No data?** Run `python generate_sample_data.py` to create realistic test data!
    """)
    
    if st.session_state.scheduled_slots is not None:
        st.success("✅ Optimization completed! Navigate to Analytics or Export pages.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Jobs Scheduled", len(st.session_state.scheduled_slots))
        col2.metric("Pareto Solutions", len(st.session_state.all_solutions) if st.session_state.all_solutions else 0)
        col3.metric("Regions Used", st.session_state.scheduled_slots['region'].nunique())

# ---------------------------
# PAGE: Data & Forecasting
# ---------------------------
elif page == "📊 Data & Forecasting":
    st.title("📊 Data Analysis & ML Forecasting")
    
    if st.session_state.df_historical is None:
        st.warning("⚠️ Please upload data on the Home page first.")
    else:
        st.success(f"✅ Loaded {len(st.session_state.df_historical)} historical records")
        
        tab1, tab2, tab3 = st.tabs(["📈 Historical Data", "🔮 Forecasts", "📊 Model Performance"])
        
        with tab1:
            st.subheader("Historical Energy & Carbon Data")
            st.markdown("""
            **Overview:** This section displays historical data collected from multiple datacenter regions. 
            The data includes carbon intensity (gCO₂/kWh), renewable energy availability (solar and wind), 
            and temperature measurements at 5-minute intervals.
            """)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(st.session_state.df_historical):,}", help="Number of 5-minute interval measurements")
            col2.metric("Regions", st.session_state.df_historical['region'].nunique(), help="Number of datacenter regions in the dataset")
            col3.metric("Date Range", f"{(st.session_state.df_historical['timestamp'].max() - st.session_state.df_historical['timestamp'].min()).days} days", help="Time span of historical data")
            col4.metric("Avg Carbon", f"{st.session_state.df_historical['carbon_intensity'].mean():.1f} gCO₂/kWh", help="Average carbon emissions per kilowatt-hour")
            
            # Time series plots
            st.subheader("Carbon Intensity Over Time")
            st.info("""
            **📊 What you're seeing:** Carbon intensity varies throughout the day and across regions. Lower values (green) 
            indicate cleaner energy (more renewables), while higher values (red) indicate fossil fuel dependency. 
            Notice how patterns differ by region due to local energy mix and demand cycles.
            """)
            fig = px.line(st.session_state.df_historical, x='timestamp', y='carbon_intensity', 
                         color='region', title="Historical Carbon Intensity by Region")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Distribution Analysis:** Box plots show the spread and median carbon intensity per region. Wider boxes = more variability.")
                fig2 = px.box(st.session_state.df_historical, x='region', y='carbon_intensity',
                             title="Carbon Intensity Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.markdown("**Correlation:** This scatter plot reveals the relationship between wind speed and carbon intensity. Darker colors indicate higher solar availability. Lower-right points show ideal conditions (high wind + high solar = low carbon).")
                fig3 = px.scatter(st.session_state.df_historical, x='wind_speed', y='carbon_intensity',
                                 color='solar_cloud_pct', title="Carbon vs Wind Speed (colored by Solar)")
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            if st.session_state.df_future is None:
                st.info("Run optimization to generate forecasts.")
            else:
                st.subheader(f"Forecasted Conditions ({len(st.session_state.df_future)} time slots)")
                st.markdown("""
                **🔮 ML Forecasting:** Using trained machine learning models, we predict future carbon intensity, 
                solar availability, wind speed, and temperature for each region. These predictions guide the 
                optimization algorithm to schedule jobs at the most sustainable times.
                """)
                
                # Forecast with confidence intervals
                fig = go.Figure()
                for region in st.session_state.df_future['region'].unique():
                    region_data = st.session_state.df_future[st.session_state.df_future['region'] == region]
                    fig.add_trace(go.Scatter(
                        x=region_data['timestamp'], 
                        y=region_data['carbon_intensity'],
                        mode='lines',
                        name=region,
                        line=dict(width=2)
                    ))
                    if 'carbon_intensity_lower' in region_data.columns:
                        fig.add_trace(go.Scatter(
                            x=region_data['timestamp'].tolist() + region_data['timestamp'].tolist()[::-1],
                            y=region_data['carbon_intensity_upper'].tolist() + region_data['carbon_intensity_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor=f'rgba(0,100,200,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f'{region} CI'
                        ))
                
                fig.update_layout(title="Forecasted Carbon Intensity with Confidence Intervals",
                                 xaxis_title="Time", yaxis_title="Carbon Intensity (gCO₂/kWh)")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **📈 Confidence Intervals:** The shaded areas represent prediction uncertainty (typically 95% confidence). 
                Wider bands indicate less certain predictions. The solid lines show the most likely forecast.
                """)
                
                # Heatmap
                st.subheader("Regional Heatmap — Carbon Intensity")
                st.info("""
                **🗺️ Pattern Recognition:** This heatmap shows average carbon intensity by hour and region. 
                Darker colors = higher carbon. Look for light-colored cells to identify the best times to schedule 
                workloads in each region (typically when renewable energy is abundant).
                """)
                pivot = st.session_state.df_future.pivot_table(
                    values='carbon_intensity',
                    index=st.session_state.df_future['timestamp'].dt.hour,
                    columns='region',
                    aggfunc='mean'
                )
                fig_heat = px.imshow(pivot, 
                                    labels=dict(x="Region", y="Hour of Day", color="Carbon"),
                                    title="Average Carbon Intensity by Region & Hour")
                st.plotly_chart(fig_heat, use_container_width=True)
        
        with tab3:
            if st.session_state.model_metrics is None:
                st.info("Model metrics will appear after running optimization.")
            else:
                st.subheader("ML Model Performance Metrics")
                st.markdown("""
                **📊 Evaluation Metrics Explained:**
                - **RMSE** (Root Mean Square Error): Average prediction error. Lower is better. Units match the target (e.g., gCO₂/kWh for carbon).
                - **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values. More intuitive than RMSE.
                - **R²** (R-squared): Model fit quality. 1.0 = perfect, 0.85+ = excellent, 0.7+ = good, <0.5 = poor.
                - **MAPE** (Mean Absolute Percentage Error): Error as a percentage. <10% = excellent, 10-20% = good, >20% = needs improvement.
                """)
                
                # Convert metrics to DataFrame
                metrics_data = []
                for region, targets in st.session_state.model_metrics.items():
                    for target, scores in targets.items():
                        metrics_data.append({
                            'Region': region,
                            'Target': target,
                            'RMSE': scores['RMSE'],
                            'MAE': scores['MAE'],
                            'R²': scores['R²'],
                            'MAPE': scores['MAPE']
                        })
                
                df_metrics = pd.DataFrame(metrics_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average R² Score", f"{df_metrics['R²'].mean():.3f}")
                    st.metric("Average RMSE", f"{df_metrics['RMSE'].mean():.2f}")
                with col2:
                    st.metric("Best R² Score", f"{df_metrics['R²'].max():.3f}")
                    st.metric("Average MAPE", f"{df_metrics['MAPE'].mean():.1f}%")
                
                st.dataframe(df_metrics.round(3), use_container_width=True)
                
                st.markdown("""
                **💡 Interpretation:** Higher R² scores indicate the model accurately captures patterns in the data. 
                If R² < 0.7 for any target, consider retraining with more data or different features.
                """)
                
                # Visualize model performance
                fig = px.bar(df_metrics, x='Target', y='R²', color='Region',
                            title="Model R² Scores by Target & Region", barmode='group')
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PAGE: Optimization
# ---------------------------
elif page == "⚙️ Optimization":
    st.title("⚙️ Multi-Objective Optimization")
    
    if st.session_state.all_solutions is None:
        st.info("Run the optimization pipeline from the sidebar to see results here.")
    else:
        st.success("✅ NSGA-II optimization completed!")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Pareto Front", "📉 Convergence", "🔍 Solution Explorer"])
        
        with tab1:
            st.subheader("Pareto-Optimal Solutions")
            st.markdown("""
            **🎯 What is a Pareto Front?**
            
            The Pareto front shows all non-dominated solutions found by the NSGA-II genetic algorithm. 
            Each point represents a different trade-off between three conflicting objectives:
            - **Carbon Score** (X-axis): Lower = less CO₂ emissions
            - **Renewable Score** (Y-axis): More negative = more solar/wind energy used
            - **Cost Score** (Z-axis): Lower = cheaper electricity costs
            
            **Why multiple solutions?** There's no single "best" schedule—it depends on your priorities. 
            A solution excellent for carbon might be expensive. One that's cheap might use fossil fuels. 
            The Pareto front lets you choose based on your organization's values.
            """)
            
            if st.session_state.all_solutions is not None and len(st.session_state.all_solutions) > 0:
                # 3D Pareto front
                fig = go.Figure(data=[go.Scatter3d(
                    x=[s['obj1'] for s in st.session_state.all_solutions],
                    y=[s['obj2'] for s in st.session_state.all_solutions],
                    z=[s['obj3'] for s in st.session_state.all_solutions],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=[s['obj1'] for s in st.session_state.all_solutions],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Carbon Score")
                    ),
                    text=[f"Solution {i+1}" for i in range(len(st.session_state.all_solutions))],
                    hovertemplate='<b>%{text}</b><br>Carbon: %{x:.3f}<br>Renewable: %{y:.3f}<br>Cost: %{z:.3f}'
                )])
                fig.update_layout(
                    title="3D Pareto Front — All Optimal Solutions",
                    scene=dict(
                        xaxis_title="Carbon Score (minimize)",
                        yaxis_title="Renewable Score (minimize=-maximize)",
                        zaxis_title="Cost Score (minimize)"
                    ),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 2D projections
                st.markdown("""
                **📉 2D Trade-off Views:** These charts show pairwise relationships between objectives. 
                Points on the lower-left edge are best for both objectives shown. The curved shape is typical 
                of Pareto fronts—improving one objective increasingly sacrifices the other.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = px.scatter(
                        x=[s['obj1'] for s in st.session_state.all_solutions],
                        y=[s['obj2'] for s in st.session_state.all_solutions],
                        labels={'x': 'Carbon Score', 'y': 'Renewable Score'},
                        title="Carbon vs Renewable Trade-off"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = px.scatter(
                        x=[s['obj1'] for s in st.session_state.all_solutions],
                        y=[s['obj3'] for s in st.session_state.all_solutions],
                        labels={'x': 'Carbon Score', 'y': 'Cost Score'},
                        title="Carbon vs Cost Trade-off"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            st.subheader("NSGA-II Convergence Analysis")
            st.markdown("""
            **📉 How the Algorithm Learns:**
            
            NSGA-II is a genetic algorithm that evolves better solutions over generations, like natural selection:
            1. **Start:** Randomly generate population of schedules
            2. **Evaluate:** Calculate carbon, renewable, and cost scores for each
            3. **Select:** Keep the best performers (survival of the fittest)
            4. **Crossover:** Combine good solutions to create offspring
            5. **Mutate:** Randomly modify some solutions for exploration
            6. **Repeat:** Iterate for 100-300 generations
            
            **Reading the charts:** Lines should trend downward (improving). Flattening indicates convergence 
            (algorithm found near-optimal solutions).
            """)
            
            if st.session_state.convergence_history:
                df_conv = pd.DataFrame(st.session_state.convergence_history)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Minimum Carbon Score', 'Minimum Renewable Score',
                                   'Average Carbon Score', 'Average Renewable Score')
                )
                
                fig.add_trace(go.Scatter(x=df_conv['n_gen'], y=df_conv['min_f1'], 
                                        mode='lines+markers', name='Min Carbon'),
                             row=1, col=1)
                fig.add_trace(go.Scatter(x=df_conv['n_gen'], y=df_conv['min_f2'], 
                                        mode='lines+markers', name='Min Renewable'),
                             row=1, col=2)
                fig.add_trace(go.Scatter(x=df_conv['n_gen'], y=df_conv['avg_f1'], 
                                        mode='lines+markers', name='Avg Carbon'),
                             row=2, col=1)
                fig.add_trace(go.Scatter(x=df_conv['n_gen'], y=df_conv['avg_f2'], 
                                        mode='lines+markers', name='Avg Renewable'),
                             row=2, col=2)
                
                fig.update_xaxes(title_text="Generation", row=2, col=1)
                fig.update_xaxes(title_text="Generation", row=2, col=2)
                fig.update_layout(height=600, showlegend=False, 
                                 title_text="Optimization Progress Over Generations")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Generations", len(df_conv), help="Number of evolution cycles")
                col2.metric("Best Carbon Score", f"{df_conv['min_f1'].min():.4f}", help="Lowest carbon score achieved")
                col3.metric("Improvement", f"{((df_conv['min_f1'].iloc[0] - df_conv['min_f1'].iloc[-1]) / df_conv['min_f1'].iloc[0] * 100):.1f}%", help="How much the algorithm improved from start to finish")
        
        with tab3:
            st.subheader("Interactive Solution Explorer")
            st.markdown("""
            **🔍 Explore Different Schedules:**
            
            Each solution on the Pareto front represents a valid optimal schedule with different trade-offs. 
            Use the slider below to browse through solutions and see their objective scores. In a full implementation, 
            this would display the complete job schedule (which jobs run when and where) for the selected solution.
            
            **Pro tip:** Solutions near the beginning (index 0-20) typically prioritize the first objective (carbon), 
            while later solutions emphasize other objectives.
            """)
            
            if st.session_state.all_solutions and st.session_state.df_future is not None:
                solution_idx = st.slider(
                    "Select Solution",
                    min_value=0,
                    max_value=len(st.session_state.all_solutions)-1,
                    value=0
                )
                
                selected = st.session_state.all_solutions[solution_idx]
                
                # Display objective scores
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Solution #", f"{solution_idx + 1}", help="Position in Pareto front")
                col2.metric("Carbon Score", f"{selected['obj1']:.4f}", help="Lower is better")
                col3.metric("Renewable Score", f"{-selected['obj2']:.4f}", help="Higher is better (negated for minimization)")
                col4.metric("Cost Score", f"{selected['obj3']:.4f}", help="Lower is better")
                
                st.markdown("---")
                
                # Extract slot indices for this solution
                slot_indices = np.array(selected['indices'], dtype=int)
                slot_indices = np.clip(slot_indices, 0, len(st.session_state.df_future)-1)
                
                # Get the scheduled slots for this solution
                solution_schedule = st.session_state.df_future.loc[slot_indices, 
                                    ['timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 
                                     'wind_speed', 'temperature']].copy()
                solution_schedule['job_id'] = range(1, len(slot_indices) + 1)
                solution_schedule = solution_schedule.sort_values('timestamp').reset_index(drop=True)
                
                # Calculate statistics
                avg_carbon = solution_schedule['carbon_intensity'].mean()
                avg_solar = solution_schedule['solar_cloud_pct'].mean()
                avg_wind = solution_schedule['wind_speed'].mean()
                
                st.subheader("📋 Solution Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Carbon", f"{avg_carbon:.1f} gCO₂/kWh")
                col2.metric("Avg Solar", f"{avg_solar:.1f}%")
                col3.metric("Avg Wind", f"{avg_wind:.1f} m/s")
                col4.metric("Jobs", len(solution_schedule))
                
                # Regional distribution
                st.subheader("🌍 Regional Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    region_counts = solution_schedule['region'].value_counts().reset_index()
                    region_counts.columns = ['region', 'count']
                    fig_pie = px.pie(region_counts, values='count', names='region',
                                     title=f"Solution {solution_idx + 1}: Jobs per Region")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    region_carbon = solution_schedule.groupby('region')['carbon_intensity'].mean().reset_index()
                    fig_bar = px.bar(region_carbon, x='region', y='carbon_intensity',
                                    title=f"Solution {solution_idx + 1}: Avg Carbon by Region")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Timeline visualization
                st.subheader("📅 Schedule Timeline")
                fig_timeline = px.scatter(
                    solution_schedule,
                    x='timestamp',
                    y='region',
                    size='carbon_intensity',
                    color='carbon_intensity',
                    hover_data=['job_id', 'wind_speed', 'solar_cloud_pct', 'temperature'],
                    title=f"Solution {solution_idx + 1}: Job Distribution Over Time",
                    color_continuous_scale='RdYlGn_r'
                )
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Carbon intensity over time
                st.subheader("📈 Carbon Intensity Timeline")
                fig_carbon = px.line(
                    solution_schedule.sort_values('timestamp'),
                    x='timestamp',
                    y='carbon_intensity',
                    color='region',
                    markers=True,
                    title=f"Solution {solution_idx + 1}: Carbon Intensity by Region",
                    labels={'carbon_intensity': 'Carbon Intensity (gCO₂/kWh)'}
                )
                st.plotly_chart(fig_carbon, use_container_width=True)
                
                # Detailed schedule table
                st.subheader("📊 Detailed Schedule")
                st.markdown(f"**Showing first 20 jobs** (total: {len(solution_schedule)})")
                st.dataframe(
                    solution_schedule.head(20)[['job_id', 'timestamp', 'region', 'carbon_intensity', 
                                                 'solar_cloud_pct', 'wind_speed', 'temperature']].round(2),
                    use_container_width=True
                )
                
                # Comparison with other solutions
                st.subheader("📊 Comparison with Other Solutions")
                st.markdown("**How this solution ranks among all Pareto-optimal solutions:**")
                
                all_carbon = [s['obj1'] for s in st.session_state.all_solutions]
                all_renewable = [-s['obj2'] for s in st.session_state.all_solutions]
                all_cost = [s['obj3'] for s in st.session_state.all_solutions]
                
                col1, col2, col3 = st.columns(3)
                
                carbon_rank = sorted(all_carbon).index(selected['obj1']) + 1
                renewable_rank = sorted(all_renewable, reverse=True).index(-selected['obj2']) + 1
                cost_rank = sorted(all_cost).index(selected['obj3']) + 1
                
                col1.metric("Carbon Rank", f"{carbon_rank} / {len(st.session_state.all_solutions)}", 
                           help="1 = best (lowest carbon)")
                col2.metric("Renewable Rank", f"{renewable_rank} / {len(st.session_state.all_solutions)}", 
                           help="1 = best (highest renewable)")
                col3.metric("Cost Rank", f"{cost_rank} / {len(st.session_state.all_solutions)}", 
                           help="1 = best (lowest cost)")
                
                # Position in Pareto front
                fig_position = go.Figure()
                
                # Plot all solutions in gray
                fig_position.add_trace(go.Scatter(
                    x=all_carbon,
                    y=all_renewable,
                    mode='markers',
                    marker=dict(size=8, color='lightgray'),
                    name='Other Solutions',
                    text=[f"Solution {i+1}" for i in range(len(st.session_state.all_solutions))],
                    hovertemplate='<b>%{text}</b><br>Carbon: %{x:.3f}<br>Renewable: %{y:.3f}'
                ))
                
                # Highlight selected solution in red
                fig_position.add_trace(go.Scatter(
                    x=[selected['obj1']],
                    y=[-selected['obj2']],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Selected (Solution {solution_idx + 1})',
                    hovertemplate=f'<b>Solution {solution_idx + 1}</b><br>Carbon: {selected["obj1"]:.3f}<br>Renewable: {-selected["obj2"]:.3f}'
                ))
                
                fig_position.update_layout(
                    title=f"Solution {solution_idx + 1} Position in Pareto Front",
                    xaxis_title="Carbon Score (lower is better)",
                    yaxis_title="Renewable Score (higher is better)",
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig_position, use_container_width=True)

# ---------------------------
# PAGE: Analytics & Insights
# ---------------------------
elif page == "📈 Analytics & Insights":
    st.title("📈 Analytics & Business Insights")
    
    if st.session_state.scheduled_slots is None:
        st.warning("⚠️ Run optimization first to see analytics.")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 KPIs & Metrics", "📅 Schedule Gantt", "💡 Insights"])
        
        with tab1:
            st.subheader("Key Performance Indicators")
            st.markdown("""
            **📊 Performance Summary:**
            
            These metrics compare our **optimized schedule** (using NSGA-II) against a **naive baseline** 
            (simply scheduling jobs sequentially in the first available slots). The improvements demonstrate 
            the value of intelligent optimization.
            
            - **Green deltas** (↓) = Good (reduction in carbon/cost)
            - **Red deltas** (↑) = Bad (increase in undesirable metrics)
            - **Blue deltas** (↑) = Good (increase in desirable metrics like renewables)
            """)
            
            # Calculate baseline for comparison
            baseline = calculate_baseline_schedule(st.session_state.df_future, N_JOBS)
            optimized = st.session_state.scheduled_slots
            
            col1, col2, col3, col4 = st.columns(4)
            
            baseline_carbon = baseline['carbon_intensity'].mean()
            optimized_carbon = optimized['carbon_intensity'].mean()
            carbon_reduction = ((baseline_carbon - optimized_carbon) / baseline_carbon) * 100
            
            baseline_renewable = (baseline['solar_cloud_pct'].mean() + baseline['wind_speed'].mean()) / 2
            optimized_renewable = (optimized['solar_cloud_pct'].mean() + optimized['wind_speed'].mean()) / 2
            renewable_improvement = ((optimized_renewable - baseline_renewable) / baseline_renewable) * 100
            
            col1.metric(
                "Carbon Intensity",
                f"{optimized_carbon:.1f} gCO₂/kWh",
                f"{-carbon_reduction:.1f}%",
                delta_color="inverse",
                help="Average carbon emissions per kWh. Lower is better. Delta shows % improvement vs baseline."
            )
            col2.metric(
                "Renewable Usage",
                f"{optimized_renewable:.1f}",
                f"+{renewable_improvement:.1f}%",
                help="Combined solar and wind availability metric. Higher means more green energy."
            )
            col3.metric(
                "Total Jobs",
                len(optimized),
                help="Number of cloud workloads scheduled across all regions and time slots."
            )
            col4.metric(
                "Regions Used",
                optimized['region'].nunique(),
                help="Number of distinct datacenter regions utilized in the schedule."
            )
            
            st.markdown("---")
            st.subheader("Baseline vs Optimized Comparison")
            st.info("""
            **🔍 Detailed Metrics:**
            - **Baseline** = Naive sequential scheduling (do nothing smart)
            - **Optimized** = NSGA-II multi-objective optimization
            - **Improvement** = Percentage change (negative for metrics we want to minimize)
            """)
            
            comparison_data = pd.DataFrame({
                'Metric': ['Avg Carbon (gCO₂/kWh)', 'Avg Solar Cover (%)', 'Avg Wind Speed (m/s)', 
                          'Carbon Std Dev', 'Regional Diversity'],
                'Baseline': [
                    baseline_carbon,
                    baseline['solar_cloud_pct'].mean(),
                    baseline['wind_speed'].mean(),
                    baseline['carbon_intensity'].std(),
                    baseline['region'].nunique()
                ],
                'Optimized': [
                    optimized_carbon,
                    optimized['solar_cloud_pct'].mean(),
                    optimized['wind_speed'].mean(),
                    optimized['carbon_intensity'].std(),
                    optimized['region'].nunique()
                ],
                'Improvement': [
                    f"{-carbon_reduction:.1f}%",
                    f"{((optimized['solar_cloud_pct'].mean() - baseline['solar_cloud_pct'].mean()) / baseline['solar_cloud_pct'].mean() * 100):.1f}%",
                    f"{((optimized['wind_speed'].mean() - baseline['wind_speed'].mean()) / baseline['wind_speed'].mean() * 100):.1f}%",
                    f"{((baseline['carbon_intensity'].std() - optimized['carbon_intensity'].std()) / baseline['carbon_intensity'].std() * 100):.1f}%",
                    f"{optimized['region'].nunique() - baseline['region'].nunique()}"
                ]
            })
            st.dataframe(comparison_data, use_container_width=True)
            
            st.markdown("""
            **💡 What these metrics mean:**
            - **Carbon Std Dev**: Lower = more consistent carbon levels (better for predictability)
            - **Regional Diversity**: Using more regions = better load balancing and redundancy
            """)
            
            # Regional distribution
            st.subheader("Regional Job Distribution")
            st.markdown("""
            **🌍 Geographic Analysis:** These charts show how jobs are distributed across datacenter regions. 
            An ideal distribution balances load while prioritizing low-carbon regions.
            """)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(optimized['region'].value_counts().reset_index(),
                            values='count', names='region',
                            title="Optimized Schedule — Jobs per Region")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.bar(optimized.groupby('region')['carbon_intensity'].mean().reset_index(),
                             x='region', y='carbon_intensity',
                             title="Average Carbon Intensity by Region")
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Interactive Gantt Chart — Job Schedule Timeline")
            st.markdown("""
            **📅 Timeline Visualization:**
            
            Gantt charts are standard in project management for visualizing schedules. Each horizontal bar 
            represents one job, showing:
            - **When** it runs (X-axis)
            - **Which region** it runs in (color)
            - **Job details** (hover over bars for carbon, wind, solar data)
            
            **How to use:** Scroll through jobs to see the complete schedule. Look for clusters of same colors 
            to identify which regions are most utilized.
            """)
            
            # Create Gantt chart
            gantt_data = optimized.copy()
            gantt_data['Start'] = gantt_data['timestamp']
            gantt_data['End'] = gantt_data['timestamp'] + pd.Timedelta(minutes=5)
            gantt_data['Job'] = 'Job ' + gantt_data['job_id'].astype(str)
            
            fig = px.timeline(
                gantt_data,
                x_start='Start',
                x_end='End',
                y='Job',
                color='region',
                hover_data=['carbon_intensity', 'wind_speed', 'solar_cloud_pct'],
                title=f"Schedule Timeline — {len(gantt_data)} Jobs Across Regions"
            )
            fig.update_yaxes(categoryorder='total ascending')
            fig.update_layout(height=max(400, len(gantt_data) * 20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline scatter
            st.subheader("Schedule Distribution Over Time")
            st.markdown("""
            **🔵 Bubble Chart:** Each bubble represents one job. 
            - **Bubble size** = carbon intensity (bigger = more carbon)
            - **Bubble color** = carbon intensity (darker = more carbon)
            - **Position** = when and where the job runs
            
            Look for patterns: Are jobs clustered at certain times? Are some regions busier than others?
            """)
            fig2 = px.scatter(
                optimized,
                x='timestamp',
                y='region',
                size='carbon_intensity',
                color='carbon_intensity',
                hover_data=['job_id', 'wind_speed', 'temperature'],
                title="Jobs by Region & Time (bubble size = carbon intensity)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("💡 Business Insights & Recommendations")
            st.markdown("""
            **📈 Data-Driven Insights:**
            
            Based on the optimized schedule analysis, here are key findings and actionable recommendations 
            for your organization. These insights help translate technical results into business value.
            """)
            
            # Generate insights
            best_region = optimized.groupby('region')['carbon_intensity'].mean().idxmin()
            worst_region = optimized.groupby('region')['carbon_intensity'].mean().idxmax()
            peak_time = optimized.groupby(optimized['timestamp'].dt.hour)['carbon_intensity'].mean().idxmin()
            
            st.success(f"✅ **Best Region**: {best_region} has the lowest average carbon intensity")
            st.warning(f"⚠️ **Attention**: {worst_region} has the highest carbon intensity — consider reducing load")
            st.info(f"💡 **Optimal Time**: Hour {peak_time}:00 shows best carbon efficiency")
            
            st.markdown("""---""")
            st.markdown("### Carbon Savings Calculation")
            st.markdown(f"""
            **🌱 Environmental Impact:**
            
            These calculations show the real-world environmental benefit of optimization. Each job consumes 
            {avg_job_power_kw} kW for {avg_job_duration_min} minutes ({avg_job_power_kw * avg_job_duration_min / 60:.3f} kWh per job).
            We compare total emissions between baseline and optimized schedules.
            """)
            # Calculate actual energy-weighted carbon emissions
            job_energy_kwh = avg_job_power_kw * (avg_job_duration_min / 60.0)
            total_carbon_baseline_g = (baseline['carbon_intensity'] * job_energy_kwh).sum()
            total_carbon_optimized_g = (optimized['carbon_intensity'] * job_energy_kwh).sum()
            carbon_saved_kg = (total_carbon_baseline_g - total_carbon_optimized_g) / 1000
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Carbon Saved", f"{carbon_saved_kg:.2f} kg CO₂", help="Total CO₂ emissions avoided through optimization")
            col2.metric("Equivalent to", f"{carbon_saved_kg * 0.00022:.1f} trees", help="Trees needed to absorb this much carbon in one year")
            col3.metric("Cost Savings", f"${carbon_saved_kg * 0.05:.2f}", help="Financial value at $50/ton carbon price")
            
            st.markdown("""---""")
            st.markdown("### Recommendations")
            st.markdown("""
            **🛠️ Action Items for Sustainability:**
            
            Based on this analysis, consider these strategies to further reduce carbon footprint:
            """)
            st.markdown("""
            - **Migrate more workloads** to low-carbon regions during off-peak hours
            - **Implement dynamic pricing** to incentivize flexible job scheduling  
            - **Invest in renewable energy** in high-demand regions
            - **Monitor real-time** carbon intensity and adjust schedules adaptively
            - **Set carbon budgets** per region/datacenter
            """)

# ---------------------------
# PAGE: Export & Reports
# ---------------------------
elif page == "💾 Export & Reports":
    st.title("💾 Export & Reports")
    
    if st.session_state.scheduled_slots is None:
        st.warning("⚠️ No schedule to export. Run optimization first.")
    else:
        st.markdown("""
        **📤 Professional Deliverables:**
        
        Export your optimized schedule in multiple formats for integration with external systems, 
        stakeholder presentations, or archival purposes. Each format serves different use cases:
        
        - **CSV** — Universal format, opens in Excel/Google Sheets, easy data analysis
        - **JSON** — API integration, web services, programming languages
        - **Excel** — Multiple sheets (schedule + Pareto solutions), formatted reports, presentations
        - **Markdown Report** — Human-readable summary, documentation, email-friendly
        """)
        
        st.subheader("Download Optimized Schedule")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV Export
        with col1:
            csv = st.session_state.scheduled_slots.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download as CSV",
                csv,
                "cloud_schedule_optimized.csv",
                "text/csv",
                key='download-csv',
                help="Comma-separated values file for Excel, Google Sheets, pandas"
            )
        
        # JSON Export
        with col2:
            json_data = st.session_state.scheduled_slots.to_json(orient='records', date_format='iso')
            st.download_button(
                "📥 Download as JSON",
                json_data,
                "cloud_schedule_optimized.json",
                "application/json",
                key='download-json',
                help="JavaScript Object Notation for APIs and web services"
            )
        
        # Excel Export
        with col3:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.scheduled_slots.to_excel(writer, index=False, sheet_name='Schedule')
                if st.session_state.all_solutions:
                    pd.DataFrame(st.session_state.all_solutions).to_excel(writer, index=False, sheet_name='Pareto Solutions')
            buffer.seek(0)
            st.download_button(
                "📥 Download as Excel",
                buffer,
                "cloud_schedule_optimized.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel',
                help="Excel workbook with multiple sheets (schedule + Pareto solutions)"
            )
        
        st.markdown("---")
        st.subheader("📊 Summary Report")
        st.markdown("""
        **📄 Comprehensive Report:**
        
        This markdown-formatted report provides a complete summary of the optimization run, including:
        - Timestamp and job count
        - Key performance metrics (carbon, wind, solar)
        - Regional job distribution
        - Algorithm parameters used
        
        Perfect for documentation, email summaries, or converting to PDF for stakeholder presentations.
        """)
        
        # Generate report
        report = f"""
        # Cloud Job Scheduling Optimization Report
        
        **Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Jobs Scheduled**: {len(st.session_state.scheduled_slots)}
        **Regions Utilized**: {st.session_state.scheduled_slots['region'].nunique()}
        
        ## Performance Metrics
        - Average Carbon Intensity: {st.session_state.scheduled_slots['carbon_intensity'].mean():.2f} gCO₂/kWh
        - Average Wind Speed: {st.session_state.scheduled_slots['wind_speed'].mean():.2f} m/s
        - Average Solar Cloud Cover: {st.session_state.scheduled_slots['solar_cloud_pct'].mean():.2f}%
        
        ## Regional Distribution
        {st.session_state.scheduled_slots['region'].value_counts().to_string()}
        
        ## Optimization Parameters
        - Algorithm: NSGA-II
        - Population Size: {pop_size}
        - Generations: {n_gen}
        - Objectives: Carbon, Renewable Energy, Cost
        """
        
        st.markdown(report)
        st.download_button(
            "📥 Download Report (Markdown)",
            report,
            "optimization_report.md",
            "text/markdown",
            key='download-report'
        )

# ---------------------------
# Optimization Pipeline (triggered by button)
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

            # Store in session state
            st.session_state.df_historical = df_historical
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
            
            # Add confidence intervals
            df_future = add_confidence_intervals(df_future, df_historical, target_cols, confidence_level)
            
            # Add cost data
            df_future = generate_cost_data(df_future)
        
        st.success(f"Generated {len(df_future)} future slots ({df_future['timestamp'].min()} → {df_future['timestamp'].max()})")
        
        # Calculate model metrics
        with st.spinner("Calculating model performance metrics..."):
            model_metrics = calculate_model_metrics(df_historical, models, feature_cols, target_cols)
            st.session_state.model_metrics = model_metrics
        st.success("Model evaluation complete ✔")

        # Normalization
        with st.spinner("Normalizing predictions..."):
            scaler = MinMaxScaler()
            scaler.fit(df_historical[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']])
            df_future[['carbon_norm', 'solar_norm', 'wind_norm', 'temp_norm']] = scaler.transform(
                df_future[['carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']]
            )
            df_future['renewable_norm'] = (df_future['solar_norm'] + df_future['wind_norm']) / 2
            
            # Normalize cost
            cost_min = df_future['electricity_cost'].min()
            cost_max = df_future['electricity_cost'].max()
            df_future['cost_norm'] = (df_future['electricity_cost'] - cost_min) / (cost_max - cost_min + 1e-8)
            
            df_future['slot_index'] = df_future.index
            st.session_state.df_future = df_future

            region_counts = df_future.groupby('region')['slot_index'].count().to_dict()
            n_regions = len(region_counts)
            max_jobs_per_region = N_JOBS // n_regions + (1 if N_JOBS % n_regions != 0 else 0)
        st.success("Normalization complete ✔")

        # Define optimization problem (3 objectives now)
        class CloudSchedulingProblem(Problem):
            def __init__(self, df, n_jobs, max_jobs_per_region, weight_carbon, weight_renewable, weight_cost, weight_load, job_power_kw, job_duration_min):
                self.df = df.reset_index(drop=True)
                self.n_jobs = n_jobs
                self.max_jobs_per_region = max_jobs_per_region
                self.weight_carbon = weight_carbon
                self.weight_renewable = weight_renewable
                self.weight_cost = weight_cost
                self.weight_load = weight_load
                self.job_power_kw = job_power_kw
                self.job_duration_min = job_duration_min
                # Calculate energy per job in kWh
                self.job_energy_kwh = job_power_kw * (job_duration_min / 60.0)

                n_var = n_jobs
                super().__init__(n_var=n_var, n_obj=3, n_constr=1, xl=0, xu=len(df)-1, type_var=int)

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                G = []
                for sol in X:
                    sol_int = np.array(np.round(sol), dtype=int)
                    # Clip indices just in case
                    sol_int = np.clip(sol_int, 0, len(self.df)-1)

                    carbon = np.array([self.df.loc[i, 'carbon_norm'] for i in sol_int])
                    renewable = np.array([self.df.loc[i, 'renewable_norm'] for i in sol_int])
                    cost = np.array([self.df.loc[i, 'cost_norm'] for i in sol_int])
                    
                    # Get actual carbon intensity values (g CO2/kWh)
                    carbon_intensity_actual = np.array([self.df.loc[i, 'carbon_intensity'] for i in sol_int])
                    
                    # Calculate TOTAL carbon emissions (kg CO2)
                    # Formula: carbon_intensity (g CO2/kWh) × energy_per_job (kWh) = g CO2 per job
                    # Sum across all jobs to get total emissions, then convert to kg
                    # This properly accounts for: (1) varying carbon intensity by time/region
                    # (2) actual energy consumption per job (power × duration)
                    total_carbon_g = (carbon_intensity_actual * self.job_energy_kwh).sum()
                    total_carbon_kg = total_carbon_g / 1000.0

                    # Duplicate constraint
                    duplicates = len(sol_int) - len(np.unique(sol_int))

                    # Region load penalty
                    region_counts = self.df.loc[sol_int, 'region'].value_counts()
                    load_penalty = 0
                    for count in region_counts:
                        if count > self.max_jobs_per_region:
                            load_penalty += (count - self.max_jobs_per_region)

                    # Use TOTAL carbon emissions, not average (properly weighted by energy)
                    obj_carbon = self.weight_carbon * (total_carbon_kg / self.n_jobs) + self.weight_load * load_penalty
                    obj_renewable = -self.weight_renewable * renewable.mean()  # Negative to maximize
                    obj_cost = self.weight_cost * cost.mean()

                    F.append([obj_carbon, obj_renewable, obj_cost])
                    G.append([duplicates])

                out["F"] = np.array(F)
                out["G"] = np.array(G)

        # Run NSGA-II with convergence tracking
        with st.spinner("Running NSGA-II optimization — this may take some time..."):
            problem = CloudSchedulingProblem(
                df_future,
                n_jobs=N_JOBS,
                max_jobs_per_region=max_jobs_per_region,
                weight_carbon=weight_carbon,
                weight_renewable=weight_renewable,
                weight_cost=weight_cost,
                weight_load=weight_load,
                job_power_kw=avg_job_power_kw,
                job_duration_min=avg_job_duration_min
            )

            callback = ConvergenceCallback()
            
            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float),
                mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float),
                eliminate_duplicates=True
            )

            res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=False, callback=callback)
            
            st.session_state.convergence_history = callback.history
        st.success("Optimization finished ✅")

        # Extract all Pareto solutions
        with st.spinner("Extracting Pareto-optimal solutions..."):
            if res.X is None or len(res.X) == 0:
                st.error("Optimization returned no solutions.")
                st.stop()
            
            # Store all solutions
            all_solutions = []
            if res.X.ndim == 1:
                res.X = res.X.reshape(1, -1)
                res.F = res.F.reshape(1, -1)
            
            for i, (sol, obj) in enumerate(zip(res.X, res.F)):
                all_solutions.append({
                    'solution_id': i,
                    'obj1': float(obj[0]),
                    'obj2': float(obj[1]),
                    'obj3': float(obj[2]) if len(obj) > 2 else 0.0,
                    'indices': np.array(np.round(sol), dtype=int).tolist()
                })
            
            st.session_state.all_solutions = all_solutions

            # Use best solution (minimize first objective)
            best_idx = np.argmin(res.F[:, 0])
            best_solution_int = np.array(np.round(res.X[best_idx]), dtype=int)
            best_solution_int = np.clip(best_solution_int, 0, len(df_future)-1)

            scheduled_slots = df_future.loc[best_solution_int,
                                ['timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 'wind_speed', 'temperature']].copy()
            scheduled_slots['job_id'] = range(1, N_JOBS + 1)
            scheduled_slots = scheduled_slots.sort_values('timestamp').reset_index(drop=True)
            st.session_state.scheduled_slots = scheduled_slots

            avg_carbon = scheduled_slots['carbon_intensity'].mean()
            avg_solar = scheduled_slots['solar_cloud_pct'].mean()
            avg_wind = scheduled_slots['wind_speed'].mean()
            region_distribution = scheduled_slots['region'].value_counts()
            
            # Calculate total carbon emissions
            job_energy_kwh = avg_job_power_kw * (avg_job_duration_min / 60.0)
            total_carbon_emissions_g = (scheduled_slots['carbon_intensity'] * job_energy_kwh).sum()
            total_carbon_emissions_kg = total_carbon_emissions_g / 1000

        # Show metrics and tables
        st.header("📋 Optimized Schedule Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Carbon Intensity", f"{avg_carbon:.2f} gCO₂/kWh", help="Average carbon intensity across all scheduled slots")
        col2.metric("Total Emissions", f"{total_carbon_emissions_kg:.3f} kg CO₂", help=f"Total carbon emissions: {N_JOBS} jobs × {job_energy_kwh:.3f} kWh × avg intensity")
        col3.metric("Avg Solar Cloud Cover", f"{avg_solar:.1f} %")
        col4.metric("Avg Wind Speed", f"{avg_wind:.1f} m/s")

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

        st.success("✅ Full pipeline completed! Navigate to Analytics or Export pages for detailed results.")
        st.balloons()
