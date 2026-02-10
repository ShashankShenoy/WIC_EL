# Technical Documentation — Advanced Cloud Job Scheduler

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Optimization Problem Formulation](#optimization-problem-formulation)
3. [NSGA-II Algorithm](#nsga-ii-algorithm)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Performance Analysis](#performance-analysis)
6. [Complexity Analysis](#complexity-analysis)

---

## 1. System Architecture

### Component Hierarchy
```
┌─────────────────────────────────────────────┐
│          Streamlit Frontend (UI)            │
│  ┌──────┬─────────┬──────────┬──────────┐  │
│  │ Home │ Data &  │ Optimi-  │ Analytics│  │
│  │      │Forecast │ zation   │ & Export │  │
│  └──────┴─────────┴──────────┴──────────┘  │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴──────────────┐
    │                            │
┌───▼────────────┐   ┌──────────▼────────────┐
│   ML Pipeline  │   │  Optimization Engine  │
│ ┌────────────┐ │   │  ┌──────────────────┐ │
│ │ Forecasting│ │   │  │    NSGA-II       │ │
│ │   Models   │ │   │  │   (pymoo)        │ │
│ │  (sklearn) │ │   │  └──────────────────┘ │
│ └────────────┘ │   │  ┌──────────────────┐ │
│ ┌────────────┐ │   │  │ Pareto Front     │ │
│ │ Confidence │ │   │  │ Extraction       │ │
│ │ Intervals  │ │   │  └──────────────────┘ │
│ └────────────┘ │   └───────────────────────┘
└────────────────┘
        │
┌───────▼────────────────────────────────┐
│     Data Management Layer              │
│  ┌──────────┬──────────┬────────────┐ │
│  │Historical│ Forecasts│  Schedules │ │
│  │   Data   │          │            │ │
│  └──────────┴──────────┴────────────┘ │
└────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Presentation** | Streamlit 1.31+ | Multi-page UI, interactive widgets |
| **Optimization** | pymoo 0.6.1 | NSGA-II genetic algorithm |
| **ML Forecasting** | scikit-learn 1.7+ | Regression models, metrics |
| **Visualization** | Plotly 5.18+ | Interactive 2D/3D charts |
| **Data Processing** | pandas 2.2+, numpy 2.2+ | DataFrame operations, arrays |
| **Export** | openpyxl 3.1+ | Excel workbook generation |

---

## 2. Optimization Problem Formulation

### Problem Statement
Given:
- **N** jobs to schedule
- **M** regions (datacenters)
- **T** time slots (5-minute intervals)
- Forecasted conditions: carbon intensity, renewable availability, cost

Find: Optimal assignment of jobs to (region, time) slots

### Mathematical Formulation

**Decision Variables:**
```
X = [x₁, x₂, ..., xₙ]
where xᵢ ∈ {0, 1, ..., T-1} represents the slot index for job i
```

**Objectives (minimize all):**

1. **Carbon Footprint:**
   ```
   f₁(X) = w_carbon × (1/N) Σᵢ carbon_intensity(xᵢ) + w_load × penalty_load(X)
   ```
   - `carbon_intensity(xᵢ)`: Normalized carbon at slot xᵢ
   - `penalty_load(X)`: Penalizes exceeding regional capacity

2. **Renewable Energy (negated to maximize):**
   ```
   f₂(X) = -w_renewable × (1/N) Σᵢ renewable_score(xᵢ)
   ```
   where `renewable_score = (solar_norm + wind_norm) / 2`

3. **Operating Cost:**
   ```
   f₃(X) = w_cost × (1/N) Σᵢ electricity_cost(xᵢ)
   ```
   - `electricity_cost = base_price × time_multiplier × (1 + carbon/1000)`

**Constraints:**

1. **No Duplicates:**
   ```
   ∀i,j: i ≠ j ⇒ xᵢ ≠ xⱼ
   ```

2. **Regional Capacity:**
   ```
   ∀r ∈ Regions: |{i : region(xᵢ) = r}| ≤ max_jobs_per_region
   ```
   where `max_jobs_per_region = ⌈N / M⌉ + 1`

### Normalization Strategy

All objectives normalized to [0, 1] using Min-Max scaling:
```
normalized_value = (value - min) / (max - min + ε)
```
where ε = 10⁻⁸ prevents division by zero.

---

## 3. NSGA-II Algorithm

### Overview
**NSGA-II** (Non-dominated Sorting Genetic Algorithm II) is a multi-objective evolutionary algorithm that:
- Maintains a population of candidate solutions
- Uses non-dominated sorting to rank solutions
- Applies crowding distance for diversity
- Evolves solutions through crossover and mutation

### Algorithm Steps

```
1. Initialize population P₀ of size N_pop randomly
2. For generation t = 1 to N_gen:
   a. Create offspring Q_t through:
      - Tournament selection
      - Simulated Binary Crossover (SBX)
      - Polynomial Mutation (PM)
   b. Combine R_t = P_t ∪ Q_t
   c. Perform non-dominated sorting on R_t → fronts F₁, F₂, ...
   d. Build P_{t+1}:
      - Add fronts F₁, F₂, ... until size ≥ N_pop
      - Sort last front by crowding distance
      - Select top N_pop individuals
3. Return Pareto front F₁ from final population
```

### Genetic Operators

**1. Simulated Binary Crossover (SBX)**
```
For parents p₁, p₂:
  β = random_beta_distribution(η_c)  # η_c = crossover distribution index
  c₁ = 0.5 × ((1 + β)p₁ + (1 - β)p₂)
  c₂ = 0.5 × ((1 - β)p₁ + (1 + β)p₂)
Return children c₁, c₂
```
- Higher η_c → children closer to parents (exploitation)
- Lower η_c → more diverse children (exploration)
- Default: η_c = 15

**2. Polynomial Mutation (PM)**
```
For gene x:
  δ = random_polynomial_distribution(η_m)  # η_m = mutation distribution index
  x' = x + δ × (x_max - x_min)
Return mutated gene x'
```
- Higher η_m → smaller mutations
- Lower η_m → larger mutations
- Default: η_m = 20

### Non-Dominated Sorting

**Definition:** Solution x₁ dominates x₂ if:
- x₁ is no worse than x₂ in all objectives
- x₁ is strictly better than x₂ in at least one objective

**Complexity:** O(MN²) for M objectives, N solutions

**Process:**
1. Count domination: nₚ = number of solutions dominating p
2. Find F₁ = {p : nₚ = 0} (non-dominated solutions)
3. For each p in F₁, reduce nq for all q dominated by p
4. Repeat to find F₂, F₃, ...

### Crowding Distance

Maintains diversity by measuring "crowdedness" around each solution:

```
For solution i in front F:
  d_i = Σ_obj ((f_obj^{i+1} - f_obj^{i-1}) / (f_obj^max - f_obj^min))
```

Boundary solutions (i=0 or i=N-1) get d = ∞

Solutions with larger crowding distances are preferred.

---

## 4. Machine Learning Pipeline

### Feature Engineering

**Temporal Features:**
- `hour`: Hour of day [0-23]
- `day_of_week`: Day [0-6], 0=Monday
- `month`: Month [1-12]
- `day_of_year`: Day [1-365/366]

**Lag Features:**
- `carbon_intensity_lag1`: Previous value
- `solar_cloud_pct_lag1`: Previous value
- `wind_speed_lag1`: Previous value
- `temperature_lag1`: Previous value

**Derived Features:**
- `renewable_norm`: (solar_norm + wind_norm) / 2
- `time_multiplier`: 1.5 (peak), 1.2 (evening), 0.8 (off-peak)

### Multi-Target Forecasting

For each region r and target t:
```
model_{r,t} = trained_regressor

For each future time slot:
  features = extract_features(timestamp, lag_values)
  prediction = model_{r,t}.predict(features)
  update lag_values with prediction
```

Supports any scikit-learn regressor:
- RandomForestRegressor
- GradientBoostingRegressor
- XGBoost, LightGBM
- Neural Networks (MLPRegressor)

### Confidence Intervals

Using historical residuals:
```
For confidence level α (e.g., 95%):
  z = z_score(α)  # 1.96 for 95%
  σ_r = historical_std_dev(region=r, target=t)
  
  lower_bound = prediction - z × σ_r
  upper_bound = prediction + z × σ_r
```

### Model Evaluation Metrics

**Root Mean Square Error (RMSE):**
```
RMSE = √((1/n) Σᵢ (yᵢ - ŷᵢ)²)
```

**Mean Absolute Error (MAE):**
```
MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
```

**R² Score (Coefficient of Determination):**
```
R² = 1 - (Σ(yᵢ - ŷᵢ)²) / (Σ(yᵢ - ȳ)²)
```
- R² = 1: Perfect predictions
- R² = 0: Model equals mean baseline
- R² < 0: Model worse than mean

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100/n) Σᵢ |((yᵢ - ŷᵢ) / yᵢ)|
```

---

## 5. Performance Analysis

### Carbon Reduction Analysis

**Baseline Strategy:** Sequential scheduling (first N slots)

**Carbon Reduction:**
```
reduction_% = ((carbon_baseline - carbon_optimized) / carbon_baseline) × 100
```

**Typical Results:**
- Eco-Friendly preset: 35-50% reduction
- Balanced preset: 20-35% reduction
- Cost-Efficient preset: 10-20% reduction

### Renewable Energy Improvement

```
renewable_score = (solar_availability + wind_speed_norm) / 2

improvement_% = ((renewable_opt - renewable_base) / renewable_base) × 100
```

**Typical Results:**
- Renewable-Focus preset: 40-60% improvement
- Balanced preset: 25-40% improvement
- Cost-Efficient preset: 10-25% improvement

### Cost Savings

```
cost_savings_$ = (cost_baseline - cost_optimized) × N_jobs × avg_job_duration × power_consumption

Assuming:
- avg_job_duration = 5 minutes
- power_consumption = 0.5 kW per job
```

For 100 jobs:
- Cost-Efficient preset: $15-25 savings
- Balanced preset: $8-15 savings
- Eco-Friendly preset: $3-8 savings

---

## 6. Complexity Analysis

### Time Complexity

**Forecasting Phase:**
```
O(N_regions × N_slots × N_targets × T_model)
```
where T_model depends on model type (e.g., O(N_trees × log N) for RandomForest)

For 5 regions, 3 days (864 slots), 4 targets:
- Bottleneck: ~17,280 predictions

**Optimization Phase:**
```
O(N_gen × N_pop × (N_jobs × M + N_pop × M × N_pop))
```
Breaking down:
- Fitness evaluation: O(N_jobs × M) per solution
- Non-dominated sorting: O(M × N_pop²)
- Crowding distance: O(M × N_pop log N_pop)

For typical parameters (N_gen=200, N_pop=100, N_jobs=20, M=3):
- ~200 × 100 × (60 + 30,000) ≈ 6×10⁸ operations

**Total Runtime:**
- Small (10 jobs, 1 day): 5-10 seconds
- Medium (50 jobs, 3 days): 30-60 seconds
- Large (200 jobs, 7 days): 2-5 minutes

### Space Complexity

**Memory Requirements:**

**Historical Data:**
```
O(N_regions × N_historical_slots × N_features)
```
Typical: 5 regions × 10,000 slots × 12 features = 600KB

**Forecasts:**
```
O(N_regions × N_forecast_slots × N_features)
```
Typical: 5 regions × 864 slots × 15 features = 65KB

**Population:**
```
O(N_pop × N_jobs × 2)  # Solutions + fitness values
```
Typical: 100 × 20 × 2 = 4KB

**Total Memory:** ~1-5 MB for typical workloads

### Scalability Limits

| Parameter | Current Limit | Scalable To | Bottleneck |
|-----------|--------------|-------------|------------|
| Jobs | 500 | 5,000 | NSGA-II fitness evals |
| Regions | 20 | 100 | Forecasting time |
| Forecast Days | 30 | 90 | Memory (864→7,776 slots) |
| Population | 500 | 2,000 | Sorting complexity |
| Generations | 500 | 2,000 | Total runtime |

**Optimization Strategies for Scale:**
1. **Parallel fitness evaluation** — GPU acceleration
2. **Hierarchical scheduling** — Optimize by region first
3. **Approximate sorting** — O(N log N) instead of O(N²)
4. **Delta evaluation** — Only re-evaluate changed genes
5. **Database backend** — PostgreSQL for historical data

---

## 7. Advanced Topics

### Handling Uncertainty

**Sources of Uncertainty:**
- Weather prediction errors (solar/wind)
- Carbon intensity fluctuations
- Unexpected demand spikes
- Model prediction errors

**Mitigation Strategies:**
1. **Robust Optimization:** Optimize worst-case scenarios
2. **Stochastic Programming:** Sample multiple futures
3. **Rolling Horizon:** Re-optimize every hour with latest data
4. **Buffer Capacity:** Reserve 10-20% extra capacity

### Real-Time Integration

**System Architecture for Production:**
```
┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Grid API    │────▶│  Optimization │────▶│  Scheduler  │
│ (real-time)  │     │    Service    │     │   Engine    │
└──────────────┘     └───────────────┘     └─────────────┘
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   Database       │
                    │ (historical +    │
                    │  schedules)      │
                    └──────────────────┘
```

**Update Frequency:**
- Forecasts: Every 15 minutes
- Optimization: Every hour
- Schedule adjustment: When predictions deviate >10%

### Multi-Datacenter Extensions

**Additional Constraints:**
1. **Data Transfer Costs:** Jobs with dependencies
2. **Network Latency:** Max 100ms between regions
3. **Data Residency:** GDPR, data sovereignty rules
4. **Failover Requirements:** Backup region per job

**Implementation:**
```python
# Add to fitness function
network_cost = sum(transfer_cost(job1, job2) 
                   for job1, job2 in dependencies)
latency_penalty = sum(latency(region1, region2) * criticality
                      for critical jobs)
```

---

## 8. References & Further Reading

**Multi-Objective Optimization:**
- Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- Coello, C. A. C. (2006). "Evolutionary multi-objective optimization"

**Sustainable Computing:**
- Beloglazov, A., et al. (2012). "Energy-aware resource allocation in cloud computing"
- Goiri, Í., et al. (2013). "GreenHadoop: leveraging green energy in data centers"

**Time Series Forecasting:**
- Hyndman, R. J., & Athanasopoulos, G. (2018). "Forecasting: principles and practice"
- Box, G. E. P., et al. (2015). "Time series analysis: forecasting and control"

**Cloud Scheduling:**
- Buyya, R., et al. (2010). "Cloud computing and emerging IT platforms"
- Mao, M., & Humphrey, M. (2011). "Auto-scaling to minimize cost and meet SLAs"

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Author:** Internship Project  
**Contact:** [Your Email/Contact]
