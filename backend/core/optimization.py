"""Multi-objective optimization using NSGA-II"""
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from sklearn.preprocessing import MinMaxScaler

class ConvergenceCallback(Callback):
    """Track convergence history during optimization"""
    def __init__(self):
        super().__init__()
        self.history = []
    
    def notify(self, algorithm):
        self.history.append({
            'n_gen': algorithm.n_gen,
            'min_f1': float(algorithm.pop.get("F")[:, 0].min()),
            'min_f2': float(algorithm.pop.get("F")[:, 1].min()),
            'avg_f1': float(algorithm.pop.get("F")[:, 0].mean()),
            'avg_f2': float(algorithm.pop.get("F")[:, 1].mean()),
        })

class CloudSchedulingProblem(Problem):
    """Multi-objective cloud scheduling problem"""
    def __init__(self, df, n_jobs, max_jobs_per_region, weight_carbon, weight_renewable, 
                 weight_cost, weight_load, job_power_kw, job_duration_min):
        self.df = df.reset_index(drop=True)
        self.n_jobs = n_jobs
        self.max_jobs_per_region = max_jobs_per_region
        self.weight_carbon = weight_carbon
        self.weight_renewable = weight_renewable
        self.weight_cost = weight_cost
        self.weight_load = weight_load
        self.job_power_kw = job_power_kw
        self.job_duration_min = job_duration_min
        self.job_energy_kwh = job_power_kw * (job_duration_min / 60.0)
        
        n_var = n_jobs
        super().__init__(n_var=n_var, n_obj=3, n_constr=1, xl=0, xu=len(df)-1, type_var=int)
    
    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        G = []
        for sol in X:
            sol_int = np.array(np.round(sol), dtype=int)
            sol_int = np.clip(sol_int, 0, len(self.df)-1)
            
            carbon = np.array([self.df.loc[i, 'carbon_norm'] for i in sol_int])
            renewable = np.array([self.df.loc[i, 'renewable_norm'] for i in sol_int])
            cost = np.array([self.df.loc[i, 'cost_norm'] for i in sol_int])
            
            # Get actual carbon intensity for proper emission calculation
            carbon_intensity_actual = np.array([self.df.loc[i, 'carbon_intensity'] for i in sol_int])
            total_carbon_g = (carbon_intensity_actual * self.job_energy_kwh).sum()
            total_carbon_kg = total_carbon_g / 1000.0
            
            # Duplicate constraint
            duplicates = len(sol_int) - len(np.unique(sol_int))
            
            # Region load balancing
            region_counts = self.df.loc[sol_int, 'region'].value_counts()
            load_penalty = sum(max(0, count - self.max_jobs_per_region) for count in region_counts)
            
            # Objectives (all minimized)
            obj_carbon = self.weight_carbon * (total_carbon_kg / self.n_jobs) + self.weight_load * load_penalty
            obj_renewable = -self.weight_renewable * renewable.mean()  # Negative to maximize renewables
            obj_cost = self.weight_cost * cost.mean()
            
            F.append([obj_carbon, obj_renewable, obj_cost])
            G.append([duplicates])
        
        out["F"] = np.array(F)
        out["G"] = np.array(G)

def run_optimization(df_future, n_jobs, weight_carbon, weight_renewable, weight_cost, 
                    weight_load, pop_size, n_gen, crossover_prob, mutation_prob,
                    crossover_eta, mutation_eta, avg_job_power_kw, avg_job_duration_min):
    """
    Run NSGA-II multi-objective optimization
    
    Returns:
        dict with best_solution, pareto_solutions, convergence, and schedule
    """
    # Normalize cost
    cost_min = df_future['electricity_cost'].min()
    cost_max = df_future['electricity_cost'].max()
    df_future['cost_norm'] = (df_future['electricity_cost'] - cost_min) / (cost_max - cost_min + 1e-8)
    
    # Calculate max jobs per region
    region_counts = df_future.groupby('region')['slot_index'].count().to_dict()
    n_regions = len(region_counts)
    max_jobs_per_region = n_jobs // n_regions + (1 if n_jobs % n_regions != 0 else 0)
    
    # Define problem
    problem = CloudSchedulingProblem(
        df=df_future,
        n_jobs=n_jobs,
        max_jobs_per_region=max_jobs_per_region,
        weight_carbon=weight_carbon,
        weight_renewable=weight_renewable,
        weight_cost=weight_cost,
        weight_load=weight_load,
        job_power_kw=avg_job_power_kw,
        job_duration_min=avg_job_duration_min
    )
    
    # Setup algorithm
    callback = ConvergenceCallback()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float),
        mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float),
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=False, callback=callback)
    
    if res.X is None or len(res.X) == 0:
        raise ValueError("Optimization returned no solutions")
    
    # Extract solutions
    all_solutions = []
    if res.X.ndim == 1:
        res.X = res.X.reshape(1, -1)
        res.F = res.F.reshape(1, -1)
    
    for i, (sol, obj) in enumerate(zip(res.X, res.F)):
        all_solutions.append({
            'solution_id': i,
            'obj_carbon': float(obj[0]),
            'obj_renewable': float(obj[1]),
            'obj_cost': float(obj[2]) if len(obj) > 2 else 0.0,
            'indices': np.array(np.round(sol), dtype=int).tolist()
        })
    
    # Get best solution (lowest carbon)
    best_idx = np.argmin(res.F[:, 0])
    best_solution_int = np.array(np.round(res.X[best_idx]), dtype=int)
    best_solution_int = np.clip(best_solution_int, 0, len(df_future)-1)
    
    # Create schedule dataframe
    scheduled_slots = df_future.loc[best_solution_int,
                        ['timestamp', 'region', 'carbon_intensity', 'solar_cloud_pct', 
                         'wind_speed', 'temperature', 'electricity_cost']].copy()
    scheduled_slots['job_id'] = range(1, n_jobs + 1)
    scheduled_slots = scheduled_slots.sort_values('timestamp').reset_index(drop=True)
    
    return {
        'best_solution': {
            'obj_carbon': float(res.F[best_idx, 0]),
            'obj_renewable': float(res.F[best_idx, 1]),
            'obj_cost': float(res.F[best_idx, 2]) if res.F.shape[1] > 2 else 0.0,
            'indices': best_solution_int.tolist()
        },
        'pareto_solutions': all_solutions,
        'convergence': callback.history,
        'schedule': scheduled_slots
    }
