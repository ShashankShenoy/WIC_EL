import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

export interface OptimizationConfig {
  n_jobs: number
  forecast_days: number
  confidence_level: number
  weight_carbon: number
  weight_renewable: number
  weight_cost: number
  weight_load: number
  pop_size: number
  n_gen: number
  crossover_prob: number
  mutation_prob: number
  crossover_eta: number
  mutation_eta: number
  avg_job_power_kw: number
  avg_job_duration_min: number
}

export const api = {
  uploadData: async (csvFile: File, pklFile: File) => {
    const formData = new FormData()
    formData.append('historical_csv', csvFile)
    formData.append('trained_model', pklFile)
    
    const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  generateForecast: async (sessionId: string, config: OptimizationConfig) => {
    const response = await axios.post(`${API_BASE_URL}/api/forecast/${sessionId}`, config)
    return response.data
  },

  runOptimization: async (sessionId: string, config: OptimizationConfig) => {
    const response = await axios.post(`${API_BASE_URL}/api/optimize/${sessionId}`, config)
    return response.data
  },

  getAnalytics: async (sessionId: string) => {
    const response = await axios.get(`${API_BASE_URL}/api/analytics/${sessionId}`)
    return response.data
  },

  exportSchedule: async (sessionId: string, format: 'json' | 'csv' = 'json') => {
    const response = await axios.get(`${API_BASE_URL}/api/export/${sessionId}?format=${format}`, {
      responseType: format === 'csv' ? 'blob' : 'json'
    })
    return response.data
  }
}
