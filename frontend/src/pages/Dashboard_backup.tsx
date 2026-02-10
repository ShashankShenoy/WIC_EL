import { useState, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { 
  Cloud, Zap, TrendingDown, Activity, Settings, 
  Play, Download, BarChart3, Leaf, MapPin, Target
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { MetricCard } from '@/components/MetricCard'
import { FileUpload } from '@/components/FileUpload'
import { api, OptimizationConfig } from '@/services/api'
import { formatNumber, formatPercentage, downloadFile } from '@/lib/utils'

export default function Dashboard() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [config, setConfig] = useState<OptimizationConfig>({
    n_jobs: 20,
    forecast_days: 3,
    confidence_level: 95,
    weight_carbon: 0.5,
    weight_renewable: 0.3,
    weight_cost: 0.2,
    weight_load: 0.15,
    pop_size: 100,
    n_gen: 200,
    crossover_prob: 0.9,
    mutation_prob: 0.1,
    crossover_eta: 15.0,
    mutation_eta: 20.0,
    avg_job_power_kw: 0.5,
    avg_job_duration_min: 5,
  })
  const [results, setResults] = useState<any>(null)
  const [forecastData, setForecastData] = useState<any>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to results when optimization completes
  useEffect(() => {
    if (results && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [results])

  const handleUpload = async (csvFile: File, pklFile: File) => {
    const toastId = toast.loading('Uploading files...')
    try {
      setIsLoading(true)
      const response = await api.uploadData(csvFile, pklFile)
      setSessionId(response.session_id)
      toast.success(`Files uploaded successfully! ${response.records} records loaded.`, { id: toastId })
    } catch (error: any) {
      toast.error('Upload failed: ' + (error.response?.data?.detail || error.message), { id: toastId })
    } finally {
      setIsLoading(false)
    }
  }

  const handleForecast = async () => {
    if (!sessionId) return
    
    const toastId = toast.loading('Generating forecast...')
    try {
      setIsLoading(true)
      const response = await api.generateForecast(sessionId, config)
      setForecastData(response)
      toast.success(`Forecast generated: ${response.num_slots} slots across ${response.regions.length} regions`, { id: toastId })
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message
      
      // Handle session expiration
      if (error.response?.status === 404) {
        setSessionId(null)
        setForecastData(null)
        setResults(null)
        toast.error('Session expired. Please upload your files again to continue.', { id: toastId, duration: 5000 })
      } else {
        toast.error('Forecast failed: ' + errorMsg, { id: toastId })
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleOptimize = async () => {
    if (!sessionId) return
    
    const toastId = toast.loading('Running optimization... This may take a minute.')
    try {
      setIsLoading(true)
      const response = await api.runOptimization(sessionId, config)
      setResults(response)
      toast.success(`Optimization complete! Found ${response.pareto_solutions.length} Pareto-optimal solutions.`, { id: toastId })
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message
      
      // Handle session expiration
      if (error.response?.status === 404) {
        setSessionId(null)
        setForecastData(null)
        setResults(null)
        toast.error('Session expired. Please upload your files again to continue.', { id: toastId, duration: 5000 })
      } else {
        toast.error('Optimization failed: ' + errorMsg, { id: toastId })
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleExport = async (format: 'json' | 'csv') => {
    if (!sessionId) return
    
    try {
      const data = await api.exportSchedule(sessionId, format)
      if (format === 'json') {
        downloadFile(data, 'schedule.json', 'application/json')
      } else {
        const blob = new Blob([data], { type: 'text/csv' })
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = 'schedule.csv'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      }
      toast.success(`Schedule exported as ${format.toUpperCase()}`)
    } catch (error: any) {
      toast.error('Export failed: ' + (error.response?.data?.detail || error.message))
    }
  }

  return (
    <div className="min-h-screen bg-black relative">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-2.5 bg-zinc-900 rounded-xl">
                <Cloud className="h-6 w-6 text-zinc-400" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-white">
                  Carbon-Aware Cloud Scheduler
                </h1>
                <p className="text-sm text-zinc-500">Multi-Objective Optimization</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 text-xs font-medium text-zinc-400 bg-zinc-900 rounded-full">
                v2.0.0
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-4">
          {/* Sidebar - Configuration */}
          <div className="lg:col-span-1">
            <Card className="sticky top-24 bg-zinc-950 border border-zinc-800">
              <CardHeader className="border-b border-zinc-800">
                <CardTitle className="flex items-center gap-2 text-white text-base font-semibold">
                  <Settings className="h-4 w-4" />
                  Configuration
                </CardTitle>
                <CardDescription className="text-zinc-500 text-sm">Upload data and configure optimization</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* File Upload */}
                {!sessionId && (
                  <FileUpload onFilesSelected={handleUpload} />
                )}

                {sessionId && (
                  <>
                    <div className="rounded-lg bg-zinc-900 p-3 text-sm border border-zinc-800">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-1.5 rounded-full bg-zinc-500" />
                        <span className="font-medium text-zinc-400 text-xs">Session: {sessionId.slice(-8)}</span>
                      </div>
                      <p className="text-xs text-zinc-600 mt-1">
                        Active
                      </p>
                    </div>

                    {/* Config Sliders */}
                    <div className="space-y-6">
                      <div>
                        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">Job Configuration</h3>
                        
                        <div className="space-y-4">
                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Jobs to Schedule</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.n_jobs}</span>
                            </label>
                            <input
                              type="range"
                              min="1"
                              max="500"
                              value={config.n_jobs}
                              onChange={(e) => setConfig({...config, n_jobs: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Forecast Days</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.forecast_days}</span>
                            </label>
                            <input
                              type="range"
                              min="1"
                              max="30"
                              value={config.forecast_days}
                              onChange={(e) => setConfig({...config, forecast_days: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 mb-2 block">Job Power (kW)</label>
                            <input
                              type="number"
                              min="0.1"
                              max="10"
                              step="0.1"
                              value={config.avg_job_power_kw}
                              onChange={(e) => setConfig({...config, avg_job_power_kw: Number(e.target.value)})}
                              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-white text-sm focus:border-zinc-700 focus:outline-none transition-colors"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 mb-2 block">Job Duration (min)</label>
                            <input
                              type="number"
                              min="1"
                              max="120"
                              value={config.avg_job_duration_min}
                              onChange={(e) => setConfig({...config, avg_job_duration_min: Number(e.target.value)})}
                              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-white text-sm focus:border-zinc-700 focus:outline-none transition-colors"
                            />
                          </div>
                        </div>
                      </div>

                      <div>
                        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">Objective Weights</h3>
                        
                        <div className="space-y-4">
                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Carbon Weight</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.weight_carbon.toFixed(2)}</span>
                            </label>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={config.weight_carbon}
                              onChange={(e) => setConfig({...config, weight_carbon: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Renewable Weight</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.weight_renewable.toFixed(2)}</span>
                            </label>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={config.weight_renewable}
                              onChange={(e) => setConfig({...config, weight_renewable: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Cost Weight</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.weight_cost.toFixed(2)}</span>
                            </label>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={config.weight_cost}
                              onChange={(e) => setConfig({...config, weight_cost: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>
                        </div>
                      </div>

                      <div>
                        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">Genetic Algorithm</h3>
                        
                        <div className="space-y-4">
                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Population Size</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.pop_size}</span>
                            </label>
                            <input
                              type="range"
                              min="20"
                              max="200"
                              step="10"
                              value={config.pop_size}
                              onChange={(e) => setConfig({...config, pop_size: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Generations</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.n_gen}</span>
                            </label>
                            <input
                              type="range"
                              min="50"
                              max="500"
                              step="10"
                              value={config.n_gen}
                              onChange={(e) => setConfig({...config, n_gen: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Crossover Probability</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.crossover_prob.toFixed(2)}</span>
                            </label>
                            <input
                              type="range"
                              min="0.5"
                              max="1"
                              step="0.05"
                              value={config.crossover_prob}
                              onChange={(e) => setConfig({...config, crossover_prob: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>

                          <div>
                            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between mb-2">
                              <span>Mutation Probability</span>
                              <span className="text-xs font-semibold text-white bg-zinc-800 px-2 py-1 rounded">{config.mutation_prob.toFixed(2)}</span>
                            </label>
                            <input
                              type="range"
                              min="0.01"
                              max="0.5"
                              step="0.01"
                              value={config.mutation_prob}
                              onChange={(e) => setConfig({...config, mutation_prob: Number(e.target.value)})}
                              className="w-full accent-zinc-400"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="space-y-2">
                      <Button 
                        className="w-full" 
                        onClick={handleForecast}
                        disabled={isLoading}
                      >
                        <Activity className="mr-2 h-4 w-4" />
                        Generate Forecast
                      </Button>
                      
                      <Button 
                        className="w-full" 
                        onClick={handleOptimize}
                        disabled={isLoading || !forecastData}
                      >
                        <Play className="mr-2 h-4 w-4" />
                        Run Optimization
                      </Button>

                      {results && (
                        <div className="flex gap-2">
                          <Button 
                            variant="outline"
                            className="flex-1" 
                            onClick={() => handleExport('json')}
                            size="sm"
                          >
                            <Download className="mr-1 h-3 w-3" />
                            JSON
                          </Button>
                          <Button 
                            variant="outline"
                            className="flex-1" 
                            onClick={() => handleExport('csv')}
                            size="sm"
                          >
                            <Download className="mr-1 h-3 w-3" />
                            CSV
                          </Button>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {!results && !forecastData && (
              <Card className="bg-zinc-950 border border-zinc-800 overflow-hidden">
                <CardHeader className="border-b border-zinc-800">
                  <CardTitle className="text-xl font-semibold text-white">Welcome</CardTitle>
                  <CardDescription className="text-zinc-500">
                    Optimize cloud workloads for minimal carbon emissions
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 pt-6">
                  <p className="text-zinc-400 leading-relaxed text-sm">
                    Schedule cloud computing jobs to minimize carbon emissions while balancing cost and renewable energy usage.
                  </p>
                  <div className="bg-zinc-900 rounded-lg p-5 border border-zinc-800">
                    <h4 className="font-medium text-white mb-4 text-sm flex items-center gap-2">
                      <Leaf className="h-4 w-4 text-zinc-400" />
                      Getting Started
                    </h4>
                    <ol className="space-y-3 text-sm text-zinc-400">
                      <li className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-zinc-800 text-white flex items-center justify-center text-xs font-medium">
                          1
                        </span>
                        <span className="pt-0.5">Upload historical carbon intensity data (CSV)</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-zinc-800 text-white flex items-center justify-center text-xs font-medium">
                          2
                        </span>
                        <span className="pt-0.5">Upload trained ML models (PKL)</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-zinc-800 text-white flex items-center justify-center text-xs font-medium">
                          3
                        </span>
                        <span className="pt-0.5">Configure optimization parameters</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-zinc-800 text-white flex items-center justify-center text-xs font-medium">
                          4
                        </span>
                        <span className="pt-0.5">Run forecast and optimization</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-zinc-800 text-white flex items-center justify-center text-xs font-medium">
                          5
                        </span>
                        <span className="pt-0.5">Analyze results and export schedule</span>
                      </li>
                    </ol>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Forecast Preview */}
            {forecastData && !results && (
              <Card className="bg-zinc-950 border border-zinc-800 overflow-hidden">
                <CardHeader className="border-b border-zinc-800">
                  <CardTitle className="text-xl flex items-center gap-2 font-semibold text-white">
                    <Activity className="h-5 w-5 text-zinc-400" />
                    Forecast Generated
                  </CardTitle>
                  <CardDescription className="text-zinc-500">
                    {forecastData.num_slots} time slots forecasted
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-5 pt-6">
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-5">
                      <h4 className="font-medium text-sm text-zinc-500 mb-2">Time Slots</h4>
                      <p className="text-3xl font-semibold text-white">{forecastData.num_slots}</p>
                      <p className="text-xs text-zinc-600 mt-1">Available windows</p>
                    </div>
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-5">
                      <h4 className="font-medium text-sm text-zinc-500 mb-2">Regions</h4>
                      <p className="text-3xl font-semibold text-white">{forecastData.regions.length}</p>
                      <p className="text-xs text-zinc-600 mt-1">{forecastData.regions.join(', ')}</p>
                    </div>
                    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-5">
                      <h4 className="font-medium text-sm text-zinc-500 mb-2">Forecast Period</h4>
                      <p className="text-3xl font-semibold text-white">{config.forecast_days}</p>
                      <p className="text-xs text-zinc-600 mt-1">Days ahead</p>
                    </div>
                  </div>
                  <div className="bg-zinc-900 rounded-lg p-5 border border-zinc-800">
                    <h4 className="font-medium text-white mb-3 text-sm flex items-center gap-2">
                      <Play className="h-4 w-4 text-zinc-400" />
                      Next Step
                    </h4>
                    <p className="text-sm text-zinc-400 leading-relaxed">
                      Click "Run Optimization" to find the best carbon-aware schedule using NSGA-II algorithm.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* KPI Metrics */}
            {results && (
              <div ref={resultsRef}>
