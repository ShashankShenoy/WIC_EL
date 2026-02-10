import { useState, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { 
  Cloud, Zap, TrendingDown, Activity, Settings, 
  Play, Download, BarChart3, Leaf, MapPin, Target
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { MetricCard } from '@/components/MetricCard'
import { FileUpload } from '@/components/FileUpload'
import { api, OptimizationConfig } from '@/services/api'
import { formatNumber, formatPercentage, downloadFile } from '@/lib/utils'

export default function Dashboard() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedSolution, setSelectedSolution] = useState(0)
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

  // Debug log results
  useEffect(() => {
    if (results) {
      console.log('Optimization Results:', results)
      console.log('Schedule:', results.schedule)
      console.log('Metrics:', results.metrics)
      console.log('Pareto Solutions:', results.pareto_solutions)
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

            {/* Results Section with Tabs */}
            {results && (
              <div ref={resultsRef} className="space-y-6">
                {/* Primary KPI Cards */}
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                  <MetricCard
                    title="Carbon Saved"
                    value={`${formatNumber(results.metrics?.carbon?.carbon_saved_kg || 0, 2)} kg`}
                    subtitle={`${formatPercentage(results.metrics?.carbon?.reduction_pct || 0)} reduction`}
                    delta={{
                      value: results.metrics?.carbon?.reduction_pct || 0,
                      isPositive: true
                    }}
                    icon={<Leaf className="h-8 w-8" />}
                  />
                  <MetricCard
                    title="Avg Carbon"
                    value={`${formatNumber(results.metrics?.carbon?.optimized_avg || 0, 2)} g/kWh`}
                    subtitle="Optimized intensity"
                    icon={<TrendingDown className="h-8 w-8" />}
                  />
                  <MetricCard
                    title="Solar Energy"
                    value={formatPercentage(results.metrics?.renewables?.optimized_solar || 0)}
                    subtitle="Solar utilization"
                    icon={<Zap className="h-8 w-8" />}
                  />
                  <MetricCard
                    title="Regions Used"
                    value={results.metrics?.distribution?.optimized_regions || 0}
                    subtitle="Cloud regions"
                    icon={<MapPin className="h-8 w-8" />}
                  />
                  <MetricCard
                    title="Pareto Solutions"
                    value={results.pareto_solutions?.length || 0}
                    subtitle="Optimal schedules"
                    icon={<Target className="h-8 w-8" />}
                  />
                </div>

                {/* Tabbed Interface */}
                <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="w-full grid grid-cols-4 lg:w-auto lg:inline-flex">
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="optimization">Optimization</TabsTrigger>
                    <TabsTrigger value="analytics">Analytics</TabsTrigger>
                    <TabsTrigger value="export">Export</TabsTrigger>
                  </TabsList>

                  {/* Overview Tab */}
                  <TabsContent value="overview">
                    {/* Enhanced Statistics Grid */}
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-5">
                        <h4 className="font-medium text-sm text-zinc-500 mb-2">Best Carbon Score</h4>
                        <p className="text-3xl font-semibold text-white">
                          {formatNumber(results.best_solution?.obj_carbon || 0, 4)}
                        </p>
                        <p className="text-xs text-zinc-600 mt-1">Minimized objective</p>
                      </div>
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-5">
                        <h4 className="font-medium text-sm text-zinc-500 mb-2">Total Emissions</h4>
                        <p className="text-3xl font-semibold text-white">
                          {formatNumber(results.metrics?.carbon?.optimized_total_kg || 0, 2)}
                        </p>
                        <p className="text-xs text-zinc-600 mt-1">kg CO₂ (optimized)</p>
                      </div>
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-5">
                        <h4 className="font-medium text-sm text-zinc-500 mb-2">Consistency</h4>
                        <p className="text-3xl font-semibold text-white">
                          {formatNumber(results.metrics?.consistency?.optimized_std || 0, 2)}
                        </p>
                        <p className="text-xs text-zinc-600 mt-1">Std dev (g/kWh)</p>
                      </div>
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-5">
                        <h4 className="font-medium text-sm text-zinc-500 mb-2">Generations</h4>
                        <p className="text-3xl font-semibold text-white">
                          {results.convergence?.length || 0}
                        </p>
                        <p className="text-xs text-zinc-600 mt-1">Evolution cycles</p>
                      </div>
                    </div>

                    {/* Charts Section */}
                    <div className="grid gap-4 md:grid-cols-2">
                      {/* Carbon Intensity Timeline */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            <Activity className="h-4 w-4 text-zinc-500" />
                            Carbon Intensity Over Time
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <Plot
                            data={[
                              {
                                x: (results.schedule || []).map((job: any) => job.timestamp),
                                y: (results.schedule || []).map((job: any) => job.carbon_intensity),
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: { color: '#10b981', size: 4 },
                                line: { color: '#10b981', width: 2 },
                                name: 'Optimized'
                              }
                            ]}
                            layout={{
                              autosize: true,
                              height: 300,
                              margin: { l: 50, r: 20, t: 20, b: 50 },
                              xaxis: { title: 'Time', showgrid: false, color: '#52525b' },
                              yaxis: { title: 'Carbon (g/kWh)', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                              paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'rgba(0,0,0,0)',
                              font: { size: 10, color: '#a1a1aa' }
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                          />
                        </CardContent>
                      </Card>

                      {/* Regional Distribution */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            <MapPin className="h-4 w-4 text-zinc-500" />
                            Jobs by Region
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <Plot
                            data={[
                              {
                                values: Object.values((results.schedule || []).reduce((acc: any, job: any) => {
                                  acc[job.region] = (acc[job.region] || 0) + 1
                                  return acc
                                }, {})),
                                labels: Object.keys((results.schedule || []).reduce((acc: any, job: any) => {
                                  acc[job.region] = (acc[job.region] || 0) + 1
                                  return acc
                                }, {})),
                                type: 'pie',
                                marker: {
                                  colors: ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16']
                                },
                                textfont: { color: '#ffffff' },
                                hoverinfo: 'label+value+percent'
                              }
                            ]}
                            layout={{
                              autosize: true,
                              height: 300,
                              margin: { l: 20, r: 20, t: 20, b: 20 },
                              paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'rgba(0,0,0,0)',
                              font: { size: 10, color: '#a1a1aa' },
                              showlegend: true,
                              legend: { font: { color: '#a1a1aa' } }
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                          />
                        </CardContent>
                      </Card>
                    </div>

                    {/* Schedule Preview */}
                    <Card className="bg-zinc-950 border border-zinc-800 overflow-hidden mt-6">
                      <CardHeader className="border-b border-zinc-800">
                        <CardTitle className="text-lg font-medium text-white flex items-center gap-2">
                          <BarChart3 className="h-5 w-5 text-zinc-500" />
                          Schedule Preview (first 10 jobs)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="overflow-x-auto rounded-lg border border-zinc-800">
                          <table className="w-full text-sm">
                            <thead className="bg-zinc-900 border-b border-zinc-800">
                              <tr>
                                <th className="p-3 text-left font-medium text-zinc-400">Job ID</th>
                                <th className="p-3 text-left font-medium text-zinc-400">Timestamp</th>
                                <th className="p-3 text-left font-medium text-zinc-400">Region</th>
                                <th className="p-3 text-right font-medium text-zinc-400">Carbon (g/kWh)</th>
                                <th className="p-3 text-right font-medium text-zinc-400">Solar (%)</th>
                                <th className="p-3 text-right font-medium text-zinc-400">Wind (m/s)</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-zinc-800">
                              {(results.schedule || []).slice(0, 10).map((job: any) => (
                                <tr key={job.job_id} className="hover:bg-zinc-900/50 transition-colors">
                                  <td className="p-3 font-medium text-white">{job.job_id}</td>
                                  <td className="p-3 text-zinc-500 text-xs">{new Date(job.timestamp).toLocaleString()}</td>
                                  <td className="p-3">
                                    <span className="rounded bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300">
                                      {job.region}
                                    </span>
                                  </td>
                                  <td className="p-3 text-right font-medium text-white">{formatNumber(job.carbon_intensity)}</td>
                                  <td className="p-3 text-right text-zinc-500">{formatNumber(job.solar_cloud_pct)}%</td>
                                  <td className="p-3 text-right text-zinc-500">{formatNumber(job.wind_speed)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  {/* Optimization Tab */}
                  <TabsContent value="optimization">
                    <div className="grid gap-4">
                      {/* 3D Pareto Front and 2D Projections */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            <Target className="h-4 w-4 text-zinc-500" />
                            Pareto-Optimal Solutions
                          </CardTitle>
                          <p className="text-sm text-zinc-500 mt-2">All non-dominated solutions showing trade-offs between objectives</p>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <Plot
                            data={[
                              {
                                x: (results.pareto_solutions || []).map((s: any) => s.obj_carbon),
                                y: (results.pareto_solutions || []).map((s: any) => -s.obj_renewable),
                                z: (results.pareto_solutions || []).map((s: any) => s.obj_cost),
                                type: 'scatter3d',
                                mode: 'markers',
                                marker: {
                                  size: 6,
                                  color: (results.pareto_solutions || []).map((s: any) => s.obj_carbon),
                                  colorscale: 'Viridis',
                                  showscale: true,
                                  colorbar: { title: 'Carbon', len: 0.5, tickfont: { color: '#a1a1aa' } }
                                },
                                text: (results.pareto_solutions || []).map((_s: any, i: number) => `Solution ${i + 1}`),
                                hovertemplate: '<b>%{text}</b><br>Carbon: %{x:.4f}<br>Renewable: %{y:.4f}<br>Cost: %{z:.4f}<extra></extra>'
                              },
                              {
                                x: [results.best_solution?.obj_carbon || 0],
                                y: [-(results.best_solution?.obj_renewable || 0)],
                                z: [results.best_solution?.obj_cost || 0],
                                type: 'scatter3d',
                                mode: 'markers',
                                marker: { size: 10, color: '#ef4444', symbol: 'diamond' },
                                name: 'Best Solution',
                                hovertemplate: '<b>Best Solution</b><br>Carbon: %{x:.4f}<br>Renewable: %{y:.4f}<br>Cost: %{z:.4f}<extra></extra>'
                              }
                            ]}
                            layout={{
                              autosize: true,
                              height: 500,
                              margin: { l: 0, r: 0, t: 0, b: 0 },
                              scene: {
                                xaxis: { title: 'Carbon Score (minimize)', gridcolor: '#18181b', color: '#52525b' },
                                yaxis: { title: 'Renewable Score (maximize)', gridcolor: '#18181b', color: '#52525b' },
                                zaxis: { title: 'Cost Score (minimize)', gridcolor: '#181b', color: '#52525b' },
                                bgcolor: 'rgba(0,0,0,0)'
                              },
                              paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'rgba(0,0,0,0)',
                              showlegend: true,
                              legend: { x: 1, y: 1, xanchor: 'right', font: { color: '#a1a1aa' } },
                              font: { size: 10, color: '#a1a1aa' }
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                          />
                        </CardContent>
                      </Card>

                      {/* 2D Trade-off Charts */}
                      <div className="grid gap-4 md:grid-cols-2">
                        <Card className="bg-zinc-950 border border-zinc-800">
                          <CardHeader className="border-b border-zinc-800">
                            <CardTitle className="text-sm font-medium text-white">Carbon vs Renewable Trade-off</CardTitle>
                          </CardHeader>
                          <CardContent className="pt-4">
                            <Plot
                              data={[
                                {
                                  x: (results.pareto_solutions || []).map((s: any) => s.obj_carbon),
                                  y: (results.pareto_solutions || []).map((s: any) => -s.obj_renewable),
                                  type: 'scatter',
                                  mode: 'markers',
                                  marker: { 
                                    size: 8, 
                                    color: (results.pareto_solutions || []).map((s: any) => s.obj_carbon),
                                    colorscale: 'Viridis',
                                    showscale: false
                                  }
                                }
                              ]}
                              layout={{
                                autosize: true,
                                height: 300,
                                margin: { l: 50, r: 20, t: 20, b: 50 },
                                xaxis: { title: 'Carbon Score', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                                yaxis: { title: 'Renewable Score', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { size: 10, color: '#a1a1aa' }
                              }}
                              config={{ displayModeBar: false, responsive: true }}
                              style={{ width: '100%' }}
                            />
                          </CardContent>
                        </Card>

                        <Card className="bg-zinc-950 border border-zinc-800">
                          <CardHeader className="border-b border-zinc-800">
                            <CardTitle className="text-sm font-medium text-white">Carbon vs Cost Trade-off</CardTitle>
                          </CardHeader>
                          <CardContent className="pt-4">
                            <Plot
                              data={[
                                {
                                  x: (results.pareto_solutions || []).map((s: any) => s.obj_carbon),
                                  y: (results.pareto_solutions || []).map((s: any) => s.obj_cost),
                                  type: 'scatter',
                                  mode: 'markers',
                                  marker: { 
                                    size: 8, 
                                    color: (results.pareto_solutions || []).map((s: any) => s.obj_cost),
                                    colorscale: 'Plasma',
                                    showscale: false
                                  }
                                }
                              ]}
                              layout={{
                                autosize: true,
                                height: 300,
                                margin: { l: 50, r: 20, t: 20, b: 50 },
                                xaxis: { title: 'Carbon Score', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                                yaxis: { title: 'Cost Score', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { size: 10, color: '#a1a1aa' }
                              }}
                              config={{ displayModeBar: false, responsive: true }}
                              style={{ width: '100%' }}
                            />
                          </CardContent>
                        </Card>
                      </div>

                      {/* Convergence History */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            <TrendingDown className="h-4 w-4 text-zinc-500" />
                            NSGA-II Convergence Analysis
                          </CardTitle>
                          <p className="text-sm text-zinc-500 mt-2">Algorithm learning progress over generations</p>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <Plot
                            data={[
                              {
                                x: results.convergence.map((_c: any, i: number) => i + 1),
                                y: results.convergence.map((c: any) => c.best_carbon),
                                type: 'scatter',
                                mode: 'lines',
                                line: { color: '#71717a', width: 3 },
                                fill: 'tozeroy',
                                fillcolor: 'rgba(113, 113, 122, 0.1)',
                                name: 'Best Carbon Score'
                              }
                            ]}
                            layout={{
                              autosize: true,
                              height: 300,
                              margin: { l: 50, r: 20, t: 20, b: 50 },
                              xaxis: { title: 'Generation', showgrid: false, color: '#52525b' },
                              yaxis: { title: 'Best Carbon Score', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                              paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'rgba(0,0,0,0)',
                              font: { size: 10, color: '#a1a1aa' }
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                          />
                        </CardContent>
                      </Card>

                      {/* Solution Explorer */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            Explore Solutions
                          </CardTitle>
                          <p className="text-sm text-zinc-500 mt-2">Browse through different Pareto-optimal schedules</p>
                        </CardHeader>
                        <CardContent className="pt-4 space-y-4">
                          <div>
                            <label className="text-sm font-medium text-zinc-300 mb-2 block">
                              Solution #{selectedSolution + 1} of {results.pareto_solutions?.length || 0}
                            </label>
                            <input
                              type="range"
                              min="0"
                              max={results.pareto_solutions?.length ? results.pareto_solutions.length - 1 : 0}
                              value={selectedSolution}
                              onChange={(e) => setSelectedSolution(Number(e.target.value))}
                              className="w-full accent-zinc-400"
                            />
                          </div>
                          
                          <div className="grid grid-cols-3 gap-4">
                            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
                              <h5 className="text-xs font-medium text-zinc-500 mb-1">Carbon Score</h5>
                              <p className="text-2xl font-semibold text-white">
                                {formatNumber(results.pareto_solutions[selectedSolution]?.obj_carbon, 4)}
                              </p>
                            </div>
                            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
                              <h5 className="text-xs font-medium text-zinc-500 mb-1">Renewable Score</h5>
                              <p className="text-2xl font-semibold text-white">
                                {formatNumber(-results.pareto_solutions[selectedSolution]?.obj_renewable, 4)}
                              </p>
                            </div>
                            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
                              <h5 className="text-xs font-medium text-zinc-500 mb-1">Cost Score</h5>
                              <p className="text-2xl font-semibold text-white">
                                {formatNumber(results.pareto_solutions[selectedSolution]?.obj_cost, 4)}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </TabsContent>

                  {/* Analytics Tab */}
                  <TabsContent value="analytics">
                    <div className="space-y-6">
                      {/* Baseline Comparison */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            Baseline vs Optimized Comparison
                          </CardTitle>
                          <p className="text-sm text-zinc-500 mt-2">Performance improvement over naive sequential scheduling</p>
                        </CardHeader>
                        <CardContent className="pt-4">
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                              <thead className="border-b border-zinc-800">
                                <tr>
                                  <th className="p-3 text-left font-medium text-zinc-400">Metric</th>
                                  <th className="p-3 text-right font-medium text-zinc-400">Baseline</th>
                                  <th className="p-3 text-right font-medium text-zinc-400">Optimized</th>
                                  <th className="p-3 text-right font-medium text-zinc-400">Improvement</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-zinc-800">
                                <tr className="hover:bg-zinc-900/50">
                                  <td className="p-3 text-zinc-300">Avg Carbon (g/kWh)</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.carbon?.baseline_avg || 0, 2)}</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.carbon?.optimized_avg || 0, 2)}</td>
                                  <td className="p-3 text-right text-white">{formatPercentage(-(results.metrics?.carbon?.reduction_pct || 0))}</td>
                                </tr>
                                <tr className="hover:bg-zinc-900/50">
                                  <td className="p-3 text-zinc-300">Total Emissions (kg)</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.carbon?.baseline_total_kg || 0, 2)}</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.carbon?.optimized_total_kg || 0, 2)}</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.carbon?.carbon_saved_kg || 0, 2)} kg saved</td>
                                </tr>
                                <tr className="hover:bg-zinc-900/50">
                                  <td className="p-3 text-zinc-300">Avg Solar (%)</td>
                                  <td className="p-3 text-right text-white">{formatPercentage(results.metrics?.renewables?.baseline_solar || 0)}</td>
                                  <td className="p-3 text-right text-white">{formatPercentage(results.metrics?.renewables?.optimized_solar || 0)}</td>
                                  <td className="p-3 text-right text-white">+{formatPercentage(results.metrics?.renewables?.solar_improvement || 0)}</td>
                                </tr>
                                <tr className="hover:bg-zinc-900/50">
                                  <td className="p-3 text-zinc-300">Carbon Std Dev</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.consistency?.baseline_std || 0, 2)}</td>
                                  <td className="p-3 text-right text-white">{formatNumber(results.metrics?.consistency?.optimized_std || 0, 2)}</td>
                                  <td className="p-3 text-right text-white">{formatPercentage(-(results.metrics?.consistency?.improvement_pct || 0))}</td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </CardContent>
                      </Card>

                      {/* Regional Analysis */}
                      <div className="grid gap-4 md:grid-cols-2">
                        <Card className="bg-zinc-950 border border-zinc-800">
                          <CardHeader className="border-b border-zinc-800">
                            <CardTitle className="text-sm font-medium text-white">Regional Carbon Intensity</CardTitle>
                          </CardHeader>
                          <CardContent className="pt-4">
                            <Plot
                              data={[
                                {
                                  x: Object.keys(results.schedule.reduce((acc: any, job: any) => {
                                    if (!acc[job.region]) acc[job.region] = []
                                    acc[job.region].push(job.carbon_intensity)
                                    return acc
                                  }, {})).map(region => region),
                                  y: Object.values(results.schedule.reduce((acc: any, job: any) => {
                                    if (!acc[job.region]) acc[job.region] = []
                                    acc[job.region].push(job.carbon_intensity)
                                    return acc
                                  }, {})).map((values: any) => 
                                    values.reduce((sum: number, v: number) => sum + v, 0) / values.length
                                  ),
                                  type: 'bar',
                                  marker: { 
                                    color: ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16']
                                  }
                                }
                              ]}
                              layout={{
                                autosize: true,
                                height: 300,
                                margin: { l: 50, r: 20, t: 20, b: 80 },
                                xaxis: { title: 'Region', showgrid: false, color: '#52525b' },
                                yaxis: { title: 'Avg Carbon (g/kWh)', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { size: 10, color: '#a1a1aa' }
                              }}
                              config={{ displayModeBar: false, responsive: true }}
                              style={{ width: '100%' }}
                            />
                          </CardContent>
                        </Card>

                        <Card className="bg-zinc-950 border border-zinc-800">
                          <CardHeader className="border-b border-zinc-800">
                            <CardTitle className="text-sm font-medium text-white">Schedule Timeline</CardTitle>
                          </CardHeader>
                          <CardContent className="pt-4">
                            <Plot
                              data={[
                                {
                                  x: (results.schedule || []).map((job: any) => job.timestamp),
                                  y: (results.schedule || []).map((job: any) => job.region),
                                  mode: 'markers',
                                  type: 'scatter',
                                  marker: {
                                    size: (results.schedule || []).map((job: any) => job.carbon_intensity / 20),
                                    color: (results.schedule || []).map((job: any) => job.carbon_intensity),
                                    colorscale: 'RdYlGn_r',
                                    showscale: true,
                                    colorbar: { title: 'Carbon', len: 0.5, tickfont: { color: '#a1a1aa' } }
                                  },
                                  hovertemplate: 'Job %{text}<br>Region: %{y}<br>Carbon: %{marker.color:.1f} g/kWh<extra></extra>',
                                  text: (results.schedule || []).map((job: any) => job.job_id)
                                }
                              ]}
                              layout={{
                                autosize: true,
                                height: 300,
                                margin: { l: 80, r: 20, t: 20, b: 50 },
                                xaxis: { title: 'Time', showgrid: false, color: '#52525b' },
                                yaxis: { title: '', showgrid: true, gridcolor: '#18181b', color: '#52525b' },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { size: 10, color: '#a1a1aa' }
                              }}
                              config={{ displayModeBar: false, responsive: true }}
                              style={{ width: '100%' }}
                            />
                          </CardContent>
                        </Card>
                      </div>

                      {/* Business Insights */}
                      <Card className="bg-zinc-950 border border-zinc-800">
                        <CardHeader className="border-b border-zinc-800">
                          <CardTitle className="text-base flex items-center gap-2 font-medium text-white">
                            Business Insights & Recommendations
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-4 space-y-4">
                          <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-4">
                            <h5 className="text-sm font-medium text-white mb-2">Environmental Impact</h5>
                            <p className="text-sm text-zinc-400 mb-3">
                              Optimization reduced carbon emissions by {formatPercentage(results.metrics.carbon.reduction_pct)}, 
                              saving {formatNumber(results.metrics.carbon.carbon_saved_kg, 2)} kg CO₂. This is equivalent to:
                            </p>
                            <div className="grid grid-cols-2 gap-3">
                              <div className="rounded bg-zinc-800 p-3">
                                <p className="text-xs text-zinc-500">Trees needed for absorption</p>
                                <p className="text-lg font-semibold text-white">
                                  {formatNumber(results.metrics.carbon.carbon_saved_kg * 0.00022, 1)} trees
                                </p>
                              </div>
                              <div className="rounded bg-zinc-800 p-3">
                                <p className="text-xs text-zinc-500">Financial value ($50/ton)</p>
                                <p className="text-lg font-semibold text-white">
                                  ${formatNumber(results.metrics.carbon.carbon_saved_kg * 0.05, 2)}
                                </p>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <h5 className="text-sm font-medium text-white">Recommendations</h5>
                            <ul className="space-y-2 text-sm text-zinc-400">
                              <li className="flex items-start gap-2">
                                <span className="text-zinc-500">•</span>
                                <span>Migrate more workloads to low-carbon regions during off-peak hours</span>
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-zinc-500">•</span>
                                <span>Implement dynamic pricing to incentivize flexible job scheduling</span>
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-zinc-500">•</span>
                                <span>Monitor real-time carbon intensity and adjust schedules adaptively</span>
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-zinc-500">•</span>
                                <span>Set carbon budgets per region/datacenter to drive accountability</span>
                              </li>
                            </ul>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </TabsContent>

                  {/* Export Tab */}
                  <TabsContent value="export">
                    <Card className="bg-zinc-950 border border-zinc-800">
                      <CardHeader className="border-b border-zinc-800">
                        <CardTitle className="text-base font-medium text-white">
                          Export Optimized Schedule
                        </CardTitle>
                        <p className="text-sm text-zinc-500 mt-2">Download your schedule in multiple formats</p>
                      </CardHeader>
                      <CardContent className="pt-6">
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
                            <h5 className="text-sm font-medium text-white mb-2">JSON Format</h5>
                            <p className="text-xs text-zinc-500 mb-4">
                              API integration, web services, programmatic access
                            </p>
                            <Button
                              onClick={() => handleExport('json')}
                              className="w-full"
                            >
                              <Download className="h-4 w-4 mr-2" />
                              Download JSON
                            </Button>
                          </div>

                          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
                            <h5 className="text-sm font-medium text-white mb-2">CSV Format</h5>
                            <p className="text-xs text-zinc-500 mb-4">
                              Excel, Google Sheets, data analysis tools
                            </p>
                            <Button
                              onClick={() => handleExport('csv')}
                              className="w-full"
                            >
                              <Download className="h-4 w-4 mr-2" />
                              Download CSV
                            </Button>
                          </div>
                        </div>

                        <div className="mt-6 rounded-lg border border-zinc-800 bg-zinc-900 p-6">
                          <h5 className="text-sm font-medium text-white mb-4">Summary Report</h5>
                          <div className="space-y-3 text-sm text-zinc-400">
                            <div className="flex justify-between">
                              <span>Jobs Scheduled:</span>
                              <span className="text-white font-medium">{results.schedule.length}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Regions Used:</span>
                              <span className="text-white font-medium">{results.metrics.distribution.optimized_regions}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Avg Carbon Intensity:</span>
                              <span className="text-white font-medium">{formatNumber(results.metrics.carbon.optimized_avg, 2)} g/kWh</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Carbon Saved:</span>
                              <span className="text-white font-medium">{formatNumber(results.metrics.carbon.carbon_saved_kg, 2)} kg CO₂</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Pareto Solutions:</span>
                              <span className="text-white font-medium">{results.pareto_solutions.length}</span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
