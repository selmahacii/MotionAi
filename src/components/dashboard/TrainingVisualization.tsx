'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { TrendingUp, Activity, Clock, Target, Zap } from 'lucide-react'

interface TrainingMetrics {
  epoch: number
  trainLoss: number
  valLoss: number
  accuracy: number
}

interface ModelPerformance {
  name: string
  metrics: {
    accuracy: number
    precision: number
    recall: number
    f1Score: number
    latency: number
    throughput: number
  }
  trainingHistory: TrainingMetrics[]
}

interface AnalyticsData {
  models: ModelPerformance[]
  inferenceStats: {
    totalInferences: number
    averageLatency: number
    p50Latency: number
    p95Latency: number
    p99Latency: number
    errors: number
  }
  classDistribution: Record<string, number>
}

const COLORS = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']

// Training Loss Chart
function TrainingChart({ data, title }: { data: TrainingMetrics[]; title: string }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="epoch" stroke="#888" fontSize={12} />
          <YAxis stroke="#888" fontSize={12} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(30, 41, 59, 0.95)', 
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px'
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="trainLoss" 
            stroke="#4ECDC4" 
            strokeWidth={2}
            dot={false}
            name="Train Loss"
          />
          <Line 
            type="monotone" 
            dataKey="valLoss" 
            stroke="#FF6B6B" 
            strokeWidth={2}
            dot={false}
            name="Val Loss"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// Accuracy Chart
function AccuracyChart({ data }: { data: TrainingMetrics[] }) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="epoch" stroke="#888" fontSize={12} />
          <YAxis stroke="#888" fontSize={12} domain={[0, 1]} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(30, 41, 59, 0.95)', 
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px'
            }}
          />
          <Area 
            type="monotone" 
            dataKey="accuracy" 
            stroke="#45B7D1" 
            fill="url(#accuracyGradient)"
            strokeWidth={2}
            name="Accuracy"
          />
          <defs>
            <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#45B7D1" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#45B7D1" stopOpacity={0}/>
            </linearGradient>
          </defs>
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// Class Distribution Chart
function ClassDistributionChart({ data }: { data: Record<string, number> }) {
  const chartData = Object.entries(data).map(([name, value]) => ({
    name: name.replace('_', ' '),
    value: Math.round(value * 100)
  }))
  
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis type="number" stroke="#888" fontSize={12} />
          <YAxis dataKey="name" type="category" stroke="#888" fontSize={11} width={80} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(30, 41, 59, 0.95)', 
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px'
            }}
            formatter={(value: number) => [`${value}%`, 'Percentage']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// Latency Distribution Pie
function LatencyDistribution({ stats }: { stats: { p50Latency: number; p95Latency: number; p99Latency: number; averageLatency: number } }) {
  const data = [
    { name: 'P50', value: stats.p50Latency },
    { name: 'P95', value: stats.p95Latency - stats.p50Latency },
    { name: 'P99', value: stats.p99Latency - stats.p95Latency }
  ]
  
  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={40}
            outerRadius={70}
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index]} />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(30, 41, 59, 0.95)', 
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px'
            }}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

// Model Metrics Card
function ModelMetricsCard({ model }: { model: ModelPerformance }) {
  const { metrics } = model
  
  return (
    <Card className="bg-slate-900/50 border-slate-700">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{model.name}</CardTitle>
        <CardDescription>Model Performance Metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">Accuracy</span>
              <span className="font-mono text-emerald-400">{(metrics.accuracy * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.accuracy * 100} className="h-1.5" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">Precision</span>
              <span className="font-mono text-blue-400">{(metrics.precision * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.precision * 100} className="h-1.5" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">Recall</span>
              <span className="font-mono text-purple-400">{(metrics.recall * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.recall * 100} className="h-1.5" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">F1 Score</span>
              <span className="font-mono text-orange-400">{(metrics.f1Score * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.f1Score * 100} className="h-1.5" />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-700">
          <div className="text-center">
            <p className="text-2xl font-bold text-primary">{metrics.latency.toFixed(1)}ms</p>
            <p className="text-xs text-muted-foreground">Latency</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-emerald-400">{metrics.throughput.toFixed(0)}</p>
            <p className="text-xs text-muted-foreground">Inferences/sec</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Main Component
export default function TrainingVisualization() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [selectedModel, setSelectedModel] = useState(0)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch('/api/analytics')
        if (response.ok) {
          const data = await response.json()
          setAnalytics(data)
        }
      } catch (error) {
        console.error('Failed to fetch analytics:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchAnalytics()
    const interval = setInterval(fetchAnalytics, 10000)
    return () => clearInterval(interval)
  }, [])
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="w-8 h-8 animate-pulse text-primary" />
      </div>
    )
  }
  
  if (!analytics) {
    return (
      <div className="text-center text-muted-foreground py-8">
        Failed to load analytics
      </div>
    )
  }
  
  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-emerald-400" />
              <span className="text-xs text-muted-foreground">Total Inferences</span>
            </div>
            <p className="text-2xl font-bold mt-1">{analytics.inferenceStats.totalInferences.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-muted-foreground">Avg Latency</span>
            </div>
            <p className="text-2xl font-bold mt-1">{analytics.inferenceStats.averageLatency.toFixed(1)}ms</p>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-muted-foreground">P95 Latency</span>
            </div>
            <p className="text-2xl font-bold mt-1">{analytics.inferenceStats.p95Latency.toFixed(1)}ms</p>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-orange-400" />
              <span className="text-xs text-muted-foreground">Error Rate</span>
            </div>
            <p className="text-2xl font-bold mt-1">{((analytics.inferenceStats.errors / analytics.inferenceStats.totalInferences) * 100).toFixed(2)}%</p>
          </CardContent>
        </Card>
      </div>
      
      {/* Model Metrics */}
      <div className="grid md:grid-cols-3 gap-4">
        {analytics.models.map((model, idx) => (
          <ModelMetricsCard key={model.name} model={model} />
        ))}
      </div>
      
      {/* Training Charts */}
      <Card className="bg-slate-900/50 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Training Progress</CardTitle>
              <CardDescription>Loss and accuracy over epochs</CardDescription>
            </div>
            <div className="flex gap-2">
              {analytics.models.map((model, idx) => (
                <Badge
                  key={model.name}
                  variant={selectedModel === idx ? 'default' : 'outline'}
                  className="cursor-pointer"
                  onClick={() => setSelectedModel(idx)}
                >
                  {model.name}
                </Badge>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="loss">
            <TabsList className="mb-4">
              <TabsTrigger value="loss">Loss Curves</TabsTrigger>
              <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            </TabsList>
            <TabsContent value="loss">
              <TrainingChart 
                data={analytics.models[selectedModel].trainingHistory} 
                title={`${analytics.models[selectedModel].name} Training`}
              />
            </TabsContent>
            <TabsContent value="accuracy">
              <AccuracyChart data={analytics.models[selectedModel].trainingHistory} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
      
      {/* Distribution Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="bg-slate-900/50 border-slate-700">
          <CardHeader>
            <CardTitle>Class Distribution</CardTitle>
            <CardDescription>Movement class usage breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            <ClassDistributionChart data={analytics.classDistribution} />
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900/50 border-slate-700">
          <CardHeader>
            <CardTitle>Latency Distribution</CardTitle>
            <CardDescription>Inference latency percentiles</CardDescription>
          </CardHeader>
          <CardContent>
            <LatencyDistribution stats={analytics.inferenceStats} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
