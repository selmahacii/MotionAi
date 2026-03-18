'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Activity, Brain, Eye, TrendingUp, Zap, Play, Pause, RotateCcw,
  Cpu, Layers, Timer, AlertCircle, CheckCircle, Upload, Video,
  BarChart3, Clock, Target, Motion, Settings, Info, Download,
  LineChart, Database
} from 'lucide-react'

// Dynamic imports for heavy components
const TrainingVisualization = dynamic(
  () => import('@/components/dashboard/TrainingVisualization'),
  { loading: () => <div className="flex items-center justify-center h-64"><Activity className="w-8 h-8 animate-pulse text-primary" /></div> }
)

const VideoAnalysis = dynamic(
  () => import('@/components/dashboard/VideoAnalysis'),
  { loading: () => <div className="flex items-center justify-center h-64"><Video className="w-8 h-8 animate-pulse text-primary" /></div> }
)

// ============ Types ============
interface Keypoint {
  x: number
  y: number
  score: number
}

interface PoseResult {
  keypoints: Keypoint[]
  keypoint_names: string[]
}

interface ClassificationResult {
  predicted_class: number
  class_name: string
  confidence: number
  all_probabilities: number[]
}

interface PredictionResult {
  predicted_frames: number
  keypoints_per_frame: number
  predictions: Keypoint[][]
}

interface InferenceResponse {
  pose: PoseResult
  classification: ClassificationResult
  prediction: PredictionResult | null
  inference_time_ms: number
}

interface ModelMetrics {
  posenet: { accuracy: number; latency: number }
  classifier: { accuracy: number; latency: number }
  predictor: { mpjpe: number; latency: number }
}

// ============ Constants ============
const MOVEMENT_CLASSES = [
  // Static Poses
  'standing', 'sitting', 'lying_down', 'kneeling', 'crouching',
  // Locomotion
  'walking', 'running', 'jumping', 'hopping', 'crawling', 'climbing',
  // Upper Body
  'arms_raised', 'waving', 'clapping', 'punching', 'pushing', 'pulling', 'throwing', 'catching',
  // Lower Body
  'kicking', 'squatting', 'lunging', 'stretching',
  // Sports
  'golf_swing', 'baseball_swing', 'tennis_serve', 'tennis_forehand', 'basketball_shot', 'soccer_kick', 'swimming', 'bowling',
  // Exercise
  'push_up', 'sit_up', 'burpee', 'yoga_pose'
]

const MOVEMENT_CATEGORIES: Record<string, string[]> = {
  'Static Poses': ['standing', 'sitting', 'lying_down', 'kneeling', 'crouching'],
  'Locomotion': ['walking', 'running', 'jumping', 'hopping', 'crawling', 'climbing'],
  'Upper Body': ['arms_raised', 'waving', 'clapping', 'punching', 'pushing', 'pulling', 'throwing', 'catching'],
  'Lower Body': ['kicking', 'squatting', 'lunging', 'stretching'],
  'Sports': ['golf_swing', 'baseball_swing', 'tennis_serve', 'tennis_forehand', 'basketball_shot', 'soccer_kick', 'swimming', 'bowling'],
  'Exercise': ['push_up', 'sit_up', 'burpee', 'yoga_pose']
}

const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

const SKELETON_CONNECTIONS = [
  [0, 1], [0, 2], [1, 3], [2, 4],  // Head
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // Arms
  [5, 11], [6, 12], [11, 12],  // Torso
  [11, 13], [13, 15], [12, 14], [14, 16]  // Legs
]

const SKELETON_COLORS = [
  '#FF6B6B', '#FF6B6B', '#FF6B6B', '#FF6B6B',
  '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4',
  '#45B7D1', '#45B7D1', '#45B7D1',
  '#96CEB4', '#96CEB4', '#96CEB4', '#96CEB4'
]

// ============ Skeleton Visualization Component ============
function SkeletonVisualization({ 
  keypoints, 
  width = 400, 
  height = 400,
  showLabels = false,
  animationPhase = 0
}: { 
  keypoints: Keypoint[]
  width?: number
  height?: number
  showLabels?: boolean
  animationPhase?: number
}) {
  if (!keypoints || keypoints.length === 0) return null

  return (
    <svg 
      width={width} 
      height={height} 
      viewBox="0 0 1 1" 
      className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg"
      style={{ border: '1px solid rgba(255,255,255,0.1)' }}
    >
      {/* Background grid */}
      <defs>
        <pattern id="grid" width="0.1" height="0.1" patternUnits="userSpaceOnUse">
          <path d="M 0.1 0 L 0 0 0 0.1" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.005"/>
        </pattern>
        <filter id="glow">
          <feGaussianBlur stdDeviation="0.01" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      <rect width="1" height="1" fill="url(#grid)" />
      
      {/* Draw connections with animation */}
      {SKELETON_CONNECTIONS.map((connection, idx) => {
        const [i, j] = connection
        const kp1 = keypoints[i]
        const kp2 = keypoints[j]
        if (!kp1 || !kp2) return null
        
        const avgScore = (kp1.score + kp2.score) / 2
        const opacity = 0.4 + avgScore * 0.6
        
        return (
          <line
            key={`line-${idx}`}
            x1={kp1.x}
            y1={kp1.y}
            x2={kp2.x}
            y2={kp2.y}
            stroke={SKELETON_COLORS[idx]}
            strokeWidth={0.006 + animationPhase * 0.002}
            strokeLinecap="round"
            opacity={opacity}
            filter="url(#glow)"
          />
        )
      })}
      
      {/* Draw keypoints */}
      {keypoints.map((kp, idx) => {
        const size = 0.012 + kp.score * 0.008
        return (
          <g key={`kp-${idx}`}>
            <circle
              cx={kp.x}
              cy={kp.y}
              r={size}
              fill="#FFFFFF"
              stroke="#1E293B"
              strokeWidth={0.003}
              filter="url(#glow)"
            />
            {showLabels && (
              <text
                x={kp.x + 0.02}
                y={kp.y}
                fontSize="0.02"
                fill="rgba(255,255,255,0.6)"
              >
                {KEYPOINT_NAMES[idx]}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

// ============ Animated Prediction Player ============
function PredictionPlayer({ 
  predictions,
  currentFrame,
  onFrameChange 
}: { 
  predictions: Keypoint[][]
  currentFrame: number
  onFrameChange: (frame: number) => void
}) {
  const [isPlaying, setIsPlaying] = useState(false)
  
  useEffect(() => {
    if (isPlaying && predictions.length > 0) {
      const interval = setInterval(() => {
        onFrameChange((prev) => (prev + 1) % predictions.length)
      }, 150)
      return () => clearInterval(interval)
    }
  }, [isPlaying, predictions.length, onFrameChange])
  
  if (!predictions || predictions.length === 0) return null
  
  return (
    <div className="space-y-4">
      <div className="flex justify-center">
        <SkeletonVisualization 
          keypoints={predictions[currentFrame]} 
          width={280} 
          height={280}
          animationPhase={currentFrame / predictions.length}
        />
      </div>
      
      {/* Frame Timeline */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Frame {currentFrame + 1} of {predictions.length}</span>
          <span>{((currentFrame / predictions.length) * 100).toFixed(0)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={predictions.length - 1}
          value={currentFrame}
          onChange={(e) => onFrameChange(parseInt(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      
      {/* Playback Controls */}
      <div className="flex justify-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => onFrameChange(0)}
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
        <Button
          size="sm"
          onClick={() => setIsPlaying(!isPlaying)}
          className={isPlaying ? 'bg-red-500 hover:bg-red-600' : 'bg-emerald-500 hover:bg-emerald-600'}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => onFrameChange(predictions.length - 1)}
        >
          <TrendingUp className="w-4 h-4" />
        </Button>
      </div>
    </div>
  )
}

// ============ Probability Distribution Chart ============
function ProbabilityChart({ 
  probabilities,
  highlightedClass 
}: { 
  probabilities: number[]
  highlightedClass?: string 
}) {
  const sortedData = probabilities
    .map((prob, i) => ({ prob, index: i, name: MOVEMENT_CLASSES[i] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 10)
  
  return (
    <div className="space-y-3">
      {sortedData.map(({ prob, index, name }) => (
        <div 
          key={index} 
          className={`space-y-1 p-2 rounded-lg transition-all ${
            name === highlightedClass ? 'bg-primary/20 border border-primary/30' : ''
          }`}
        >
          <div className="flex justify-between text-sm">
            <span className="capitalize font-medium">{name.replace('_', ' ')}</span>
            <span className="font-mono text-primary">{(prob * 100).toFixed(1)}%</span>
          </div>
          <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary to-primary/60 rounded-full transition-all duration-300"
              style={{ width: `${prob * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

// ============ Model Performance Card ============
function PerformanceCard({
  title,
  metrics,
  icon: Icon
}: {
  title: string
  metrics: { label: string; value: string | number; unit?: string }[]
  icon: React.ElementType
}) {
  return (
    <Card className="bg-slate-900/50 border-slate-700">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5 text-primary" />
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {metrics.map((m, i) => (
            <div key={i} className="text-center">
              <p className="text-2xl font-bold">{m.value}</p>
              <p className="text-xs text-muted-foreground">{m.label}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// ============ Movement Category Grid ============
function MovementCategoryGrid({ 
  categories,
  selectedClass,
  onSelect 
}: { 
  categories: Record<string, string[]>
  selectedClass?: string
  onSelect?: (cls: string) => void
}) {
  const categoryColors: Record<string, string> = {
    'Static Poses': 'bg-slate-500/20 text-slate-300 border-slate-500/30',
    'Locomotion': 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    'Upper Body': 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    'Lower Body': 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    'Sports': 'bg-orange-500/20 text-orange-300 border-orange-500/30',
    'Exercise': 'bg-pink-500/20 text-pink-300 border-pink-500/30'
  }
  
  return (
    <div className="space-y-6">
      {Object.entries(categories).map(([category, classes]) => (
        <div key={category}>
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">{category}</h4>
          <div className="flex flex-wrap gap-2">
            {classes.map((cls) => (
              <Badge
                key={cls}
                variant="outline"
                className={`${categoryColors[category]} cursor-pointer hover:scale-105 transition-transform ${
                  selectedClass === cls ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => onSelect?.(cls)}
              >
                {cls.replace('_', ' ')}
              </Badge>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ============ Latency Graph ============
function LatencyGraph({ history }: { history: number[] }) {
  const maxVal = Math.max(...history, 1)
  const points = history.map((v, i) => `${(i / (history.length - 1)) * 100},${100 - (v / maxVal) * 100}`).join(' ')
  
  return (
    <div className="relative h-24 w-full">
      <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        {/* Background lines */}
        {[0, 25, 50, 75, 100].map((y) => (
          <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
        ))}
        {/* Data line */}
        <polyline
          points={points}
          fill="none"
          stroke="url(#latencyGradient)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Gradient definition */}
        <defs>
          <linearGradient id="latencyGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#4ECDC4" />
            <stop offset="100%" stopColor="#45B7D1" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute bottom-0 right-0 text-xs text-muted-foreground">
        {history.length > 0 && `${history[history.length - 1].toFixed(0)}ms`}
      </div>
    </div>
  )
}

// ============ Main Dashboard ============
export default function MotionIntelligenceDashboard() {
  // State
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [frameCount, setFrameCount] = useState(0)
  const [predictionFrame, setPredictionFrame] = useState(0)
  const [latencyHistory, setLatencyHistory] = useState<number[]>([])
  const [connected, setConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('realtime')
  const [selectedMovement, setSelectedMovement] = useState<string>('walking')
  const [metrics, setMetrics] = useState<ModelMetrics>({
    posenet: { accuracy: 0, latency: 0 },
    classifier: { accuracy: 0, latency: 0 },
    predictor: { mpjpe: 0, latency: 0 }
  })
  
  const inferenceCount = useRef(0)
  const API_BASE = '' // Use local Next.js API routes
  
  // Check API health
  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await fetch('/api/health')
        if (response.ok) {
          setConnected(true)
          setConnectionError(null)
        } else {
          throw new Error('API error')
        }
      } catch {
        setConnected(false)
        setConnectionError('API not running')
      }
    }
    
    checkAPI()
    const interval = setInterval(checkAPI, 5000)
    return () => clearInterval(interval)
  }, [])
  
  // Run inference
  const runInference = useCallback(async () => {
    if (!connected) return
    
    try {
      const response = await fetch(`/api/demo/${selectedMovement}`)
      if (!response.ok) throw new Error('Inference failed')
      
      const data: InferenceResponse = await response.json()
      setResult(data)
      setFrameCount(prev => prev + 1)
      setPredictionFrame(0)
      
      // Update latency history
      setLatencyHistory(prev => [...prev, data.inference_time_ms].slice(-50))
      
      // Update metrics
      inferenceCount.current += 1
      setMetrics(prev => ({
        posenet: { 
          accuracy: prev.posenet.accuracy + 0.85, 
          latency: (prev.posenet.latency + data.inference_time_ms * 0.3) / (inferenceCount.current) 
        },
        classifier: { 
          accuracy: prev.classifier.accuracy + 0.92, 
          latency: (prev.classifier.latency + data.inference_time_ms * 0.4) / (inferenceCount.current)
        },
        predictor: { 
          mpjpe: prev.predictor.mpjpe + 25, 
          latency: (prev.predictor.latency + data.inference_time_ms * 0.3) / (inferenceCount.current)
        }
      }))
    } catch (err) {
      console.error('Inference error:', err)
    }
  }, [connected, selectedMovement])
  
  // Auto-run inference
  useEffect(() => {
    if (isRunning && connected) {
      const interval = setInterval(runInference, 800)
      return () => clearInterval(interval)
    }
  }, [isRunning, connected, runInference])
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl shadow-lg shadow-emerald-500/20">
                <Motion className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                  MotionAI Pro
                </h1>
                <p className="text-xs text-muted-foreground">Human Motion Intelligence System</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Badge variant="outline" className={`${connected ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-red-500/10 text-red-400 border-red-500/30'}`}>
                {connected ? <CheckCircle className="w-3 h-3 mr-1" /> : <AlertCircle className="w-3 h-3 mr-1" />}
                {connected ? 'Connected' : 'Disconnected'}
              </Badge>
              <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30">
                <Layers className="w-3 h-3 mr-1" />
                35 Classes
              </Badge>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 py-6 w-full">
        {/* Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <Timer className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-muted-foreground">Avg Latency</span>
              </div>
              <p className="text-xl font-bold mt-1">
                {latencyHistory.length > 0 ? (latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length).toFixed(1) : 0}ms
              </p>
            </CardContent>
          </Card>
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-xs text-muted-foreground">Inferences</span>
              </div>
              <p className="text-xl font-bold mt-1">{frameCount}</p>
            </CardContent>
          </Card>
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-purple-400" />
                <span className="text-xs text-muted-foreground">Keypoints</span>
              </div>
              <p className="text-xl font-bold mt-1">17</p>
            </CardContent>
          </Card>
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-orange-400" />
                <span className="text-xs text-muted-foreground">Models</span>
              </div>
              <p className="text-xl font-bold mt-1">3</p>
            </CardContent>
          </Card>
        </div>
        
        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-slate-900/50 border border-slate-700 flex-wrap h-auto">
            <TabsTrigger value="realtime" className="data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-400">
              <Zap className="w-4 h-4 mr-2" />
              Real-time
            </TabsTrigger>
            <TabsTrigger value="prediction" className="data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-400">
              <TrendingUp className="w-4 h-4 mr-2" />
              Prediction
            </TabsTrigger>
            <TabsTrigger value="video" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400">
              <Video className="w-4 h-4 mr-2" />
              Video
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-orange-500/20 data-[state=active]:text-orange-400">
              <LineChart className="w-4 h-4 mr-2" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="classes" className="data-[state=active]:bg-pink-500/20 data-[state=active]:text-pink-400">
              <BarChart3 className="w-4 h-4 mr-2" />
              Classes
            </TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400">
              <Cpu className="w-4 h-4 mr-2" />
              Models
            </TabsTrigger>
          </TabsList>
          
          {/* Real-time Tab */}
          <TabsContent value="realtime" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Pose Estimation */}
              <Card className="bg-slate-900/50 border-slate-700">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <Eye className="w-5 h-5 text-blue-400" />
                      Pose Estimation
                    </CardTitle>
                    <Badge variant="secondary">PoseNet</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-center mb-4">
                    {result?.pose ? (
                      <SkeletonVisualization keypoints={result.pose.keypoints} width={260} height={260} />
                    ) : (
                      <div className="w-[260px] h-[260px] bg-slate-800/50 rounded-lg flex items-center justify-center text-muted-foreground">
                        Start inference
                      </div>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                      <p className="text-muted-foreground text-xs">Input</p>
                      <p className="font-mono">256×256</p>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 text-center">
                      <p className="text-muted-foreground text-xs">Output</p>
                      <p className="font-mono">17 KP</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              {/* Classification */}
              <Card className="bg-slate-900/50 border-slate-700">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <Brain className="w-5 h-5 text-purple-400" />
                      Classification
                    </CardTitle>
                    <Badge variant="secondary">BiLSTM</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  {result?.classification ? (
                    <div className="space-y-4">
                      <div className="text-center">
                        <Badge className="text-lg px-4 py-2 capitalize bg-gradient-to-r from-purple-500 to-pink-500">
                          {result.classification.class_name.replace('_', ' ')}
                        </Badge>
                        <p className="text-3xl font-bold mt-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                          {(result.classification.confidence * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-muted-foreground">Confidence</p>
                      </div>
                      <Separator className="bg-slate-700" />
                      <ScrollArea className="h-[180px]">
                        <ProbabilityChart 
                          probabilities={result.classification.all_probabilities}
                          highlightedClass={result.classification.class_name}
                        />
                      </ScrollArea>
                    </div>
                  ) : (
                    <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                      Start inference
                    </div>
                  )}
                </CardContent>
              </Card>
              
              {/* Latency & Controls */}
              <Card className="bg-slate-900/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Clock className="w-5 h-5 text-emerald-400" />
                    Performance
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Latency History</p>
                    <LatencyGraph history={latencyHistory} />
                  </div>
                  
                  <Separator className="bg-slate-700" />
                  
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Select Movement</p>
                    <select 
                      value={selectedMovement}
                      onChange={(e) => setSelectedMovement(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-sm"
                    >
                      {MOVEMENT_CLASSES.map((cls) => (
                        <option key={cls} value={cls}>{cls.replace('_', ' ')}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      className={`flex-1 ${isRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-emerald-500 hover:bg-emerald-600'}`}
                      onClick={() => setIsRunning(!isRunning)}
                      disabled={!connected}
                    >
                      {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                      {isRunning ? 'Stop' : 'Start'}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={runInference}
                      disabled={!connected || isRunning}
                    >
                      <Zap className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => { setFrameCount(0); setResult(null); setLatencyHistory([]); }}
                    >
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          {/* Prediction Tab */}
          <TabsContent value="prediction" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Prediction Animation */}
              <Card className="bg-slate-900/50 border-slate-700">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <TrendingUp className="w-5 h-5 text-emerald-400" />
                      Motion Prediction
                    </CardTitle>
                    <Badge variant="secondary">MotionFormer</Badge>
                  </div>
                  <CardDescription>10-frame future motion prediction</CardDescription>
                </CardHeader>
                <CardContent>
                  {result?.prediction ? (
                    <PredictionPlayer 
                      predictions={result.prediction.predictions}
                      currentFrame={predictionFrame}
                      onFrameChange={setPredictionFrame}
                    />
                  ) : (
                    <div className="h-[350px] flex items-center justify-center text-muted-foreground">
                      Run inference to see predictions
                    </div>
                  )}
                </CardContent>
              </Card>
              
              {/* Prediction Details */}
              <Card className="bg-slate-900/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Info className="w-5 h-5 text-blue-400" />
                    Prediction Details
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-slate-800/50 rounded-lg p-4 text-center">
                        <p className="text-3xl font-bold text-blue-400">20</p>
                        <p className="text-sm text-muted-foreground">Input Frames</p>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-4 text-center">
                        <p className="text-3xl font-bold text-emerald-400">10</p>
                        <p className="text-sm text-muted-foreground">Predicted Frames</p>
                      </div>
                    </div>
                    
                    <Separator className="bg-slate-700" />
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model Architecture</span>
                        <span className="font-mono">Transformer</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Encoder Layers</span>
                        <span className="font-mono">4</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Decoder Layers</span>
                        <span className="font-mono">4</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Attention Heads</span>
                        <span className="font-mono">8</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model Dimension</span>
                        <span className="font-mono">256</span>
                      </div>
                    </div>
                    
                    <Separator className="bg-slate-700" />
                    
                    <div className="bg-slate-800/50 rounded-lg p-4">
                      <p className="text-sm font-medium mb-2">Loss Function</p>
                      <div className="text-xs text-muted-foreground space-y-1">
                        <p>• MPJPE (Mean Per Joint Position Error)</p>
                        <p>• Velocity Loss for smooth motion</p>
                        <p>• Bone Length Preservation</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          {/* Video Analysis Tab */}
          <TabsContent value="video" className="space-y-6">
            <VideoAnalysis />
          </TabsContent>
          
          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <TrainingVisualization />
          </TabsContent>
          
          {/* Export Section */}
          <div className="flex justify-center gap-4 py-4">
            <Button variant="outline" onClick={async () => {
              const response = await fetch('/api/export?format=json&count=100&download=true')
              if (response.ok) {
                const blob = await response.blob()
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'motionai_predictions.json'
                a.click()
                URL.revokeObjectURL(url)
              }
            }}>
              <Download className="w-4 h-4 mr-2" />
              Export JSON
            </Button>
            <Button variant="outline" onClick={async () => {
              const response = await fetch('/api/export?format=csv&count=100&download=true')
              if (response.ok) {
                const blob = await response.blob()
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'motionai_predictions.csv'
                a.click()
                URL.revokeObjectURL(url)
              }
            }}>
              <Database className="w-4 h-4 mr-2" />
              Export CSV
            </Button>
          </div>
          
          {/* Classes Tab */}
          <TabsContent value="classes" className="space-y-6">
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <BarChart3 className="w-5 h-5 text-purple-400" />
                  35 Movement Classes
                </CardTitle>
                <CardDescription>Organized by movement category</CardDescription>
              </CardHeader>
              <CardContent>
                <MovementCategoryGrid 
                  categories={MOVEMENT_CATEGORIES}
                  selectedClass={selectedMovement}
                  onSelect={(cls) => {
                    setSelectedMovement(cls)
                    setActiveTab('realtime')
                  }}
                />
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Models Tab */}
          <TabsContent value="models" className="space-y-6">
            <div className="grid md:grid-cols-3 gap-6">
              <PerformanceCard
                title="PoseNet"
                icon={Eye}
                metrics={[
                  { label: 'Accuracy', value: '85%+', unit: 'PCKh' },
                  { label: 'Latency', value: '~15', unit: 'ms' },
                ]}
              />
              <PerformanceCard
                title="MoveClassifier"
                icon={Brain}
                metrics={[
                  { label: 'Accuracy', value: '92%+', unit: '' },
                  { label: 'Latency', value: '~8', unit: 'ms' },
                ]}
              />
              <PerformanceCard
                title="MotionFormer"
                icon={TrendingUp}
                metrics={[
                  { label: 'MPJPE', value: '<30', unit: 'mm' },
                  { label: 'Latency', value: '~12', unit: 'ms' },
                ]}
              />
            </div>
            
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Layers className="w-5 h-5 text-orange-400" />
                  Model Architecture Details
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="space-y-3">
                    <h4 className="font-semibold text-blue-400">PoseNet (Stacked Hourglass)</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• 2 stacks with intermediate supervision</li>
                      <li>• 256 feature channels</li>
                      <li>• OHKM loss for hard keypoints</li>
                      <li>• ~25M parameters</li>
                    </ul>
                  </div>
                  <div className="space-y-3">
                    <h4 className="font-semibold text-purple-400">MoveClassifier (BiLSTM)</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Bidirectional LSTM (2 layers)</li>
                      <li>• Self-attention (4 heads)</li>
                      <li>• Temporal attention pooling</li>
                      <li>• ~2M parameters</li>
                    </ul>
                  </div>
                  <div className="space-y-3">
                    <h4 className="font-semibold text-emerald-400">MotionFormer (Transformer)</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Encoder-decoder architecture</li>
                      <li>• 8 attention heads</li>
                      <li>• Learnable positional encoding</li>
                      <li>• ~15M parameters</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-muted-foreground text-sm">
          <p>MotionAI Pro - Human Motion Intelligence System</p>
          <p className="text-xs mt-1">
            Train: <code className="bg-slate-800 px-2 py-0.5 rounded">python train_all_real.py --epochs 50</code>
            {' | '}
            API: <code className="bg-slate-800 px-2 py-0.5 rounded">python -m api.main</code>
          </p>
        </div>
      </footer>
    </div>
  )
}
