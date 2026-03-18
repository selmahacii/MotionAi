'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { 
  Video, Upload, Play, Pause, Download, Clock, Target, Activity, TrendingUp,
  AlertCircle, CheckCircle, FileVideo
} from 'lucide-react'

const MOVEMENT_CLASSES = [
  'standing', 'sitting', 'lying_down', 'kneeling', 'crouching',
  'walking', 'running', 'jumping', 'hopping', 'crawling', 'climbing',
  'arms_raised', 'waving', 'clapping', 'punching', 'pushing', 'pulling', 'throwing', 'catching',
  'kicking', 'squatting', 'lunging', 'stretching',
  'golf_swing', 'baseball_swing', 'tennis_serve', 'tennis_forehand', 'basketball_shot', 'soccer_kick', 'swimming', 'bowling',
  'push_up', 'sit_up', 'burpee', 'yoga_pose'
]

interface FrameResult {
  frameNumber: number
  timestamp: number
  keypoints: { x: number; y: number; score: number }[]
  movement: string
  confidence: number
}

interface VideoResult {
  videoId: string
  totalFrames: number
  fps: number
  duration: number
  frames: FrameResult[]
  summary: {
    dominantMovement: string
    movementDistribution: Record<string, number>
    averageConfidence: number
    keyMoments: { frame: number; movement: string; confidence: number }[]
  }
}

// Skeleton visualization component
function MiniSkeleton({ keypoints, size = 80 }: { keypoints: { x: number; y: number; score: number }[]; size?: number }) {
  const connections = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
  ]
  
  return (
    <svg width={size} height={size} viewBox="0 0 1 1" className="bg-slate-800 rounded">
      {connections.map(([i, j], idx) => {
        const kp1 = keypoints[i]
        const kp2 = keypoints[j]
        if (!kp1 || !kp2) return null
        return (
          <line
            key={idx}
            x1={kp1.x}
            y1={kp1.y}
            x2={kp2.x}
            y2={kp2.y}
            stroke="#4ECDC4"
            strokeWidth={0.015}
            strokeLinecap="round"
          />
        )
      })}
      {keypoints.map((kp, idx) => (
        <circle key={idx} cx={kp.x} cy={kp.y} r={0.012} fill="white" />
      ))}
    </svg>
  )
}

// Timeline component
function FrameTimeline({ 
  frames, 
  currentIndex, 
  onSelect 
}: { 
  frames: FrameResult[]
  currentIndex: number
  onSelect: (index: number) => void 
}) {
  return (
    <ScrollArea className="w-full">
      <div className="flex gap-1 p-2">
        {frames.map((frame, idx) => (
          <button
            key={idx}
            onClick={() => onSelect(idx)}
            className={`flex-shrink-0 w-12 h-12 rounded border-2 transition-all ${
              idx === currentIndex 
                ? 'border-primary bg-primary/20' 
                : 'border-slate-700 hover:border-slate-500'
            }`}
          >
            <span className="text-xs text-muted-foreground">{frame.frameNumber}</span>
          </button>
        ))}
      </div>
    </ScrollArea>
  )
}

// Main Video Analysis Component
export default function VideoAnalysis() {
  const [selectedMovements, setSelectedMovements] = useState<string[]>(['walking', 'running', 'jumping'])
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<VideoResult | null>(null)
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  
  // Toggle movement selection
  const toggleMovement = useCallback((movement: string) => {
    setSelectedMovements(prev => 
      prev.includes(movement) 
        ? prev.filter(m => m !== movement)
        : [...prev, movement]
    )
  }, [])
  
  // Process video
  const processVideo = useCallback(async () => {
    if (selectedMovements.length === 0) return
    
    setIsProcessing(true)
    setProgress(0)
    setResult(null)
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90))
      }, 200)
      
      const response = await fetch('/api/video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          movements: selectedMovements,
          fps: 30
        })
      })
      
      clearInterval(progressInterval)
      setProgress(100)
      
      if (response.ok) {
        const data = await response.json()
        setResult(data)
        setCurrentFrameIndex(0)
      }
    } catch (error) {
      console.error('Video processing error:', error)
    } finally {
      setIsProcessing(false)
    }
  }, [selectedMovements])
  
  // Export results
  const exportResults = useCallback(async () => {
    if (!result) return
    
    const exportData = result.frames.map(f => ({
      frame: f.frameNumber,
      timestamp: f.timestamp,
      movement: f.movement,
      confidence: f.confidence
    }))
    
    const response = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        format: 'json',
        data: exportData,
        filename: 'video_analysis'
      })
    })
    
    if (response.ok) {
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'video_analysis.json'
      a.click()
      URL.revokeObjectURL(url)
    }
  }, [result])
  
  // Auto-play frames
  useState(() => {
    if (isPlaying && result) {
      const interval = setInterval(() => {
        setCurrentFrameIndex(prev => (prev + 1) % result.frames.length)
      }, 100)
      return () => clearInterval(interval)
    }
  })
  
  const currentFrame = result?.frames[currentFrameIndex]
  
  return (
    <div className="space-y-6">
      {/* Movement Selection */}
      <Card className="bg-slate-900/50 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="w-5 h-5 text-primary" />
            Video Analysis
          </CardTitle>
          <CardDescription>Select movements to simulate in the video</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {MOVEMENT_CLASSES.slice(0, 18).map(movement => (
              <Badge
                key={movement}
                variant={selectedMovements.includes(movement) ? 'default' : 'outline'}
                className="cursor-pointer capitalize"
                onClick={() => toggleMovement(movement)}
              >
                {movement.replace('_', ' ')}
              </Badge>
            ))}
          </div>
          
          <div className="flex gap-2">
            <Button 
              onClick={processVideo} 
              disabled={isProcessing || selectedMovements.length === 0}
              className="flex-1"
            >
              {isProcessing ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-pulse" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Analyze Video
                </>
              )}
            </Button>
          </div>
          
          {isProcessing && (
            <Progress value={progress} className="h-2" />
          )}
        </CardContent>
      </Card>
      
      {/* Results */}
      {result && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="bg-slate-900/50 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-400" />
                  <span className="text-xs text-muted-foreground">Duration</span>
                </div>
                <p className="text-2xl font-bold mt-1">{result.duration.toFixed(1)}s</p>
              </CardContent>
            </Card>
            <Card className="bg-slate-900/50 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs text-muted-foreground">Frames</span>
                </div>
                <p className="text-2xl font-bold mt-1">{result.totalFrames}</p>
              </CardContent>
            </Card>
            <Card className="bg-slate-900/50 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                  <span className="text-xs text-muted-foreground">Dominant</span>
                </div>
                <p className="text-xl font-bold mt-1 capitalize">{result.summary.dominantMovement.replace('_', ' ')}</p>
              </CardContent>
            </Card>
            <Card className="bg-slate-900/50 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-orange-400" />
                  <span className="text-xs text-muted-foreground">Avg Confidence</span>
                </div>
                <p className="text-2xl font-bold mt-1">{(result.summary.averageConfidence * 100).toFixed(1)}%</p>
              </CardContent>
            </Card>
          </div>
          
          {/* Frame Player */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Frame Analysis</CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={() => setIsPlaying(!isPlaying)}>
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                  <Button size="sm" variant="outline" onClick={exportResults}>
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {currentFrame && (
                <div className="flex gap-6">
                  <div className="flex-shrink-0">
                    <MiniSkeleton keypoints={currentFrame.keypoints} size={200} />
                  </div>
                  <div className="flex-1 space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Frame</p>
                        <p className="text-xl font-bold">{currentFrame.frameNumber}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Timestamp</p>
                        <p className="text-xl font-bold">{currentFrame.timestamp.toFixed(3)}s</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Movement</p>
                        <Badge className="capitalize">{currentFrame.movement.replace('_', ' ')}</Badge>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Confidence</p>
                        <p className="text-xl font-bold text-emerald-400">{(currentFrame.confidence * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                    <Progress value={currentFrame.confidence * 100} className="h-2" />
                  </div>
                </div>
              )}
              
              <Separator className="bg-slate-700" />
              
              {/* Timeline */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>Timeline</span>
                  <span>{currentFrameIndex + 1} / {result.frames.length}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={result.frames.length - 1}
                  value={currentFrameIndex}
                  onChange={(e) => setCurrentFrameIndex(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </CardContent>
          </Card>
          
          {/* Key Moments */}
          <Card className="bg-slate-900/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-lg">Key Moments</CardTitle>
              <CardDescription>High-confidence movement detections</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {result.summary.keyMoments.map((moment, idx) => (
                  <Badge
                    key={idx}
                    variant="outline"
                    className="cursor-pointer"
                    onClick={() => setCurrentFrameIndex(moment.frame)}
                  >
                    Frame {moment.frame}: {moment.movement.replace('_', ' ')} ({(moment.confidence * 100).toFixed(0)}%)
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
