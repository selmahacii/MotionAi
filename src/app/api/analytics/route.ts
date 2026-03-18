import { NextRequest, NextResponse } from 'next/server'

// Analytics data for training and inference statistics

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
  trainingHistory: {
    epoch: number
    trainLoss: number
    valLoss: number
    accuracy: number
  }[]
}

interface SystemAnalytics {
  timestamp: string
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
  recentActivity: {
    timestamp: string
    type: 'inference' | 'training' | 'export'
    details: string
  }[]
}

// Generate simulated training history
function generateTrainingHistory(epochs: number = 50): { epoch: number; trainLoss: number; valLoss: number; accuracy: number }[] {
  const history = []
  for (let i = 1; i <= epochs; i++) {
    const progress = i / epochs
    history.push({
      epoch: i,
      trainLoss: 2.5 * Math.exp(-3 * progress) + 0.1 + Math.random() * 0.05,
      valLoss: 2.8 * Math.exp(-2.5 * progress) + 0.15 + Math.random() * 0.08,
      accuracy: 0.5 + 0.45 * (1 - Math.exp(-4 * progress)) + Math.random() * 0.02
    })
  }
  return history
}

// Generate analytics data
function generateAnalytics(): SystemAnalytics {
  const models: ModelPerformance[] = [
    {
      name: 'PoseNet',
      metrics: {
        accuracy: 0.873,
        precision: 0.865,
        recall: 0.881,
        f1Score: 0.873,
        latency: 15.3,
        throughput: 65.4
      },
      trainingHistory: generateTrainingHistory(100)
    },
    {
      name: 'MoveClassifier',
      metrics: {
        accuracy: 0.924,
        precision: 0.918,
        recall: 0.931,
        f1Score: 0.924,
        latency: 8.2,
        throughput: 122.0
      },
      trainingHistory: generateTrainingHistory(80)
    },
    {
      name: 'MotionFormer',
      metrics: {
        accuracy: 0.891,
        precision: 0.885,
        recall: 0.897,
        f1Score: 0.891,
        latency: 12.7,
        throughput: 78.7
      },
      trainingHistory: generateTrainingHistory(120)
    }
  ]
  
  const classDistribution: Record<string, number> = {
    walking: 0.18,
    running: 0.12,
    jumping: 0.09,
    standing: 0.08,
    sitting: 0.07,
    squatting: 0.06,
    punching: 0.05,
    kicking: 0.05,
    throwing: 0.04,
    climbing: 0.04,
    other: 0.22
  }
  
  const recentActivity = [
    { timestamp: new Date(Date.now() - 60000).toISOString(), type: 'inference' as const, details: 'Processed 30 frames' },
    { timestamp: new Date(Date.now() - 120000).toISOString(), type: 'inference' as const, details: 'Video analysis completed' },
    { timestamp: new Date(Date.now() - 300000).toISOString(), type: 'training' as const, details: 'PoseNet epoch 87/100' },
    { timestamp: new Date(Date.now() - 600000).toISOString(), type: 'export' as const, details: 'Exported 100 predictions to CSV' },
    { timestamp: new Date(Date.now() - 900000).toISOString(), type: 'inference' as const, details: 'Batch inference: 50 samples' }
  ]
  
  return {
    timestamp: new Date().toISOString(),
    models,
    inferenceStats: {
      totalInferences: 15420,
      averageLatency: 28.5,
      p50Latency: 24.2,
      p95Latency: 52.8,
      p99Latency: 78.3,
      errors: 12
    },
    classDistribution,
    recentActivity
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const type = searchParams.get('type') || 'all'
  
  const analytics = generateAnalytics()
  
  switch (type) {
    case 'models':
      return NextResponse.json({ models: analytics.models })
    case 'inference':
      return NextResponse.json({ inferenceStats: analytics.inferenceStats })
    case 'classes':
      return NextResponse.json({ classDistribution: analytics.classDistribution })
    case 'activity':
      return NextResponse.json({ recentActivity: analytics.recentActivity })
    default:
      return NextResponse.json(analytics)
  }
}
