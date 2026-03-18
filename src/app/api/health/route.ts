import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    service: 'MotionAI Pro Dashboard',
    version: '1.0.0',
    models: ['PoseNet', 'MoveClassifier', 'MotionFormer'],
    classes: 35,
    keypoints: 17,
    timestamp: new Date().toISOString()
  })
}
