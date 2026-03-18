import { NextRequest, NextResponse } from 'next/server'

// Movement classes matching the config
const MOVEMENT_CLASSES = [
  'standing', 'sitting', 'lying_down', 'kneeling', 'crouching',
  'walking', 'running', 'jumping', 'hopping', 'crawling', 'climbing',
  'arms_raised', 'waving', 'clapping', 'punching', 'pushing', 'pulling', 'throwing', 'catching',
  'kicking', 'squatting', 'lunging', 'stretching',
  'golf_swing', 'baseball_swing', 'tennis_serve', 'tennis_forehand', 'basketball_shot', 'soccer_kick', 'swimming', 'bowling',
  'push_up', 'sit_up', 'burpee', 'yoga_pose'
]

const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

// Generate realistic pose based on movement type
function generatePose(movement: string, frame: number): { x: number; y: number; score: number }[] {
  const t = frame * 0.1
  const keypoints: { x: number; y: number; score: number }[] = []
  
  // Base pose (standing)
  const basePose = [
    { x: 0.5, y: 0.15 },   // nose
    { x: 0.48, y: 0.12 },  // left_eye
    { x: 0.52, y: 0.12 },  // right_eye
    { x: 0.45, y: 0.15 },  // left_ear
    { x: 0.55, y: 0.15 },  // right_ear
    { x: 0.4, y: 0.28 },   // left_shoulder
    { x: 0.6, y: 0.28 },   // right_shoulder
    { x: 0.35, y: 0.42 },  // left_elbow
    { x: 0.65, y: 0.42 },  // right_elbow
    { x: 0.32, y: 0.55 },  // left_wrist
    { x: 0.68, y: 0.55 },  // right_wrist
    { x: 0.45, y: 0.55 },  // left_hip
    { x: 0.55, y: 0.55 },  // right_hip
    { x: 0.43, y: 0.75 },  // left_knee
    { x: 0.57, y: 0.75 },  // right_knee
    { x: 0.42, y: 0.95 },  // left_ankle
    { x: 0.58, y: 0.95 },  // right_ankle
  ]
  
  // Apply movement-specific modifications
  switch (movement) {
    case 'walking':
      basePose[9].x += Math.sin(t * 4) * 0.08  // left_wrist swing
      basePose[10].x -= Math.sin(t * 4) * 0.08 // right_wrist swing
      basePose[15].x += Math.sin(t * 4) * 0.05 // left_ankle step
      basePose[16].x -= Math.sin(t * 4) * 0.05 // right_ankle step
      break
      
    case 'running':
      basePose[9].x += Math.sin(t * 8) * 0.12
      basePose[10].x -= Math.sin(t * 8) * 0.12
      basePose[15].x += Math.sin(t * 8) * 0.08
      basePose[16].x -= Math.sin(t * 8) * 0.08
      basePose.forEach((_, i) => { basePose[i].y -= 0.05 + Math.abs(Math.sin(t * 8)) * 0.05 })
      break
      
    case 'jumping':
      const jumpHeight = Math.max(0, Math.sin(t * 3)) * 0.15
      basePose.forEach((_, i) => { basePose[i].y -= jumpHeight })
      basePose[9].y -= 0.1
      basePose[10].y -= 0.1
      break
      
    case 'squatting':
      basePose[7].y += 0.15  // left_elbow
      basePose[8].y += 0.15  // right_elbow
      basePose[9].y += 0.1   // left_wrist
      basePose[10].y += 0.1  // right_wrist
      basePose[13].y += 0.1  // left_knee
      basePose[14].y += 0.1  // right_knee
      basePose[15].y -= 0.1  // left_ankle
      basePose[16].y -= 0.1  // right_ankle
      break
      
    case 'sitting':
      basePose[7].y += 0.2
      basePose[8].y += 0.2
      basePose[9].y += 0.15
      basePose[10].y += 0.15
      basePose[13].y += 0.2
      basePose[14].y += 0.2
      basePose[15].y += 0.15
      basePose[16].y += 0.15
      break
      
    case 'lying_down':
      basePose.forEach((kp, i) => {
        const temp = kp.x
        basePose[i].x = 0.2 + kp.y * 0.6
        basePose[i].y = 0.5
      })
      break
      
    case 'arms_raised':
      basePose[7].y = 0.2   // left_elbow up
      basePose[8].y = 0.2   // right_elbow up
      basePose[9].y = 0.1   // left_wrist up
      basePose[10].y = 0.1  // right_wrist up
      break
      
    case 'punching':
      basePose[9].x = 0.7 + Math.sin(t * 2) * 0.1
      basePose[9].y = 0.4
      break
      
    case 'kicking':
      basePose[15].x = 0.7 + Math.sin(t * 2) * 0.1
      basePose[15].y = 0.6
      basePose[13].y = 0.65
      break
      
    case 'waving':
      basePose[9].x = 0.65 + Math.sin(t * 6) * 0.08
      basePose[9].y = 0.2 + Math.abs(Math.sin(t * 6)) * 0.1
      basePose[7].y = 0.25
      break
      
    case 'clapping':
      const clap = Math.sin(t * 8) > 0
      basePose[9].x = clap ? 0.5 : 0.35
      basePose[10].x = clap ? 0.5 : 0.65
      basePose[9].y = 0.35
      basePose[10].y = 0.35
      break
      
    case 'golf_swing':
      basePose[7].y = 0.2 + Math.sin(t) * 0.1
      basePose[8].y = 0.35
      basePose[9].x = 0.3 + Math.sin(t * 2) * 0.15
      basePose[9].y = 0.3
      basePose[10].x = 0.4 + Math.sin(t * 2) * 0.1
      break
      
    case 'baseball_swing':
      basePose[7].y = 0.25
      basePose[8].y = 0.3
      basePose[9].x = 0.25 + Math.sin(t * 2) * 0.2
      basePose[10].x = 0.35 + Math.sin(t * 2) * 0.15
      break
      
    case 'basketball_shot':
      const shootPhase = (t % 2) / 2
      if (shootPhase > 0.5) {
        basePose[9].y = 0.15
        basePose[10].y = 0.15
        basePose[7].y = 0.22
        basePose[8].y = 0.22
      }
      break
      
    case 'push_up':
      const pushPhase = Math.sin(t * 2)
      basePose[0].y = 0.75 + pushPhase * 0.05
      basePose[9].y = 0.9
      basePose[10].y = 0.9
      basePose.forEach((_, i) => { basePose[i].x *= 0.7; basePose[i].x += 0.15 })
      break
      
    case 'yoga_pose':
      basePose[9].x = 0.55
      basePose[9].y = 0.45
      basePose[10].x = 0.45
      basePose[10].y = 0.45
      basePose[15].x = 0.5
      basePose[15].y = 0.5
      break
      
    case 'climbing':
      basePose[9].y = 0.25 + Math.sin(t * 3) * 0.1
      basePose[10].y = 0.3 + Math.sin(t * 3 + 1) * 0.1
      basePose[15].y = 0.7 + Math.sin(t * 3) * 0.1
      basePose[16].y = 0.75 + Math.sin(t * 3 + 1) * 0.1
      break
      
    default:
      // Add subtle idle motion
      basePose.forEach((_, i) => {
        basePose[i].x += Math.sin(t + i) * 0.005
        basePose[i].y += Math.cos(t + i) * 0.005
      })
  }
  
  // Convert to output format with confidence scores
  for (let i = 0; i < 17; i++) {
    keypoints.push({
      x: Math.max(0.05, Math.min(0.95, basePose[i].x + (Math.random() - 0.5) * 0.02)),
      y: Math.max(0.05, Math.min(0.95, basePose[i].y + (Math.random() - 0.5) * 0.02)),
      score: 0.85 + Math.random() * 0.14
    })
  }
  
  return keypoints
}

// Generate probability distribution
function generateProbabilities(targetClass: number, numClasses: number): number[] {
  const probs: number[] = []
  const targetConfidence = 0.7 + Math.random() * 0.25
  
  for (let i = 0; i < numClasses; i++) {
    if (i === targetClass) {
      probs.push(targetConfidence)
    } else {
      probs.push((1 - targetConfidence) * Math.random() * 0.3)
    }
  }
  
  // Normalize
  const sum = probs.reduce((a, b) => a + b, 0)
  return probs.map(p => p / sum)
}

// Generate predicted frames
function generatePredictions(baseKeypoints: { x: number; y: number; score: number }[], numFrames: number): { x: number; y: number; score: number }[][] {
  const predictions: { x: number; y: number; score: number }[][] = []
  
  for (let f = 0; f < numFrames; f++) {
    const frameKeypoints = baseKeypoints.map((kp, i) => ({
      x: Math.max(0.05, Math.min(0.95, kp.x + (Math.random() - 0.5) * 0.03 * (f + 1))),
      y: Math.max(0.05, Math.min(0.95, kp.y + (Math.random() - 0.5) * 0.02 * (f + 1))),
      score: Math.max(0.5, kp.score - f * 0.03)
    }))
    predictions.push(frameKeypoints)
  }
  
  return predictions
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ movement: string }> }
) {
  const { movement: movementParam } = await params
  const movement = movementParam || 'walking'
  
  // Validate movement class
  const classIndex = MOVEMENT_CLASSES.indexOf(movement)
  if (classIndex === -1) {
    return NextResponse.json(
      { error: `Unknown movement: ${movement}. Available: ${MOVEMENT_CLASSES.join(', ')}` },
      { status: 400 }
    )
  }
  
  // Simulate inference delay
  const startTime = Date.now()
  
  // Generate pose
  const keypoints = generatePose(movement, Date.now() / 1000)
  
  // Generate classification
  const probabilities = generateProbabilities(classIndex, MOVEMENT_CLASSES.length)
  const predictedClass = probabilities.indexOf(Math.max(...probabilities))
  
  // Generate predictions
  const predictions = generatePredictions(keypoints, 10)
  
  const inferenceTime = Date.now() - startTime + 15 + Math.random() * 25
  
  const response = {
    pose: {
      keypoints,
      keypoint_names: KEYPOINT_NAMES
    },
    classification: {
      predicted_class: predictedClass,
      class_name: MOVEMENT_CLASSES[predictedClass],
      confidence: probabilities[predictedClass],
      all_probabilities: probabilities
    },
    prediction: {
      predicted_frames: 10,
      keypoints_per_frame: 17,
      predictions
    },
    inference_time_ms: inferenceTime
  }
  
  return NextResponse.json(response)
}
