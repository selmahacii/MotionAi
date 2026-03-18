import { NextRequest, NextResponse } from 'next/server'

// Export predictions and analysis results in various formats

interface ExportData {
  format: 'json' | 'csv' | 'markdown'
  data: unknown
  filename?: string
}

// Convert predictions to CSV format
function toCSV(data: Record<string, unknown>[]): string {
  if (data.length === 0) return ''
  
  const headers = Object.keys(data[0])
  const rows = data.map(row => 
    headers.map(h => {
      const value = row[h]
      if (typeof value === 'object') return JSON.stringify(value)
      return String(value)
    }).join(',')
  )
  
  return [headers.join(','), ...rows].join('\n')
}

// Convert predictions to Markdown format
function toMarkdown(data: Record<string, unknown>[], title: string): string {
  if (data.length === 0) return '# No data'
  
  const headers = Object.keys(data[0])
  const headerRow = `| ${headers.join(' | ')} |`
  const separatorRow = `| ${headers.map(() => '---').join(' | ')} |`
  const dataRows = data.map(row => 
    `| ${headers.map(h => {
      const value = row[h]
      if (typeof value === 'object') return JSON.stringify(value)
      return String(value)
    }).join(' | ')} |`
  )
  
  return `# ${title}\n\n${headerRow}\n${separatorRow}\n${dataRows.join('\n')}`
}

// Generate sample export data
function generateSamplePredictions(count: number = 100): Record<string, unknown>[] {
  const movements = ['walking', 'running', 'jumping', 'sitting', 'standing', 'squatting']
  const predictions = []
  
  for (let i = 0; i < count; i++) {
    predictions.push({
      frame_id: i,
      timestamp: (i * 0.033).toFixed(3),
      predicted_movement: movements[Math.floor(Math.random() * movements.length)],
      confidence: (0.7 + Math.random() * 0.3).toFixed(4),
      keypoints_detected: 17,
      inference_time_ms: (15 + Math.random() * 20).toFixed(2)
    })
  }
  
  return predictions
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const format = searchParams.get('format') || 'json'
  const count = parseInt(searchParams.get('count') || '100')
  const download = searchParams.get('download') === 'true'
  
  const data = generateSamplePredictions(count)
  let content: string
  let contentType: string
  let filename: string
  
  switch (format) {
    case 'csv':
      content = toCSV(data)
      contentType = 'text/csv'
      filename = 'motionai_predictions.csv'
      break
    case 'markdown':
      content = toMarkdown(data, 'MotionAI Pro - Predictions Export')
      contentType = 'text/markdown'
      filename = 'motionai_predictions.md'
      break
    default:
      content = JSON.stringify(data, null, 2)
      contentType = 'application/json'
      filename = 'motionai_predictions.json'
  }
  
  if (download) {
    return new NextResponse(content, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`
      }
    })
  }
  
  return NextResponse.json({
    format,
    count: data.length,
    filename,
    preview: content.substring(0, 1000) + (content.length > 1000 ? '\n...(truncated)' : ''),
    downloadUrl: `/api/export?format=${format}&count=${count}&download=true`
  })
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { format = 'json', data, filename = 'export' } = body
    
    if (!data) {
      return NextResponse.json({ error: 'No data provided' }, { status: 400 })
    }
    
    let content: string
    let contentType: string
    let actualFilename: string
    
    switch (format) {
      case 'csv':
        content = toCSV(Array.isArray(data) ? data : [data])
        contentType = 'text/csv'
        actualFilename = `${filename}.csv`
        break
      case 'markdown':
        content = toMarkdown(Array.isArray(data) ? data : [data], filename)
        contentType = 'text/markdown'
        actualFilename = `${filename}.md`
        break
      default:
        content = JSON.stringify(data, null, 2)
        contentType = 'application/json'
        actualFilename = `${filename}.json`
    }
    
    return new NextResponse(content, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${actualFilename}"`
      }
    })
  } catch (error) {
    console.error('Export error:', error)
    return NextResponse.json({ error: 'Export failed' }, { status: 500 })
  }
}
