import React, { useCallback, useEffect, useRef, useState } from 'react'
import type {
  ExtractionResult, JobStatusResponse, MCQOption,
  PageInfo, Question, QuestionPaper, QuestionType, WorkerInfo
} from './types'
import { uploadDocument, getJobStatus } from './services/api'

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function confClass(c: number): string {
  if (c >= 0.75) return 'high'
  if (c >= 0.5) return 'medium'
  return 'low'
}

function typeBadgeClass(t: QuestionType): string {
  return t.replace(/\s+/g, '-')
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function ConfBar({ value }: { value: number }) {
  return (
    <div className="confidence-bar">
      <div className="conf-track">
        <div
          className={`conf-fill ${confClass(value)}`}
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="conf-label">{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

function ReviewFlag() {
  return <span className="review-flag">⚠ Review</span>
}

function MCQOptions({ options }: { options: MCQOption[] }) {
  return (
    <div className="mcq-options">
      {options.map(opt => (
        <div className="mcq-option" key={opt.label}>
          <span className="mcq-label">{opt.label.toUpperCase()}.</span>
          <span>{opt.text}</span>
        </div>
      ))}
    </div>
  )
}

function QuestionItem({ q }: { q: Question }) {
  let heading = q.heading
  let bodyText = q.text || ''

  if (!heading && bodyText) {
    const firstLine = bodyText.split('\n')[0].trim()
    if (/^(?:write\s+(?:short|brief|a\s+short)\s+notes?\s+on|explain\s+why[\?:]?|give\s+reasons?\s+for[\?:]?|structured\s+long\s+essay)/i.test(firstLine)) {
      heading = firstLine
      bodyText = bodyText.split('\n').slice(1).join('\n').trim()
    }
  }

  return (
    <div className={`question-item ${q.status === 'needs_review' ? 'needs-review' : ''}`}>
      {heading && (
        <div className="question-heading-banner">
          <span className="heading-icon">📌</span>
          <span className="heading-title">{heading}</span>
        </div>
      )}
      <div className="question-top">
        <span className="question-num">Q{q.number}</span>
        <div className="question-text">{bodyText || <em style={{ color: 'var(--text-muted)' }}>[empty]</em>}</div>
        <div className="question-badges">
          {q.part && (
            <span className="part-chip">📑 {q.part}</span>
          )}
          <span className={`type-badge ${typeBadgeClass(q.type)}`}>{q.type}</span>
          {q.marks && (
            <span className="meta-chip marks">🎯 {q.marks}</span>
          )}
          {q.status === 'needs_review' && <ReviewFlag />}
        </div>
      </div>
      {q.options && q.options.length > 0 && <MCQOptions options={q.options} />}
      {q.confidence_breakdown ? (
        <div className="q-confidence-row">
          <div className="q-conf-bar-wrap">
            <ConfBar value={q.confidence} />
          </div>
          <div className="q-conf-chips">
            <span className="q-conf-chip">Boundary: {(q.confidence_breakdown.boundary * 100).toFixed(0)}%</span>
            <span className="q-conf-chip">Category: {(q.confidence_breakdown.classification * 100).toFixed(0)}%</span>
            {q.confidence_breakdown.ocr !== null && q.confidence_breakdown.ocr !== undefined && (
              <span className="q-conf-chip">OCR: {(q.confidence_breakdown.ocr * 100).toFixed(0)}%</span>
            )}
          </div>
        </div>
      ) : (
        <ConfBar value={q.confidence} />
      )}
      {q.anomalies && q.anomalies.length > 0 && (
        <div className="q-anomalies-list">
          {q.anomalies.map((a, i) => (
            <span key={i} className={`q-anomaly-badge sev-${a.severity}`}>⚠️ {a.reason}</span>
          ))}
        </div>
      )}
    </div>
  )
}

function PaperQualityAuditCard({ paper }: { paper: QuestionPaper }) {
  const conf = paper.confidence
  if (!conf) return null

  const status = paper.confidence_level || 'HIGH'
  const statusClass = status.toLowerCase().replace('_', '-')

  return (
    <div className={`quality-audit-card ${statusClass}`}>
      <div className="audit-header">
        <div className="audit-title-group">
          <span className="audit-badge-icon">🛡️</span>
          <span className="audit-title">Quality &amp; Extraction Confidence Audit</span>
        </div>
        <div className="audit-status-group">
          {paper.vision_verified ? (
            paper.vision_status === 'CORRECTED' ? (
              <span className="vision-status-chip corrected">✨ Vision Corrected</span>
            ) : paper.vision_status === 'MANUAL_REVIEW' ? (
              <span className="vision-status-chip manual-review">⚠️ Manual Review Required</span>
            ) : paper.vision_status === 'FAILED' ? (
              <span className="vision-status-chip failed">❌ Vision Failed (Manual Review)</span>
            ) : (
              <span className="vision-status-chip verified">✅ Vision Verified</span>
            )
          ) : paper.needs_visual_verification ? (
            <span className="vision-status-chip pending">⏳ Vision Verification Pending</span>
          ) : null}

          {paper.vision_confidence !== undefined && paper.vision_confidence !== null && (
            <span className="vision-conf-pill">Vision Conf: {(paper.vision_confidence * 100).toFixed(0)}%</span>
          )}

          <span className={`status-pill status-${statusClass}`}>
            Status: {status.replace('_', ' ')}
          </span>
        </div>
      </div>

      <div className="audit-metrics-grid">
        <div className="audit-metric-box overall-box">
          <span className="metric-label">Overall</span>
          <span className="metric-value">{(conf.overall_confidence * 100).toFixed(0)}%</span>
          <div className="metric-bar">
            <div className={`metric-fill ${confClass(conf.overall_confidence)}`} style={{ width: `${conf.overall_confidence * 100}%` }} />
          </div>
        </div>

        <div className="audit-metric-box">
          <span className="metric-label">Structure</span>
          <span className="metric-value">{(conf.structure_confidence * 100).toFixed(0)}%</span>
          <div className="metric-bar">
            <div className={`metric-fill ${confClass(conf.structure_confidence)}`} style={{ width: `${conf.structure_confidence * 100}%` }} />
          </div>
        </div>

        <div className="audit-metric-box">
          <span className="metric-label">OCR Quality</span>
          <span className="metric-value">
            {conf.ocr_confidence !== null && conf.ocr_confidence !== undefined
              ? `${(conf.ocr_confidence * 100).toFixed(0)}%`
              : 'N/A'}
          </span>
          <div className="metric-bar">
            <div className={`metric-fill ${confClass(conf.ocr_confidence ?? 0.85)}`} style={{ width: `${(conf.ocr_confidence ?? 0.85) * 100}%` }} />
          </div>
        </div>

        <div className="audit-metric-box">
          <span className="metric-label">Segmentation</span>
          <span className="metric-value">{(conf.segmentation_confidence * 100).toFixed(0)}%</span>
          <div className="metric-bar">
            <div className={`metric-fill ${confClass(conf.segmentation_confidence)}`} style={{ width: `${conf.segmentation_confidence * 100}%` }} />
          </div>
        </div>

        <div className="audit-metric-box">
          <span className="metric-label">Classification</span>
          <span className="metric-value">{(conf.classification_confidence * 100).toFixed(0)}%</span>
          <div className="metric-bar">
            <div className={`metric-fill ${confClass(conf.classification_confidence)}`} style={{ width: `${conf.classification_confidence * 100}%` }} />
          </div>
        </div>
      </div>

      {paper.anomalies && paper.anomalies.length > 0 && (
        <div className="audit-reasons-box">
          <span className="reasons-heading">⚠️ Quality Warnings &amp; Review Reasons:</span>
          <ul className="reasons-list">
            {paper.anomalies.map((anom, idx) => (
              <li key={idx} className={`anomaly-item anomaly-${anom.severity}`}>
                <span className="anomaly-sev-badge">[{anom.severity.toUpperCase()}]</span>
                <span className="anomaly-reason">{anom.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function PaperCard({ paper }: { paper: QuestionPaper }) {
  const [open, setOpen] = useState(true)
  const [activeType, setActiveType] = useState<string>('All')

  const { metadata } = paper

  const typeCounts = paper.questions.reduce<Record<string, number>>((acc, q) => {
    acc[q.type] = (acc[q.type] ?? 0) + 1
    acc['All'] = (acc['All'] ?? 0) + 1
    return acc
  }, { All: 0 })

  const filtered = activeType === 'All'
    ? paper.questions
    : paper.questions.filter(q => q.type === activeType)

  const types: QuestionType[] = ['Essay', 'Short Notes', 'Very Short Answers', 'MCQ']

  return (
    <div className="paper-card">
      {/* Header */}
      <div className="paper-header" onClick={() => setOpen(o => !o)}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 16, fontWeight: 700 }}>
              📄 Paper {paper.paper_index + 1}
            </span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              pages {paper.page_range[0]}–{paper.page_range[1]}
            </span>
          </div>
          <div className="paper-meta">
            {paper.parts && paper.parts.length > 0 && (
              <span className="meta-chip part-chip-meta">📑 Parts: {paper.parts.join(', ')}</span>
            )}
            {metadata.university && (
              <span className="meta-chip university">🏛 {metadata.university.slice(0, 50)}</span>
            )}
            {metadata.subject && (
              <span className="meta-chip subject">📚 {metadata.subject}</span>
            )}
            {(metadata.exam_month || metadata.exam_year) && (
              <span className="meta-chip date">
                📅 {[metadata.exam_month, metadata.exam_year].filter(Boolean).join(' ')}
              </span>
            )}
            {metadata.session_code && (
              <span className="meta-chip code">🔖 {metadata.session_code}</span>
            )}
            {metadata.max_marks && (
              <span className="meta-chip marks">🎯 {metadata.max_marks} marks</span>
            )}
            {metadata.status === 'needs_review' && (
              <span className="meta-chip review">⚠ Metadata needs review</span>
            )}
          </div>
        </div>
        <div className="paper-stats">
          <span className="paper-stat-pill">{paper.questions.length} questions</span>
          {paper.confidence ? (
            <span className={`paper-stat-pill audit-stat-pill status-${(paper.confidence_level || 'HIGH').toLowerCase().replace('_', '-')}`}>
              Overall: {(paper.confidence.overall_confidence * 100).toFixed(0)}% ({paper.confidence_level?.replace('_', ' ')})
            </span>
          ) : (
            <span className="paper-stat-pill">
              {(metadata.confidence * 100).toFixed(0)}% metadata conf
            </span>
          )}
          <span className={`collapse-icon ${open ? 'open' : ''}`}>▼</span>
        </div>
      </div>

      {/* Body */}
      {open && (
        <div className="questions-container">
          {/* Quality Audit Card */}
          <PaperQualityAuditCard paper={paper} />
          {/* Instructions Box */}
          {paper.instructions && paper.instructions.length > 0 && (
            <div className="instructions-card">
              <div className="instructions-card-header">
                <span>📋 Exam Instructions & Guidelines ({paper.instructions.length})</span>
              </div>
              <ul className="instructions-list">
                {paper.instructions.map((inst, idx) => (
                  <li key={idx} className="instruction-item">
                    {inst}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Filter buttons */}
          <div className="questions-filter">
            {['All', ...types].map(t => (
              <button
                key={t}
                className={`filter-btn ${activeType === t ? 'active' : ''}`}
                onClick={() => setActiveType(t)}
              >
                {t} {typeCounts[t] !== undefined ? `(${typeCounts[t]})` : '(0)'}
              </button>
            ))}
          </div>

          {filtered.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🔍</div>
              <div>No questions of this type found.</div>
            </div>
          ) : (
            filtered.map((q, idx) => {
              const prevPart = idx > 0 ? filtered[idx - 1].part : null
              const showPartHeader = q.part && q.part !== prevPart
              return (
                <div key={q.number || idx} style={{ width: '100%' }}>
                  {showPartHeader && (
                    <div className="part-section-divider">
                      <span className="part-section-title">📑 {q.part}</span>
                      <div className="part-section-line" />
                    </div>
                  )}
                  <QuestionItem q={q} />
                </div>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}

function JsonModal({
  result,
  onClose,
}: {
  result: ExtractionResult
  onClose: () => void
}) {
  const json = JSON.stringify(result, null, 2)

  function download() {
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `gecko_med_extraction_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal">
        <div className="modal-header">
          <span className="modal-title">📋 Extraction JSON</span>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <button className="btn btn-outline" onClick={download}>⬇ Download</button>
            <button className="modal-close" onClick={onClose}>✕</button>
          </div>
        </div>
        <div className="modal-body">
          <pre className="json-view">{json}</pre>
        </div>
      </div>
    </div>
  )
}

function PageInfoStrip({ pages }: { pages: PageInfo[] }) {
  const digital = pages.filter(p => p.type === 'digital').length
  const scanned = pages.filter(p => p.type === 'scanned').length
  const mixed = pages.filter(p => p.type === 'mixed').length

  return (
    <div className="page-info-strip">
      <span className="page-chip digital">✅ {digital} digital</span>
      <span className="page-chip scanned">📷 {scanned} scanned (OCR)</span>
      {mixed > 0 && <span className="page-chip mixed">🔀 {mixed} mixed</span>}
    </div>
  )
}

function WorkerCardItem({ worker }: { worker: WorkerInfo }) {
  const isDone = worker.status.toLowerCase().includes('completed')
  const isOCR = worker.status.toLowerCase().includes('neural') || worker.status.toLowerCase().includes('paddleocr')
  const startTimeRef = useRef<number>(Date.now() - (worker.elapsed_seconds ?? 0) * 1000)
  const [liveSecs, setLiveSecs] = useState<number>(worker.elapsed_seconds ?? 0)

  useEffect(() => {
    if (isDone) {
      setLiveSecs(worker.elapsed_seconds ?? 0)
      return
    }
    const interval = setInterval(() => {
      setLiveSecs(Math.max(0, Math.round(((Date.now() - startTimeRef.current) / 1000) * 10) / 10))
    }, 400)
    return () => clearInterval(interval)
  }, [isDone, worker.elapsed_seconds])

  return (
    <div
      className={`worker-card ${isDone ? 'worker-card-done' : 'worker-card-active'}`}
    >
      <div className="worker-card-header">
        <div className="worker-badge">
          <span className="worker-gear">{isDone ? '✅' : '⚙️'}</span>
          <span className="worker-label">Worker <strong className="worker-pid">PID {worker.worker_pid}</strong></span>
        </div>
        <span className="worker-page-badge">📄 Page {worker.page_number}</span>
      </div>

      <div className="worker-card-body">
        <div className="worker-status-row">
          <span className={`worker-beacon ${isDone ? 'beacon-done' : isOCR ? 'beacon-ocr' : 'beacon-active'}`} />
          <span className="worker-status-text" title={worker.status}>
            {worker.status}
          </span>
        </div>

        <div className="worker-timer">
          <span className="timer-icon">⏱️</span>
          <span className="timer-val">
            {isDone
              ? `${(worker.elapsed_seconds ?? liveSecs).toFixed(1)}s (completed)`
              : `${liveSecs.toFixed(1)}s active`}
          </span>
        </div>
      </div>

      {!isDone && (
        <div className="worker-mini-track">
          <div className={`worker-mini-fill ${isOCR ? 'fill-neural' : 'fill-active'}`} />
        </div>
      )}
    </div>
  )
}

function WorkersMonitor({ workers }: { workers: WorkerInfo[] }) {
  if (!workers || workers.length === 0) return null

  return (
    <div className="workers-section">
      <div className="workers-header">
        <div className="workers-title">
          <span className="workers-pulse-icon">⚡</span>
          <span>Parallel OCR Workers ({workers.length} Processes)</span>
        </div>
        <span className="workers-tag">Page-level Multiprocessing</span>
      </div>

      <div className="worker-grid">
        {workers.map((worker) => (
          <WorkerCardItem
            key={`${worker.worker_pid}-${worker.page_number}`}
            worker={worker}
          />
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────

type AppPhase = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export default function App() {
  const [phase, setPhase] = useState<AppPhase>('idle')
  const [jobState, setJobState] = useState<JobStatusResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [showJson, setShowJson] = useState(false)
  const [displayProgress, setDisplayProgress] = useState<number>(0)
  const [elapsedTime, setElapsedTime] = useState<number>(0)
  const jobStartTimeRef = useRef<number>(0)
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Cleanup on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  // Live elapsed timer while processing
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null
    if (phase === 'processing' || phase === 'uploading') {
      timer = setInterval(() => {
        if (jobStartTimeRef.current > 0) {
          setElapsedTime(Math.floor((Date.now() - jobStartTimeRef.current) / 1000))
        }
      }, 500)
    }
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [phase])

  // Sync displayProgress with jobState progress
  useEffect(() => {
    if (phase === 'processing') {
      const target = jobState?.progress ?? 5
      setDisplayProgress(prev => Math.max(prev, target))
    } else if (phase === 'uploading') {
      setDisplayProgress(5)
    } else if (phase === 'done') {
      setDisplayProgress(100)
    } else if (phase === 'idle') {
      setDisplayProgress(0)
    }
  }, [jobState?.progress, phase])

  // Micro-progress trickle so the bar continuously feels active during heavy OCR inference
  useEffect(() => {
    if (phase !== 'processing') return
    const interval = setInterval(() => {
      setDisplayProgress(prev => {
        const backendProgress = jobState?.progress ?? 5
        if (backendProgress < 70 && prev < 68) {
          return Math.min(prev + 0.35, 68)
        }
        if (backendProgress >= 70 && backendProgress < 95 && prev < 93) {
          return Math.min(prev + 0.5, 93)
        }
        return prev
      })
    }, 250)
    return () => clearInterval(interval)
  }, [phase, jobState?.progress])

  async function handleFile(file: File) {
    if (!file) return
    setPhase('uploading')
    setJobState(null)
    setErrorMsg(null)
    setDisplayProgress(5)
    jobStartTimeRef.current = Date.now()
    setElapsedTime(0)

    try {
      const created = await uploadDocument(file)
      setPhase('processing')
      startPolling(created.job_id)
    } catch (e: unknown) {
      setPhase('error')
      setErrorMsg(e instanceof Error ? e.message : 'Upload failed')
    }
  }

  function startPolling(jobId: string) {
    if (pollRef.current) clearInterval(pollRef.current)
    let consecutiveErrors = 0
    pollRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId)
        consecutiveErrors = 0
        setJobState(status)
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollRef.current!)
          pollRef.current = null
          setPhase(status.status === 'completed' ? 'done' : 'error')
          if (status.status === 'failed') setErrorMsg(status.error ?? 'Processing failed')
        }
      } catch (e) {
        consecutiveErrors++
        console.error('Poll error:', e)
        if (consecutiveErrors >= 5) {
          clearInterval(pollRef.current!)
          pollRef.current = null
          setPhase('error')
          setErrorMsg('Lost connection to processing job. Please re-upload your document.')
        }
      }
    }, 1000)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }, [])

  function reset() {
    if (pollRef.current) clearInterval(pollRef.current)
    setPhase('idle')
    setJobState(null)
    setErrorMsg(null)
    setDisplayProgress(0)
    setElapsedTime(0)
    jobStartTimeRef.current = 0
  }

  const result = jobState?.result

  return (
    <div className="app">
      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="dot" />
          <span>Gecko Med</span>
        </div>
        <span className="navbar-badge">Question Paper Pipeline</span>
      </nav>

      <main className="main-content">
        {/* Hero */}
        {phase === 'idle' && (
          <div className="hero">
            <div className="hero-eyebrow">🏥 Medical Examination AI</div>
            <h1>Extract &amp; Classify<br /><span>Question Papers</span></h1>
            <p className="hero-subtitle">
              Upload scanned or digital exam PDFs. Our AI pipeline detects paper
              boundaries, extracts questions, and classifies them — automatically.
            </p>
            <div className="hero-stats">
              <div className="hero-stat">
                <span className="hero-stat-value">98%</span>
                <span className="hero-stat-label">Scanned PDFs</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value">4</span>
                <span className="hero-stat-label">Question Types</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value">3+</span>
                <span className="hero-stat-label">Universities</span>
              </div>
            </div>
          </div>
        )}

        {/* Upload zone */}
        {(phase === 'idle') && (
          <section className="upload-section">
            <div
              className={`upload-zone ${dragging ? 'dragging' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
            >
              <span className="upload-icon">📄</span>
              <div className="upload-title">Drop your PDF here</div>
              <div className="upload-subtitle">or click to browse your files</div>
              <button className="upload-btn" onClick={e => { e.stopPropagation(); fileRef.current?.click() }}>
                📂 Choose File
              </button>
              <div className="upload-hint">Supports: PDF, JPEG, PNG · Max 200 MB</div>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png"
              style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
            />
          </section>
        )}

        {/* Uploading */}
        {phase === 'uploading' && (
          <div className="progress-card">
            <div className="progress-header">
              <span className="progress-label"><span className="spinner" /> Uploading…</span>
              <span className="progress-pct">0%</span>
            </div>
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: '5%' }} />
            </div>
          </div>
        )}

        {/* Processing */}
        {phase === 'processing' && jobState && (
          <div className="progress-card">
            <div className="progress-header">
              <div className="progress-title-area">
                <span className="spinner" />
                <span className="progress-main-title">Processing Document</span>
                <span className="progress-filename" title={jobState.filename}>{jobState.filename}</span>
              </div>
              <div className="progress-header-right">
                <span className="progress-elapsed">⏱️ {elapsedTime}s</span>
                <span className="progress-pct">{Math.round(displayProgress)}%</span>
              </div>
            </div>

            <div className="progress-bar-track">
              <div
                className="progress-bar-fill animated-shimmer"
                style={{ width: `${Math.min(Math.max(displayProgress, 5), 100)}%` }}
              />
            </div>

            <div className="progress-footer-row">
              <div className="progress-step">
                <span className="step-icon">📍</span>
                <span className="step-text">{jobState.current_step}</span>
              </div>
              {jobState.workers_info && jobState.workers_info.length > 1 && (
                <span className="parallel-badge">
                  ⚡ {jobState.workers_info.length}x Multi-Process Parallel Mode
                </span>
              )}
            </div>

            {/* Parallel Workers Monitor */}
            {jobState.workers_info && jobState.workers_info.length > 0 ? (
              <WorkersMonitor workers={jobState.workers_info} />
            ) : (
              jobState.progress < 15 && (
                <div className="workers-placeholder">
                  <span className="spinner mini-spinner" />
                  <span>Preparing pages and initializing parallel OCR workers...</span>
                </div>
              )
            )}
          </div>
        )}

        {/* Error */}
        {phase === 'error' && (
          <div className="error-card">
            <div className="error-icon">❌</div>
            <p className="error-message">{errorMsg ?? 'An unknown error occurred.'}</p>
            <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={reset}>
              ↩ Try Again
            </button>
          </div>
        )}

        {/* Results */}
        {phase === 'done' && result && (
          <>
            <div className="results-header">
              <div className="results-title">
                ✅ Extraction Complete
                <span style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 400 }}>
                  — {result.document}
                </span>
              </div>
              <div className="results-actions">
                <button className="btn btn-outline" onClick={reset}>↩ New Upload</button>
                <button className="btn btn-primary" onClick={() => setShowJson(true)}>
                  {'{ }'} View JSON
                </button>
              </div>
            </div>

            {/* Vision Verification Alert Banner */}
            {result.needs_manual_review ? (
              <div className="vision-alert-banner warning-mode">
                <span className="vision-alert-icon">⚠️</span>
                <div className="vision-alert-content">
                  <div className="vision-alert-title">Manual Review Required</div>
                  <div className="vision-alert-desc">
                    Structural quality audit or Vision verification flagged uncertainties: {result.visual_verification_reasons?.join(', ')}.
                  </div>
                </div>
              </div>
            ) : result.vision_verified_pages && result.vision_verified_pages.length > 0 ? (
              <div className="vision-alert-banner verified-mode">
                <span className="vision-alert-icon">✨</span>
                <div className="vision-alert-content">
                  <div className="vision-alert-title">Multimodal Vision Verification Applied</div>
                  <div className="vision-alert-desc">
                    Pages {result.vision_verified_pages.join(', ')} were verified with Gemini Vision and structural corrections applied.
                  </div>
                </div>
              </div>
            ) : result.needs_visual_verification ? (
              <div className="vision-alert-banner">
                <span className="vision-alert-icon">👁️</span>
                <div className="vision-alert-content">
                  <div className="vision-alert-title">Visual Verification Recommended for this Document</div>
                  <div className="vision-alert-desc">
                    Structural quality audit detected anomalies: {result.visual_verification_reasons?.join(', ')}.
                  </div>
                </div>
              </div>
            ) : null}

            {/* Page info */}
            {jobState?.pages_info && jobState.pages_info.length > 0 && (
              <PageInfoStrip pages={jobState.pages_info} />
            )}

            {/* Processing notes */}
            {result.processing_notes.length > 0 && (
              <div className="notes-section">
                <div className="notes-title">⚠ Processing Notes</div>
                <ul className="notes-list">
                  {result.processing_notes.map((n, i) => <li key={i}>• {n}</li>)}
                </ul>
              </div>
            )}

            {/* Papers */}
            {result.question_papers.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">📭</div>
                <div>No question papers were detected in this document.</div>
              </div>
            ) : (
              result.question_papers.map(paper => (
                <PaperCard key={paper.paper_index} paper={paper} />
              ))
            )}
          </>
        )}
      </main>

      {showJson && result && (
        <JsonModal result={result} onClose={() => setShowJson(false)} />
      )}
    </div>
  )
}
