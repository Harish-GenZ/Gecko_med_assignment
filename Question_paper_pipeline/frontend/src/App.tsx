import React, { useCallback, useEffect, useRef, useState } from 'react'
import type {
  ExtractionResult, JobStatusResponse, MCQOption,
  PageInfo, Question, QuestionPaper, QuestionType
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
      <ConfBar value={q.confidence} />
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
          <span className="paper-stat-pill">
            {(metadata.confidence * 100).toFixed(0)}% metadata conf
          </span>
          <span className={`collapse-icon ${open ? 'open' : ''}`}>▼</span>
        </div>
      </div>

      {/* Body */}
      {open && (
        <div className="questions-container">
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
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Cleanup on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  async function handleFile(file: File) {
    if (!file) return
    setPhase('uploading')
    setJobState(null)
    setErrorMsg(null)

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
    }, 1500)
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
        <span className="navbar-badge">Question Paper AI</span>
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
              <span className="progress-label"><span className="spinner" /> Processing…</span>
              <span className="progress-pct">{jobState.progress}%</span>
            </div>
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${jobState.progress}%` }} />
            </div>
            <div className="progress-step">📌 {jobState.current_step}</div>
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
