// TypeScript types matching backend Pydantic schemas

export type QuestionType = 'Essay' | 'Short Notes' | 'Very Short Answers' | 'MCQ' | 'Unknown'
export type JobStatus    = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
export type ReviewStatus = 'ok' | 'needs_review'
export type PageType     = 'digital' | 'scanned' | 'mixed'

export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW' | 'REVIEW_REQUIRED'
export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical'

export interface StructuralAnomaly {
  type: string
  severity: AnomalySeverity
  text_preview?: string | null
  reason: string
}

export interface QuestionConfidenceBreakdown {
  ocr?: number | null
  boundary: number
  classification: number
  overall: number
}

export interface PaperConfidenceBreakdown {
  ocr_confidence?: number | null
  ocr_confidence_available: boolean
  structure_confidence: number
  segmentation_confidence: number
  metadata_confidence: number
  classification_confidence: number
  consistency_confidence: number
  overall_confidence: number
}

export type PageProcessingState = 
  | 'OCR_PROCESSING'
  | 'OCR_COMPLETE'
  | 'AUDIT_COMPLETE'
  | 'VISION_PENDING'
  | 'VISION_PROCESSING'
  | 'VISION_VERIFIED'
  | 'VISION_CORRECTED'
  | 'VISION_FAILED'
  | 'FINALIZED'

export interface PageConfidenceAudit {
  page: number
  confidence: PaperConfidenceBreakdown
  status: ConfidenceLevel
  anomalies: StructuralAnomaly[]
  needs_visual_verification: boolean
  visual_verification_reasons: string[]
  processing_state?: PageProcessingState
  vision_confidence?: number | null
  vision_status?: string | null
  vision_error?: string | null
  needs_manual_review?: boolean
  corrections_applied?: number
}

export interface MCQOption {
  label: string
  text:  string
}

export interface Question {
  number:               string
  text:                 string
  type:                 QuestionType
  part?:                string | null
  heading?:             string | null
  options:              MCQOption[] | null
  marks:                string | null
  confidence:           number
  confidence_breakdown?: QuestionConfidenceBreakdown | null
  status:               ReviewStatus
  confidence_level?:    ConfidenceLevel | null
  anomalies?:           StructuralAnomaly[]
}

export interface ExamMetadata {
  university:     string | null
  exam_month:     string | null
  exam_year:      string | null
  session_code:   string | null
  subject:        string | null
  paper_code:     string | null
  max_marks:      string | null
  duration:       string | null
  confidence:     number
  status:         ReviewStatus
  missing_fields?: string[]
}

export interface QuestionPaper {
  paper_index:                  number
  page_range:                   [number, number]
  metadata:                     ExamMetadata
  questions:                    Question[]
  instructions?:                string[] | null
  parts?:                       string[] | null
  raw_text:                     string | null
  confidence?:                  PaperConfidenceBreakdown | null
  confidence_level?:            ConfidenceLevel
  needs_visual_verification?:   boolean
  visual_verification_reasons?: string[]
  anomalies?:                   StructuralAnomaly[]
  vision_verified?:              boolean
  vision_confidence?:            number | null
  vision_status?:                string | null
  needs_manual_review?:          boolean
}

export interface ExtractionResult {
  document:                     string
  total_pages:                  number
  question_papers:              QuestionPaper[]
  page_audits?:                 PageConfidenceAudit[]
  processing_notes:             string[]
  needs_visual_verification?:   boolean
  visual_verification_reasons?: string[]
  vision_verified_pages?:        number[]
  needs_manual_review?:          boolean
}

export interface PageInfo {
  page_number:      number
  type:             PageType
  ocr_used:         boolean
  char_count:       number
  processing_state?: PageProcessingState
}

export interface WorkerInfo {
  worker_pid:      number
  page_number:     number
  status:          string
  elapsed_seconds?: number
}

export interface JobStatusResponse {
  job_id:        string
  status:        JobStatus
  filename:      string
  progress:      number
  current_step:  string
  error:         string | null
  pages_info:    PageInfo[]
  result:        ExtractionResult | null
  workers_info?: WorkerInfo[] | null
}

export interface JobCreateResponse {
  job_id:   string
  status:   JobStatus
  filename: string
  message:  string
}

