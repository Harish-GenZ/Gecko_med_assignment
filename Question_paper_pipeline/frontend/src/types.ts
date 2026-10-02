// TypeScript types matching backend Pydantic schemas

export type QuestionType = 'Essay' | 'Short Notes' | 'Very Short Answers' | 'MCQ' | 'Unknown'
export type JobStatus    = 'queued' | 'processing' | 'completed' | 'failed'
export type ReviewStatus = 'ok' | 'needs_review'
export type PageType     = 'digital' | 'scanned' | 'mixed'

export interface MCQOption {
  label: string
  text:  string
}

export interface Question {
  number:     string
  text:       string
  type:       QuestionType
  part?:      string | null
  heading?:   string | null
  options:    MCQOption[] | null
  marks:      string | null
  confidence: number
  status:     ReviewStatus
}

export interface ExamMetadata {
  university:   string | null
  exam_month:   string | null
  exam_year:    string | null
  session_code: string | null
  subject:      string | null
  paper_code:   string | null
  max_marks:    string | null
  duration:     string | null
  confidence:   number
  status:       ReviewStatus
}

export interface QuestionPaper {
  paper_index: number
  page_range:  [number, number]
  metadata:    ExamMetadata
  questions:   Question[]
  instructions?: string[] | null
  parts?:      string[] | null
  raw_text:    string | null
}

export interface ExtractionResult {
  document:         string
  total_pages:      number
  question_papers:  QuestionPaper[]
  processing_notes: string[]
}

export interface PageInfo {
  page_number: number
  type:        PageType
  ocr_used:    boolean
  char_count:  number
}

export interface JobStatusResponse {
  job_id:       string
  status:       JobStatus
  filename:     string
  progress:     number
  current_step: string
  error:        string | null
  pages_info:   PageInfo[]
  result:       ExtractionResult | null
}

export interface JobCreateResponse {
  job_id:   string
  status:   JobStatus
  filename: string
  message:  string
}
