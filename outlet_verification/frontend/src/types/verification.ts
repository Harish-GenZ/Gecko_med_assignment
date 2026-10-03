export type VerificationDecision = 'DUPLICATE' | 'GENUINE' | 'NEEDS_REVIEW' | 'REJECTED';

export interface AuthenticityAssessment {
  is_authentic_store: boolean;
  status: 'AUTHENTIC' | 'SUSPICIOUS' | 'REJECTED';
  authenticity_confidence: number;
  visual_is_store: boolean;
  visual_probability: number;
  visual_category: string;
  name_is_valid_outlet: boolean;
  name_category: string;
  ocr_lines: string[];
  ocr_retail_detected: boolean;
  ocr_name_match_score: number | null;
  rejection_reasons: string[];
  summary: string;
}

export type ReasonCode =
  | 'STRONG_MULTIMODAL_MATCH'
  | 'STRONG_IMAGE_GEO_MATCH'
  | 'HIGH_NAME_IMAGE_GEO_MATCH'
  | 'NO_MATCHING_CANDIDATE'
  | 'CONFLICTING_NAME_IMAGE'
  | 'AMBIGUOUS_TOP_CANDIDATES'
  | 'INSUFFICIENT_EVIDENCE'
  | 'POSSIBLE_SAME_BRAND_DIFFERENT_LOCATION'
  | 'WEAK_EVIDENCE_REQUIRES_REVIEW'
  | 'LOW_CONFIDENCE_GENUINE'
  | string;

export interface MatchedOutlet {
  outlet_id: string;
  name: string;
  image_url: string;
  latitude: number;
  longitude: number;
}

export interface VerificationEvidence {
  name_similarity: number | null;
  image_similarity: number | null;
  distance_meters: number | null;
  geo_proximity_score: number | null;
}

export interface CandidateEvidence {
  outlet_id: string;
  name: string;
  latitude: number;
  longitude: number;
  image_url: string;
  name_similarity: number | null;
  image_similarity: number | null;
  distance_meters: number | null;
  matched_methods: string[];
}

export interface ScoredCandidate {
  candidate: CandidateEvidence;
  geo_proximity_score: number | null;
  base_score: number;
  evidence_coverage: number;
  evidence_agreement: number;
  duplicate_confidence: number;
  flags?: {
    strong_duplicate_evidence?: boolean;
    conflicting_evidence?: boolean;
    possible_same_brand_different_location?: boolean;
  };
}

export interface VerificationResponse {
  decision: VerificationDecision;
  duplicate_confidence: number;
  evidence_coverage: number;
  evidence_agreement: number;
  matched_outlet: MatchedOutlet | null;
  evidence: VerificationEvidence | null;
  matched_methods: string[];
  reason_codes: ReasonCode[];
  reason_summary: string;
  candidate_margin: number | null;
  evidence_status: string | null;
  candidates_evaluated: number;
  registered?: boolean;
  registered_outlet_id?: string | null;
  stored_image_url?: string | null;
  candidates?: ScoredCandidate[];
  authenticity?: AuthenticityAssessment | null;
}

export interface BackendHealth {
  status: 'ok' | 'degraded' | 'offline';
  backend?: {
    name: string;
    status: string;
    environment: string;
  };
  database?: {
    status: string;
    message: string;
    target?: string;
  };
}

export interface VerificationSubmissionForm {
  name: string;
  latitude: string;
  longitude: string;
  image: File | null;
  autoRegister?: boolean;
}

export interface ImageMetadata {
  file: File;
  previewUrl: string;
  width?: number;
  height?: number;
  sizeKb: number;
}
