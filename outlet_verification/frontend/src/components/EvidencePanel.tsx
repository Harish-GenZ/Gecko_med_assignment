import React from 'react';
import { Layers, FileText, Compass, CheckCircle2, AlertTriangle, ShieldQuestion } from 'lucide-react';
import type { VerificationResponse } from '../types/verification';

interface EvidencePanelProps {
  response: VerificationResponse;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({ response }) => {
  const ev = response.evidence;

  const formatPercent = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'Not available';
    return `${(val * 100).toFixed(1)}%`;
  };

  const formatDistance = (meters: number | null | undefined): string => {
    if (meters === null || meters === undefined) return 'Not available';
    if (meters < 1000) return `${meters.toFixed(1)} m`;
    return `${(meters / 1000).toFixed(2)} km`;
  };

  const formatDecimal = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'Not available';
    return val.toFixed(4);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 space-y-5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-100 uppercase tracking-wider">
            Multimodal Evidence Profile
          </h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          {response.candidates_evaluated} candidates evaluated
        </span>
      </div>

      {/* Signal Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Name Vector Similarity */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-3">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-medium">Name Similarity</span>
            <FileText className="w-3.5 h-3.5 text-slate-500" />
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono">
            {formatPercent(ev?.name_similarity)}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">all-MiniLM-L6-v2 (384d)</p>
        </div>

        {/* Storefront Visual Similarity */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-3">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-medium">Image Similarity</span>
            <Compass className="w-3.5 h-3.5 text-slate-500" />
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono">
            {formatPercent(ev?.image_similarity)}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">open_clip ViT-B/32 (512d)</p>
        </div>

        {/* Physical Proximity */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-3">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-medium">GPS Distance</span>
            <Compass className="w-3.5 h-3.5 text-slate-500" />
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono">
            {formatDistance(ev?.distance_meters)}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            {ev?.geo_proximity_score !== null && ev?.geo_proximity_score !== undefined
              ? `Score: ${(ev.geo_proximity_score * 100).toFixed(1)}%`
              : 'Earthdistance spherical'}
          </p>
        </div>
      </div>

      {/* Fusion & Meta-Evidence Metrics */}
      <div className="border-t border-slate-800/80 pt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div>
          <span className="text-slate-500 block">Duplicate Confidence</span>
          <span className="font-mono font-semibold text-slate-200 text-sm">
            {(response.duplicate_confidence * 100).toFixed(1)}%
          </span>
        </div>

        <div>
          <span className="text-slate-500 block">Evidence Coverage</span>
          <span className="font-mono font-semibold text-slate-200 text-sm">
            {(response.evidence_coverage * 100).toFixed(1)}%
          </span>
        </div>

        <div>
          <span className="text-slate-500 block">Inter-Signal Agreement</span>
          <span className="font-mono font-semibold text-slate-200 text-sm">
            {(response.evidence_agreement * 100).toFixed(1)}%
          </span>
        </div>

        <div>
          <span className="text-slate-500 block">Candidate Margin</span>
          <span className="font-mono font-semibold text-slate-200 text-sm">
            {formatDecimal(response.candidate_margin)}
          </span>
        </div>
      </div>

      {/* Deterministic Reason Codes */}
      <div className="border-t border-slate-800/80 pt-3">
        <span className="text-xs font-semibold text-slate-400 block mb-2 uppercase tracking-wider">
          Decision Audit Codes
        </span>
        <div className="flex flex-wrap gap-1.5">
          {response.reason_codes.map((code) => {
            const isPositive = code.includes('STRONG') || code.includes('MATCH');
            const isWarning = code.includes('AMBIGUOUS') || code.includes('CONFLICTING') || code.includes('INSUFFICIENT') || code.includes('WEAK');

            return (
              <span
                key={code}
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-mono font-medium border ${
                  isPositive
                    ? 'bg-emerald-950/40 text-emerald-300 border-emerald-800/60'
                    : isWarning
                    ? 'bg-amber-950/40 text-amber-300 border-amber-800/60'
                    : 'bg-slate-800 text-slate-300 border-slate-700'
                }`}
              >
                {isPositive ? (
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                ) : isWarning ? (
                  <AlertTriangle className="w-3 h-3 text-amber-400" />
                ) : (
                  <ShieldQuestion className="w-3 h-3 text-slate-400" />
                )}
                {code}
              </span>
            );
          })}
        </div>
      </div>

      {/* Matched Methods */}
      {response.matched_methods.length > 0 && (
        <div className="text-xs text-slate-500 flex items-center gap-1.5 pt-1">
          <span>Surfacing Channels:</span>
          <span className="font-mono text-slate-400">
            {response.matched_methods.join(' + ').toUpperCase()}
          </span>
        </div>
      )}
    </div>
  );
};
