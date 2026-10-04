import React from 'react';
import { Store, MapPin, ImageOff } from 'lucide-react';
import type { MatchedOutlet, VerificationEvidence } from '../types/verification';

interface CandidateCardProps {
  outlet: MatchedOutlet;
  evidence?: VerificationEvidence | null;
  submittedImagePreview?: string | null;
  confidence?: number | null;
  isMatched?: boolean;
}

export const CandidateCard: React.FC<CandidateCardProps> = ({
  outlet,
  evidence,
  submittedImagePreview,
  confidence,
  isMatched = false,
}) => {
  const formatPercent = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${(val * 100).toFixed(1)}%`;
  };

  const formatDistance = (meters: number | null | undefined): string => {
    if (meters === null || meters === undefined) return 'N/A';
    if (meters < 1000) return `${meters.toFixed(1)} m`;
    return `${(meters / 1000).toFixed(2)} km`;
  };

  const hasReferenceImage = Boolean(outlet.image_url && outlet.image_url.trim() !== '');

  return (
    <div
      className={`border rounded-lg overflow-hidden transition-colors ${
        isMatched
          ? 'bg-slate-900 border-indigo-500/60 shadow-lg shadow-indigo-950/20'
          : 'bg-slate-900/70 border-slate-800'
      }`}
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-800/80 flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div
            className={`w-9 h-9 rounded flex items-center justify-center flex-shrink-0 ${
              isMatched
                ? 'bg-indigo-950 text-indigo-400 border border-indigo-800'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            <Store className="w-5 h-5" />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-slate-100">{outlet.name}</h4>
              {isMatched && (
                <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  Top Match
                </span>
              )}
            </div>

            <p className="text-xs text-slate-500 font-mono mt-0.5">
              ID: {outlet.outlet_id}
            </p>

            <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-400">
              <span className="inline-flex items-center gap-1 font-mono">
                <MapPin className="w-3 h-3 text-slate-500" />
                {outlet.latitude.toFixed(5)}, {outlet.longitude.toFixed(5)}
              </span>
              {evidence?.distance_meters !== null && evidence?.distance_meters !== undefined && (
                <>
                  <span>•</span>
                  <span className="font-mono text-indigo-300 font-medium">
                    {formatDistance(evidence.distance_meters)} away
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {confidence !== null && confidence !== undefined && (
          <div className="text-right flex-shrink-0">
            <span className="text-[11px] text-slate-500 block uppercase font-medium">3-Metric Average</span>
            <span className={`text-lg font-bold font-mono ${
              confidence >= 0.75 ? 'text-rose-400' : 'text-emerald-400'
            }`}>
              {(confidence * 100).toFixed(1)}%
            </span>
            <span className={`text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded block mt-0.5 ${
              confidence >= 0.75
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
            }`}>
              {confidence >= 0.75 ? 'Duplicate (>= 75%)' : 'Distinct (< 75%)'}
            </span>
          </div>
        )}
      </div>

      {/* Side-by-Side Visual Comparison (Step 8) */}
      <div className="p-4 bg-slate-950/40">
        <span className="text-xs font-semibold text-slate-400 block mb-2 uppercase tracking-wider">
          Visual Comparison
        </span>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Submitted Storefront */}
          <div className="border border-slate-800 rounded bg-slate-950 p-2 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5 font-medium">
              <span>Submitted Photo</span>
              {evidence?.name_similarity !== null && evidence?.name_similarity !== undefined && (
                <span className="text-[11px] font-mono text-slate-400">
                  Name Sim: {formatPercent(evidence.name_similarity)}
                </span>
              )}
            </div>

            <div className="w-full h-36 rounded border border-slate-800 bg-slate-900 flex items-center justify-center overflow-hidden">
              {submittedImagePreview ? (
                <img
                  src={submittedImagePreview}
                  alt="Submitted storefront"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="text-center text-slate-600 p-2">
                  <ImageOff className="w-6 h-6 mx-auto mb-1" />
                  <span className="text-xs">No submitted photo preview</span>
                </div>
              )}
            </div>
          </div>

          {/* Database Reference Photo */}
          <div className="border border-slate-800 rounded bg-slate-950 p-2 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5 font-medium">
              <span>Reference Database Photo</span>
              {evidence?.image_similarity !== null && evidence?.image_similarity !== undefined && (
                <span className="text-[11px] font-mono text-indigo-400 font-semibold">
                  CLIP Sim: {formatPercent(evidence.image_similarity)}
                </span>
              )}
            </div>

            <div className="w-full h-36 rounded border border-slate-800 bg-slate-900 flex items-center justify-center overflow-hidden">
              {hasReferenceImage ? (
                <img
                  src={outlet.image_url}
                  alt={outlet.name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    // Fallback if URL is a local or invalid path
                    (e.target as HTMLElement).style.display = 'none';
                    const parent = (e.target as HTMLElement).parentElement;
                    if (parent) {
                      parent.innerHTML = `
                        <div class="text-center text-slate-500 p-2">
                          <span class="text-xs font-mono block text-slate-400 break-all">${outlet.image_url}</span>
                          <span class="text-[11px] text-slate-500 mt-1 block">Reference image file on record</span>
                        </div>
                      `;
                    }
                  }}
                />
              ) : (
                <div className="text-center text-slate-600 p-2">
                  <ImageOff className="w-6 h-6 mx-auto mb-1" />
                  <span className="text-xs">Reference image unavailable</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
