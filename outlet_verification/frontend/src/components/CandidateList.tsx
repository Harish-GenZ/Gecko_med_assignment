import React from 'react';
import { Users } from 'lucide-react';
import type { ScoredCandidate, VerificationEvidence } from '../types/verification';

interface CandidateListProps {
  candidates?: ScoredCandidate[];
  evidence?: VerificationEvidence | null;
}

export const CandidateList: React.FC<CandidateListProps> = ({ candidates = [] }) => {
  if (!candidates || candidates.length === 0) {
    return null;
  }

  const formatPercent = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${(val * 100).toFixed(1)}%`;
  };

  const formatDistance = (meters: number | null | undefined): string => {
    if (meters === null || meters === undefined) return 'N/A';
    if (meters < 1000) return `${meters.toFixed(1)} m`;
    return `${(meters / 1000).toFixed(2)} km`;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-100 uppercase tracking-wider">
            Candidate Matches Pool ({candidates.length})
          </h3>
        </div>
        <span className="text-xs text-slate-500">Ranked by Duplicate Confidence</span>
      </div>

      <div className="space-y-2.5">
        {candidates.map((sc, idx) => {
          const c = sc.candidate;
          const isTop = idx === 0;

          return (
            <div
              key={c.outlet_id || idx}
              className={`p-3.5 rounded border transition-colors ${
                isTop
                  ? 'bg-slate-950/80 border-indigo-900/60'
                  : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5">
                  <span
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-mono font-bold flex-shrink-0 mt-0.5 ${
                      isTop
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {idx + 1}
                  </span>

                  <div>
                    <h5 className="text-xs font-semibold text-slate-200">
                      {c.name}
                    </h5>
                    <p className="text-[11px] text-slate-500 font-mono mt-0.5">
                      ID: {c.outlet_id}
                    </p>
                  </div>
                </div>

                <div className="text-right flex-shrink-0">
                  <span className="text-[10px] text-slate-500 block uppercase font-medium">3-Metric Avg</span>
                  <span
                    className={`text-sm font-bold font-mono ${
                      sc.duplicate_confidence >= 0.75
                        ? 'text-rose-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {(sc.duplicate_confidence * 100).toFixed(1)}%
                  </span>
                  <span className={`text-[9px] font-semibold uppercase block ${
                    sc.duplicate_confidence >= 0.75 ? 'text-rose-400' : 'text-emerald-400'
                  }`}>
                    {sc.duplicate_confidence >= 0.75 ? '>= 75% Dup' : '< 75% Distinct'}
                  </span>
                </div>
              </div>

              {/* Similarity Metrics Row */}
              <div className="grid grid-cols-3 gap-2 mt-2.5 pt-2 border-t border-slate-900 text-xs">
                <div>
                  <span className="text-[10px] text-slate-500 block">Name Sim</span>
                  <span className="font-mono text-slate-300">
                    {formatPercent(c.name_similarity)}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block">Image Sim</span>
                  <span className="font-mono text-slate-300">
                    {formatPercent(c.image_similarity)}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block">Distance</span>
                  <span className="font-mono text-slate-300">
                    {formatDistance(c.distance_meters)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
