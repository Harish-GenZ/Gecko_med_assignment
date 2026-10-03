import React from 'react';
import { Cpu, Database, Search, Shield, Loader2 } from 'lucide-react';

export const LoadingState: React.FC = () => {
  const stages = [
    { label: 'Embedding Generation', desc: 'MiniLM 384d text + CLIP 512d vision inference', icon: <Cpu className="w-4 h-4 text-indigo-400" /> },
    { label: 'Multi-Channel Retrieval', desc: 'HNSW vector cosine search + earthdistance radius', icon: <Search className="w-4 h-4 text-cyan-400" /> },
    { label: 'Database Evidence Assembly', desc: 'Retrieving candidates from Railway PostgreSQL', icon: <Database className="w-4 h-4 text-amber-400" /> },
    { label: 'Multimodal Fusion & Scoring', desc: 'Weighted signal fusion, distance decay & safeguards', icon: <Shield className="w-4 h-4 text-emerald-400" /> },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-8 text-center space-y-6">
      <div className="flex flex-col items-center">
        <div className="relative w-16 h-16 rounded-full bg-indigo-950/60 border border-indigo-500/30 flex items-center justify-center mb-4">
          <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
        </div>
        <h3 className="text-lg font-bold text-slate-100">
          Verifying Outlet Submission
        </h3>
        <p className="text-xs text-slate-400 mt-1 max-w-sm">
          The production pipeline is executing multimodal embedding, candidate retrieval, and deterministic decision logic...
        </p>
      </div>

      <div className="max-w-md mx-auto space-y-2.5 text-left border-t border-slate-800 pt-5">
        {stages.map((stg, i) => (
          <div
            key={stg.label}
            className="flex items-start gap-3 p-2.5 rounded bg-slate-950/50 border border-slate-800/60 text-xs"
          >
            <div className="p-1 rounded bg-slate-900 border border-slate-800 mt-0.5">
              {stg.icon}
            </div>
            <div>
              <div className="font-semibold text-slate-200 flex items-center gap-2">
                <span>{stg.label}</span>
                <span className="text-[10px] text-slate-500 font-mono">Stage {i + 1}</span>
              </div>
              <p className="text-slate-400 text-[11px] mt-0.5">{stg.desc}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="text-[11px] text-slate-500 font-mono">
        Active backend processing • Please wait
      </div>
    </div>
  );
};
