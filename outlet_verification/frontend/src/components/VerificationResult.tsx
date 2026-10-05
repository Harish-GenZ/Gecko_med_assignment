import React, { useState } from 'react';
import {
  AlertOctagon,
  CheckCircle,
  HelpCircle,
  RefreshCw,
  Info,
  ShieldAlert,
  Database,
  ExternalLink,
  Copy,
  Check,
  UserCheck,
} from 'lucide-react';
import type { VerificationResponse } from '../types/verification';
import { EvidencePanel } from './EvidencePanel';
import { CandidateCard } from './CandidateCard';
import { CandidateList } from './CandidateList';

interface VerificationResultProps {
  response: VerificationResponse;
  submittedImagePreview: string | null;
  onReset: () => void;
  onApproveAndRegister?: () => void;
  isRegistering?: boolean;
}

export const VerificationResult: React.FC<VerificationResultProps> = ({
  response,
  submittedImagePreview,
  onReset,
  onApproveAndRegister,
  isRegistering = false,
}) => {
  const [copied, setCopied] = useState(false);
  const { decision, duplicate_confidence, reason_summary, matched_outlet, evidence } = response;

  const getDecisionBadge = () => {
    switch (decision) {
      case 'DUPLICATE':
        return {
          title: 'DUPLICATE OUTLET DETECTED',
          subtitle: 'Matches existing outlet record in database',
          icon: <AlertOctagon className="w-6 h-6 text-rose-400" />,
          borderColor: 'border-rose-500/50',
          bgGradient: 'from-rose-950/40 via-slate-900 to-slate-900',
          badgeColor: 'bg-rose-500 text-white',
          textColor: 'text-rose-400',
        };
      case 'GENUINE':
        return {
          title: 'GENUINE NEW OUTLET',
          subtitle: 'Distinct physical and visual entity — approved for onboarding',
          icon: <CheckCircle className="w-6 h-6 text-emerald-400" />,
          borderColor: 'border-emerald-500/50',
          bgGradient: 'from-emerald-950/40 via-slate-900 to-slate-900',
          badgeColor: 'bg-emerald-500 text-white',
          textColor: 'text-emerald-400',
        };
      case 'REJECTED': {
        const isNameMismatch =
          response.reason_codes.includes('NAME_MISMATCH') ||
          response.reason_codes.includes('SIGNBOARD_MISMATCH');
        return {
          title: isNameMismatch ? 'REJECTED: NAME MISMATCH' : 'REJECTED: FAKE / NON-STORE OUTLET',
          subtitle: isNameMismatch
            ? 'Signboard text extracted from photo contradicts the submitted outlet name'
            : 'Input does not qualify as an authentic commercial store or supermarket',
          icon: <ShieldAlert className="w-6 h-6 text-rose-500" />,
          borderColor: 'border-rose-600/70',
          bgGradient: 'from-rose-950/70 via-slate-900 to-slate-900',
          badgeColor: 'bg-rose-600 text-white font-bold',
          textColor: 'text-rose-400',
        };
      }
      case 'NEEDS_REVIEW':
      default:
        return {
          title: 'MANUAL REVIEW REQUIRED',
          subtitle: 'Available evidence is inconclusive, borderline, or conflicting',
          icon: <HelpCircle className="w-6 h-6 text-amber-400" />,
          borderColor: 'border-amber-500/50',
          bgGradient: 'from-amber-950/40 via-slate-900 to-slate-900',
          badgeColor: 'bg-amber-500 text-slate-950',
          textColor: 'text-amber-400',
        };
    }
  };

  const badge = getDecisionBadge();

  return (
    <div className="space-y-6">
      {/* Top Banner Decision Card */}
      <div
        className={`border rounded-lg bg-gradient-to-br ${badge.bgGradient} ${badge.borderColor} p-6 shadow-xl relative overflow-hidden`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex-shrink-0">
              {badge.icon}
            </div>

            <div>
              <div className="flex items-center gap-2.5">
                <span
                  className={`text-xs font-bold tracking-wider uppercase px-2.5 py-0.5 rounded-full ${badge.badgeColor}`}
                >
                  {decision}
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Engine Decision
                </span>
              </div>

              <h2 className="text-xl font-bold text-slate-100 mt-1">
                {badge.title}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">{badge.subtitle}</p>
            </div>
          </div>

          {/* Duplicate Confidence / 3-Metric Average Callout */}
          <div className="text-left sm:text-right border-t sm:border-t-0 sm:border-l border-slate-800/80 pt-3 sm:pt-0 sm:pl-6 flex-shrink-0">
            <span className="text-xs text-slate-400 block font-medium">
              3-Metric Average
            </span>
            <span className={`text-3xl font-extrabold font-mono ${badge.textColor}`}>
              {(duplicate_confidence * 100).toFixed(1)}%
            </span>
            <span className="text-[11px] text-slate-500 block">
              {duplicate_confidence >= 0.75
                ? '>= 75.0% (Duplicate)'
                : duplicate_confidence >= 0.65
                ? '65.0% - 75.0% (Manual Review)'
                : '< 65.0% (Genuine)'}
            </span>
          </div>
        </div>

        {/* Reason Summary from Backend */}
        <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-start gap-2 text-xs text-slate-300">
          <Info className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <span className="font-semibold text-slate-200">System Explanation: </span>
            {reason_summary}
          </p>
        </div>

        {decision === 'NEEDS_REVIEW' && (
          <div className="mt-4 bg-amber-950/40 border border-amber-800/60 rounded-lg p-3.5 space-y-3">
            <div className="flex items-center gap-2 text-xs text-amber-200">
              <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <span>
                This submission has been diverted to human review. If you verify that this outlet is genuine and distinct, you can onboard it directly:
              </span>
            </div>
            {onApproveAndRegister && (
              <div className="flex items-center justify-end pt-1">
                <button
                  type="button"
                  onClick={onApproveAndRegister}
                  disabled={isRegistering}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-md shadow-emerald-950/40 transition-colors disabled:opacity-50 cursor-pointer"
                >
                  <UserCheck className="w-4 h-4" />
                  <span>{isRegistering ? 'Registering to Database...' : 'Approve & Register Genuine Outlet'}</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* Database & Cloud Bucket Registration Banner */}
        {response.registered && (
          <div className="mt-4 bg-emerald-950/40 border border-emerald-500/40 rounded-lg p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-emerald-300 font-semibold text-xs tracking-wider uppercase">
                <Database className="w-4 h-4 text-emerald-400" />
                <span>Saved to PostgreSQL & Railway Bucket</span>
              </div>
              <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/30 font-medium">
                Live In Database
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
              {response.registered_outlet_id && (
                <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800 flex items-center justify-between">
                  <div className="truncate mr-2">
                    <span className="text-slate-400 block text-[10px] uppercase font-mono">PostgreSQL Outlet ID</span>
                    <span className="text-slate-200 font-mono text-xs">{response.registered_outlet_id}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      if (response.registered_outlet_id) {
                        navigator.clipboard.writeText(response.registered_outlet_id);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }
                    }}
                    className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors flex-shrink-0"
                    title="Copy UUID"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              )}

              {response.stored_image_url && (
                <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800 flex items-center justify-between">
                  <div className="truncate mr-2">
                    <span className="text-slate-400 block text-[10px] uppercase font-mono">Bucket Storage Image</span>
                    <span className="text-emerald-400 font-mono text-xs truncate block">{response.stored_image_url}</span>
                  </div>
                  <a
                    href={response.stored_image_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors flex-shrink-0"
                    title="Open in new tab"
                  >
                    <ExternalLink className="w-3.5 h-3.5 text-indigo-400" />
                  </a>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Domain Authenticity Inspection (Zero-Shot CLIP + OCR + Name Validation) */}
        {response.authenticity && (
          <div className="mt-4 bg-slate-900/90 border border-slate-800 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-300">
                <ShieldAlert className="w-4 h-4 text-cyan-400" />
                <span>Storefront Domain Authenticity</span>
              </div>
              <span
                className={`text-[10px] font-bold px-2.5 py-0.5 rounded uppercase ${
                  response.authenticity.is_authentic_store
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                }`}
              >
                {response.authenticity.status}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-mono block">1. Visual Domain (CLIP)</span>
                <span className="font-semibold text-slate-200 block truncate" title={response.authenticity.visual_category}>
                  {response.authenticity.visual_is_store ? 'Retail Storefront' : response.authenticity.visual_category}
                </span>
                <span className="text-[11px] text-slate-500">
                  Storefront match: {(response.authenticity.visual_probability * 100).toFixed(1)}%
                </span>
              </div>

              <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-mono block">2. Signboard OCR</span>
                <span className="font-semibold text-slate-200 block">
                  {response.authenticity.ocr_lines.length > 0
                    ? `${response.authenticity.ocr_lines.length} lines detected`
                    : response.authenticity.is_unclear_or_multilingual
                    ? (response.authenticity.ocr_clarity_status === 'MULTILINGUAL_OR_STYLIZED' ? 'Multilingual / Regional Script' : 'Unclear / Blurry Photo')
                    : 'No signboard text'}
                </span>
                {response.authenticity.ocr_name_match_score !== null && response.authenticity.ocr_name_match_score !== undefined && (
                  <span
                    className={`text-[11px] block font-mono font-medium ${
                      response.authenticity.is_cross_lingual_match || response.authenticity.is_incidental_signage
                        ? 'text-emerald-400'
                        : response.authenticity.ocr_name_match_score < 0.3
                        ? 'text-rose-400 font-bold'
                        : 'text-emerald-400'
                    }`}
                  >
                    Name match: {(response.authenticity.ocr_name_match_score * 100).toFixed(0)}%
                    {response.authenticity.is_cross_lingual_match && !response.authenticity.is_incidental_signage && ' (Semantic/Tamil Match)'}
                    {response.authenticity.is_incidental_signage && ' (Incidental Signage)'}
                    {response.authenticity.ocr_name_match_score < 0.3 && !response.authenticity.is_cross_lingual_match && !response.authenticity.is_incidental_signage && ' (Mismatch)'}
                  </span>
                )}
                {response.authenticity.ocr_lines.length > 0 && (
                  <span className="text-[11px] text-cyan-400 block truncate" title={response.authenticity.ocr_lines.join(', ')}>
                    {response.authenticity.ocr_lines.join(', ')}
                  </span>
                )}
                {response.authenticity.is_unclear_or_multilingual && response.authenticity.ocr_lines.length === 0 && (
                  <span className="text-[11px] text-amber-400/90 block">
                    {response.authenticity.ocr_clarity_status === 'MULTILINGUAL_OR_STYLIZED'
                      ? 'Signboard is in regional script or stylized font'
                      : 'Photo is blurry or unreadable'}
                  </span>
                )}
              </div>

              <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-mono block">3. Name Classification</span>
                <span className="font-semibold text-slate-200 block">
                  {response.authenticity.name_category}
                </span>
                <span className="text-[11px] text-slate-500">
                  {response.authenticity.name_is_valid_outlet ? 'Valid Outlet Name' : 'Flagged Non-Store'}
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-400 italic">
              {response.authenticity.summary}
            </p>
          </div>
        )}
      </div>

      {/* Multimodal Evidence Panel */}
      <EvidencePanel response={response} />

      {/* Matched Outlet / Top Candidate Inspection */}
      {matched_outlet && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Matched Reference Outlet Profile
            </h3>
            <span className="text-xs text-slate-500 font-mono">
              Database Match
            </span>
          </div>
          <CandidateCard
            outlet={matched_outlet}
            evidence={evidence}
            submittedImagePreview={submittedImagePreview}
            confidence={duplicate_confidence}
            isMatched={decision === 'DUPLICATE'}
          />
        </div>
      )}

      {/* Candidates List if provided */}
      {response.candidates && response.candidates.length > 0 && (
        <CandidateList candidates={response.candidates} evidence={evidence} />
      )}

      {/* Action Bar (Reset / Verify Another) */}
      <div className="pt-2 flex justify-center">
        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-sm font-semibold transition-colors cursor-pointer"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Verify Another Outlet</span>
        </button>
      </div>
    </div>
  );
};
