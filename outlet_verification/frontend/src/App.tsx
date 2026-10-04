import { useEffect, useRef, useState } from 'react';
import {
  Shield,
  Activity,
  Terminal,
  AlertCircle,
  Clock,
  MapPin,
  FileText,
  Server,
  RefreshCw,
} from 'lucide-react';
import { OutletForm } from './components/OutletForm';
import { VerificationResult } from './components/VerificationResult';
import { LoadingState } from './components/LoadingState';
import {
  API_BASE_URL,
  checkBackendHealth,
  registerOutletDirect,
  verifyOutlet,
} from './services/verificationApi';
import type { BackendHealth, VerificationResponse } from './types/verification';

export function App() {
  const [health, setHealth] = useState<BackendHealth>({ status: 'ok' });
  const [isHealthChecking, setIsHealthChecking] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const isLoadingRef = useRef(false);
  isLoadingRef.current = isLoading;

  const [isRegistering, setIsRegistering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verificationResult, setVerificationResult] = useState<VerificationResponse | null>(null);
  const [lastSubmissionFile, setLastSubmissionFile] = useState<File | null>(null);

  // Metadata for debug / audit panel (Step 13)
  const [lastSubmissionMeta, setLastSubmissionMeta] = useState<{
    name: string;
    lat: number;
    lng: number;
    filename: string;
    filesizeKb: number;
    timestamp: string;
    previewUrl: string | null;
  } | null>(null);

  const refreshHealth = async () => {
    setIsHealthChecking(true);
    try {
      const res = await checkBackendHealth();
      setHealth(res);
    } catch {
      setHealth({ status: 'offline' });
    } finally {
      setIsHealthChecking(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    checkBackendHealth().then((res) => {
      if (isMounted) setHealth(res);
    });

    const interval = setInterval(async () => {
      if (isLoadingRef.current) return;
      const res = await checkBackendHealth();
      if (isMounted) setHealth(res);
    }, 20000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleFormSubmit = async (data: {
    name: string;
    latitude: number;
    longitude: number;
    imageFile: File;
    autoRegister: boolean;
  }) => {
    setIsLoading(true);
    setError(null);
    setLastSubmissionFile(data.imageFile);

    const previewUrl = URL.createObjectURL(data.imageFile);
    setLastSubmissionMeta({
      name: data.name,
      lat: data.latitude,
      lng: data.longitude,
      filename: data.imageFile.name,
      filesizeKb: Math.round(data.imageFile.size / 1024),
      timestamp: new Date().toLocaleTimeString(),
      previewUrl,
    });

    try {
      const result = await verifyOutlet(
        data.name,
        data.latitude,
        data.longitude,
        data.imageFile,
        data.autoRegister
      );
      setVerificationResult(result);
      setHealth((prev) => ({ ...prev, status: 'ok' }));
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during verification.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleApproveAndRegister = async () => {
    if (!lastSubmissionMeta || !lastSubmissionFile) return;
    setIsRegistering(true);
    setError(null);
    try {
      const reg = await registerOutletDirect(
        lastSubmissionMeta.name,
        lastSubmissionMeta.lat,
        lastSubmissionMeta.lng,
        lastSubmissionFile
      );
      if (verificationResult) {
        setVerificationResult({
          ...verificationResult,
          registered: true,
          registered_outlet_id: reg.id,
          stored_image_url: reg.image_url,
        });
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to register outlet');
    } finally {
      setIsRegistering(false);
    }
  };

  const handleReset = () => {
    setVerificationResult(null);
    setLastSubmissionFile(null);
    setError(null);
    if (lastSubmissionMeta?.previewUrl) {
      URL.revokeObjectURL(lastSubmissionMeta.previewUrl);
    }
    setLastSubmissionMeta(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Engineering Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-20 px-4 sm:px-8 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-600/30">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-slate-100 tracking-tight m-0">
                  OUTLET VERIFICATION
                </h1>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                  v0.1.0-prod
                </span>
              </div>
              <p className="text-xs text-slate-400 m-0">
                Verify whether a submitted outlet matches an existing outlet.
              </p>
            </div>
          </div>

          {/* System & Health Status Indicator */}
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950 border border-slate-800">
              <span
                className={`w-2 h-2 rounded-full ${
                  health.status === 'ok'
                    ? 'bg-emerald-400 animate-pulse'
                    : health.status === 'degraded'
                    ? 'bg-amber-400'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300 font-medium">
                Backend:{' '}
                {health.status === 'ok'
                  ? 'Connected'
                  : health.status === 'degraded'
                  ? 'Degraded'
                  : 'Offline'}
              </span>
              <button
                type="button"
                onClick={refreshHealth}
                title="Refresh backend status"
                className="text-slate-500 hover:text-slate-300 ml-1"
              >
                <RefreshCw
                  className={`w-3 h-3 ${isHealthChecking ? 'animate-spin' : ''}`}
                />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Two-Column Desktop / Stacked Mobile Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Input Form (5 cols on lg) */}
        <section className="lg:col-span-5 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
            <div className="border-b border-slate-800 pb-3 mb-4">
              <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider flex items-center gap-2 m-0">
                <FileText className="w-4 h-4 text-indigo-400" />
                <span>Outlet Submission Details</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5 m-0">
                Upload a physical storefront photograph and GPS coordinates.
              </p>
            </div>

            <OutletForm onSubmit={handleFormSubmit} isLoading={isLoading} />
          </div>

          {/* Connection Trouble Notice */}
          {health.status === 'offline' && !isLoading && (
            <div className="bg-rose-950/40 border border-rose-900/60 rounded-lg p-3 text-xs text-rose-300 flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-200">Backend Unavailable</p>
                <p className="mt-0.5 text-rose-300/90 leading-relaxed">
                  FastAPI service at <code className="bg-rose-900/40 px-1 rounded">{API_BASE_URL}</code> is unreachable.
                  Please ensure the Python backend is running via <code className="bg-rose-900/40 px-1 rounded">uvicorn app.main:app</code>.
                </p>
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Execution & Results (7 cols on lg) */}
        <section className="lg:col-span-7 space-y-4">
          {/* Error Banner */}
          {error && (
            <div className="bg-rose-950/60 border border-rose-800/80 rounded-lg p-4 flex items-start justify-between gap-3 text-xs text-rose-200">
              <div className="flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-rose-100">Verification Failed</h4>
                  <p className="mt-1 leading-relaxed">{error}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setError(null)}
                className="px-2.5 py-1 rounded bg-rose-900/60 hover:bg-rose-900 text-rose-200 font-medium text-xs flex-shrink-0"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Active View State */}
          {isLoading ? (
            <LoadingState />
          ) : verificationResult ? (
            <VerificationResult
              response={verificationResult}
              submittedImagePreview={lastSubmissionMeta?.previewUrl || null}
              onReset={handleReset}
              onApproveAndRegister={handleApproveAndRegister}
              isRegistering={isRegistering}
            />
          ) : (
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-lg p-8 text-center text-slate-400 flex flex-col items-center justify-center min-h-[380px]">
              <div className="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700/60 flex items-center justify-center text-slate-400 mb-3">
                <Activity className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200">
                Ready for Verification
              </h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1 leading-relaxed">
                Provide an outlet name, storefront photograph, and latitude/longitude coordinates on the left panel, then submit to execute the verification pipeline.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2 text-[11px] text-slate-500 font-mono">
                <span className="px-2 py-1 rounded bg-slate-950 border border-slate-800">
                  SentenceTransformers MiniLM-L6
                </span>
                <span className="px-2 py-1 rounded bg-slate-950 border border-slate-800">
                  OpenCLIP ViT-B/32
                </span>
                <span className="px-2 py-1 rounded bg-slate-950 border border-slate-800">
                  HNSW Cosine Vector Search
                </span>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Manual Testing & Audit Footer (Step 13) */}
      <footer className="border-t border-slate-800 bg-slate-950 text-slate-500 text-xs px-4 sm:px-8 py-4 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 font-mono">
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-1.5 text-slate-400">
              <Server className="w-3.5 h-3.5 text-indigo-400" />
              API: {API_BASE_URL}
            </span>
            <span>•</span>
            <span className="inline-flex items-center gap-1.5 text-slate-400">
              <Terminal className="w-3.5 h-3.5 text-emerald-400" />
              POST /verify
            </span>
            {lastSubmissionMeta && (
              <>
                <span>•</span>
                <span className="inline-flex items-center gap-1 text-slate-400">
                  <Clock className="w-3 h-3" />
                  {lastSubmissionMeta.timestamp}
                </span>
                <span>•</span>
                <span className="inline-flex items-center gap-1 text-slate-400 truncate max-w-[200px]">
                  <MapPin className="w-3 h-3" />
                  {lastSubmissionMeta.lat}, {lastSubmissionMeta.lng}
                </span>
              </>
            )}
          </div>

          <div className="text-[11px] text-slate-600">
            Outlet Verification Engine • Phase 6 Frontend Diagnostic Suite
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
