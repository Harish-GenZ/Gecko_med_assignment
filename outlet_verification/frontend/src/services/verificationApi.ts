import type { BackendHealth, VerificationResponse } from '../types/verification';

export const API_BASE_URL = (() => {
  const envUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (envUrl) return envUrl.replace(/\/+$/, '');
  if (typeof window !== 'undefined' && window.location.port !== '5173') {
    return window.location.origin;
  }
  return 'http://localhost:8000';
})();

export class ApiError extends Error {
  statusCode?: number;
  details?: unknown;

  constructor(message: string, statusCode?: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Checks backend and database health via GET /health.
 */
export async function checkBackendHealth(): Promise<BackendHealth> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      return { status: 'offline' };
    }

    const data = await res.json();
    return {
      status: data.status === 'ok' ? 'ok' : 'degraded',
      backend: data.backend,
      database: data.database,
    };
  } catch {
    clearTimeout(timeoutId);
    return { status: 'offline' };
  }
}

/**
 * Submits outlet data to the production POST /verify endpoint.
 * Sends multipart/form-data with name, latitude, longitude, and image file.
 */
export async function verifyOutlet(
  name: string,
  latitude: number,
  longitude: number,
  imageFile: File,
  autoRegister: boolean = true
): Promise<VerificationResponse> {
  const formData = new FormData();
  formData.append('name', name.trim());
  formData.append('latitude', latitude.toString());
  formData.append('longitude', longitude.toString());
  formData.append('image', imageFile);
  formData.append('auto_register_if_genuine', autoRegister.toString());

  try {
    const response = await fetch(`${API_BASE_URL}/verify`, {
      method: 'POST',
      body: formData,
      // Note: Do NOT set Content-Type header; browser must set boundary automatically
    });

    if (response.ok) {
      const data = (await response.json()) as VerificationResponse;
      return data;
    }

    // Handle error statuses
    let errorMessage = `Verification failed (HTTP ${response.status})`;
    let parsedDetails: unknown = null;

    try {
      const errorJson = await response.json();
      parsedDetails = errorJson;

      if (response.status === 422 && Array.isArray(errorJson.detail)) {
        // FastAPI Pydantic validation error format
        const errorDetails = errorJson.detail
          .map((err: { loc: (string | number)[]; msg: string }) => {
            const field = err.loc[err.loc.length - 1];
            return `${field}: ${err.msg}`;
          })
          .join(', ');
        errorMessage = `Input Validation Error: ${errorDetails}`;
      } else if (typeof errorJson.detail === 'string') {
        errorMessage = errorJson.detail;
      }
    } catch {
      // Non-JSON response
      const rawText = await response.text().catch(() => '');
      if (rawText) {
        errorMessage = `Server Error: ${rawText.slice(0, 100)}`;
      }
    }

    throw new ApiError(errorMessage, response.status, parsedDetails);
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }

    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new ApiError(
        `Unable to reach verification backend at ${API_BASE_URL}. Ensure the FastAPI server is running and CORS is enabled.`,
        0
      );
    }

    throw new ApiError(
      err instanceof Error ? err.message : 'Unknown communication error occurred with the verification service.'
    );
  }
}

/**
 * Registers an approved outlet directly into PostgreSQL and S3 Object Storage.
 */
export async function registerOutletDirect(
  name: string,
  latitude: number,
  longitude: number,
  imageFile: File
): Promise<{ id: string; name: string; latitude: number; longitude: number; image_url: string; message: string }> {
  const formData = new FormData();
  formData.append('name', name.trim());
  formData.append('latitude', latitude.toString());
  formData.append('longitude', longitude.toString());
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/outlets/register`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to register outlet' }));
    throw new ApiError(err.detail || 'Failed to register outlet', response.status);
  }

  return response.json();
}
