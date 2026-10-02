// API service layer

import type { JobCreateResponse, JobStatusResponse } from '../types'

// Use VITE_API_URL if set (e.g. for separate backend service on Railway), otherwise use relative /api
const API_HOST = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')
const BASE = `${API_HOST}/api`

export async function uploadDocument(file: File, signal?: AbortSignal): Promise<JobCreateResponse> {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    body:   form,
    signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(err.detail ?? 'Upload failed')
  }

  return res.json()
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${BASE}/job/${jobId}`)

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Not found' }))
    throw new Error(err.detail ?? 'Failed to fetch job status')
  }

  return res.json()
}

export async function cancelJob(jobId: string): Promise<{ success: boolean; message: string }> {
  try {
    const res = await fetch(`${BASE}/job/${jobId}/cancel`, {
      method: 'POST',
    })
    return await res.json()
  } catch {
    return { success: false, message: 'Failed to notify server of cancellation.' }
  }
}
