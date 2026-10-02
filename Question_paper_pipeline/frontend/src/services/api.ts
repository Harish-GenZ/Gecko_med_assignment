// API service layer

import type { JobCreateResponse, JobStatusResponse } from '../types'

const BASE = '/api'

export async function uploadDocument(file: File): Promise<JobCreateResponse> {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    body:   form,
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
