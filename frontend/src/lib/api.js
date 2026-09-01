import { getAccessToken, supabase } from './supabase'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

/**
 * Fetch wrapper that attaches the Supabase access token.
 *
 * A 401 means the token is gone or no longer valid — the local session is
 * cleared so the router falls back to the login screen rather than leaving the
 * UI in a signed-in state that every subsequent request rejects.
 */
export async function apiFetch(path, { method = 'GET', body, signal } = {}) {
  const token = await getAccessToken()

  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    signal,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  })

  if (response.status === 401) {
    await supabase?.auth.signOut()
    throw new ApiError('Session expired. Please sign in again.', 401, null)
  }

  const payload = response.status === 204 ? null : await response.json().catch(() => null)

  if (!response.ok) {
    const detail = payload?.detail
    // FastAPI validation errors arrive as a list of objects; flatten so the UI
    // never renders "[object Object]".
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join(', ')
      : (detail ?? `Request failed with status ${response.status}`)
    throw new ApiError(message, response.status, payload)
  }

  return payload
}

/**
 * File upload.
 *
 * Separate from apiFetch because a multipart body must NOT carry an explicit
 * Content-Type — the browser has to set it itself so it can append the
 * boundary token. Setting `application/json` here, as apiFetch does, makes the
 * server reject the body as malformed.
 */
async function uploadResume(file, label) {
  const token = await getAccessToken()
  const form = new FormData()
  form.append('file', file)
  if (label) form.append('label', label)

  const response = await fetch(`${BASE_URL}/api/v1/resumes`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
  })

  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.detail
    throw new ApiError(
      Array.isArray(detail) ? detail.map((d) => d.msg).join(', ') : (detail ?? 'Upload failed'),
      response.status,
      payload,
    )
  }
  return payload
}

/** Drops empty values so `?status=&search=` never reaches the API. */
function query(params) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '' || value === false) continue
    if (Array.isArray(value)) value.forEach((v) => search.append(key, v))
    else search.append(key, String(value))
  }
  const qs = search.toString()
  return qs ? `?${qs}` : ''
}

export const api = {
  getMe: () => apiFetch('/api/v1/me'),
  updateMe: (updates) => apiFetch('/api/v1/me', { method: 'PATCH', body: updates }),
  health: () => apiFetch('/health'),

  listApplications: (params = {}) => apiFetch(`/api/v1/applications${query(params)}`),
  getApplicationStats: () => apiFetch('/api/v1/applications/stats'),
  getApplication: (id) => apiFetch(`/api/v1/applications/${id}`),
  createApplication: (body) => apiFetch('/api/v1/applications', { method: 'POST', body }),
  updateApplication: (id, body) =>
    apiFetch(`/api/v1/applications/${id}`, { method: 'PATCH', body }),
  deleteApplication: (id) => apiFetch(`/api/v1/applications/${id}`, { method: 'DELETE' }),

  // Returns the whole application: appending is the only way status moves, so
  // the caller always needs the recomputed status back.
  addEvent: (id, body) => apiFetch(`/api/v1/applications/${id}/events`, { method: 'POST', body }),
  addStage: (id, body) => apiFetch(`/api/v1/applications/${id}/stages`, { method: 'POST', body }),
  updateStage: (id, stageId, body) =>
    apiFetch(`/api/v1/applications/${id}/stages/${stageId}`, { method: 'PATCH', body }),

  // Extracts and returns a preview. Writes nothing — the user reviews the
  // result and posts it to createApplication.
  ingestJob: (body) => apiFetch('/api/v1/jobs/ingest', { method: 'POST', body }),

  listResumes: () => apiFetch('/api/v1/resumes'),
  // Multipart, so this bypasses apiFetch's JSON handling — see uploadResume.
  uploadResumeFile: (file, label) => uploadResume(file, label),
  uploadResumeText: (body) => apiFetch('/api/v1/resumes/text', { method: 'POST', body }),
  setDefaultResume: (id) => apiFetch(`/api/v1/resumes/${id}/default`, { method: 'POST' }),
  deleteResume: (id) => apiFetch(`/api/v1/resumes/${id}`, { method: 'DELETE' }),

  getMatch: (applicationId) => apiFetch(`/api/v1/applications/${applicationId}/match`),
  computeMatch: (applicationId) =>
    apiFetch(`/api/v1/applications/${applicationId}/match`, { method: 'POST' }),

  // Applications a follow-up rule has fired on, with the rule that fired.
  needsAttention: () => apiFetch('/api/v1/needs-attention'),
  closeGhosted: () => apiFetch('/api/v1/needs-attention/close-ghosted', { method: 'POST' }),
  listRules: () => apiFetch('/api/v1/follow-up-rules'),
  updateRule: (id, body) =>
    apiFetch(`/api/v1/follow-up-rules/${id}`, { method: 'PATCH', body }),

  // chat NEVER writes — it returns a proposal. confirm performs the write, and
  // the body is `{kind, ...proposal.payload}`: the server picks the schema to
  // validate against from `kind`, so the client never builds it field by field.
  agentChat: (message) => apiFetch('/api/v1/agent/chat', { method: 'POST', body: { message } }),
  agentConfirm: (body) => apiFetch('/api/v1/agent/confirm', { method: 'POST', body }),

  // Ranked by meaning, not keyword — so it answers "the RAG roles" for a
  // posting that never uses the word. Unrelated to listApplications' `search`,
  // which is an ILIKE filter over title and company.
  searchByMeaning: (q, limit) => apiFetch(`/api/v1/search${query({ q, limit })}`),
  getAnalytics: () => apiFetch('/api/v1/analytics'),

  listSkills: (params = {}) => apiFetch(`/api/v1/skills${query(params)}`),
  listCompanies: (params = {}) => apiFetch(`/api/v1/companies${query(params)}`),
  updateJob: (id, body) => apiFetch(`/api/v1/jobs/${id}`, { method: 'PATCH', body }),
}
