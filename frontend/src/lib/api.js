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

  listSkills: (params = {}) => apiFetch(`/api/v1/skills${query(params)}`),
  listCompanies: (params = {}) => apiFetch(`/api/v1/companies${query(params)}`),
  updateJob: (id, body) => apiFetch(`/api/v1/jobs/${id}`, { method: 'PATCH', body }),
}
