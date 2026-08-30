import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

/**
 * Whether Supabase credentials are present.
 *
 * Checked explicitly so a fresh clone renders a setup screen instead of
 * `createClient` throwing on an undefined URL — the error that surfaces
 * otherwise points at node_modules and says nothing about the missing .env.
 */
export const isSupabaseConfigured = Boolean(url && anonKey)

export const supabase = isSupabaseConfigured
  ? createClient(url, anonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    })
  : null

/** Current access token, or null when signed out. */
export async function getAccessToken() {
  if (!supabase) return null
  // getSession() serves from local storage and refreshes when near expiry, so
  // this is cheap enough to call per request.
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}
