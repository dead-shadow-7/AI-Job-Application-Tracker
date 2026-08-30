import { useState } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '@/auth/authContext'
import { SetupNotice } from '@/components/SetupNotice'
import { supabase } from '@/lib/supabase'

export function Login() {
  const { session, isConfigured } = useAuth()
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState({ kind: 'idle' })

  if (!isConfigured) return <SetupNotice />
  if (session) return <Navigate to="/" replace />

  async function sendMagicLink(event) {
    event.preventDefault()
    setStatus({ kind: 'sending' })
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: window.location.origin },
    })
    setStatus(error ? { kind: 'error', message: error.message } : { kind: 'sent' })
  }

  async function signInWithGoogle() {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin },
    })
    if (error) setStatus({ kind: 'error', message: error.message })
  }

  return (
    <main className="flex min-h-dvh items-center justify-center px-4">
      <div className="w-full max-w-sm rounded-xl border border-border-subtle bg-surface p-8 shadow-sm">
        <h1 className="text-xl font-semibold tracking-tight">AI Job Tracker</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Sign in to track your applications.
        </p>

        {status.kind === 'sent' ? (
          <p className="mt-6 rounded-lg bg-surface-muted p-4 text-sm">
            Check <span className="font-medium">{email}</span> for a sign-in link.
          </p>
        ) : (
          <form onSubmit={sendMagicLink} className="mt-6 space-y-3">
            <label htmlFor="email" className="block text-sm font-medium">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
            />
            <button
              type="submit"
              disabled={status.kind === 'sending'}
              className="w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
            >
              {status.kind === 'sending' ? 'Sending…' : 'Send magic link'}
            </button>
          </form>
        )}

        <div className="my-5 flex items-center gap-3 text-xs text-ink-muted">
          <span className="h-px flex-1 bg-border-subtle" />
          or
          <span className="h-px flex-1 bg-border-subtle" />
        </div>

        <button
          type="button"
          onClick={signInWithGoogle}
          className="w-full rounded-lg border border-border-subtle px-3 py-2 text-sm font-medium transition hover:bg-surface-muted"
        >
          Continue with Google
        </button>

        {status.kind === 'error' ? (
          <p className="mt-4 text-sm text-red-600" role="alert">
            {status.message}
          </p>
        ) : null}
      </div>
    </main>
  )
}
