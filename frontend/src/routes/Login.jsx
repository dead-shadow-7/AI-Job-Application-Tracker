import { ClipboardList, MailCheck, TriangleAlert } from 'lucide-react'
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
    <main className="relative z-10 flex min-h-dvh items-center justify-center px-4 py-10">
      <div className="rise glass w-full max-w-sm rounded-2xl p-8">
        <span
          aria-hidden="true"
          className="grid size-11 place-items-center rounded-2xl bg-accent text-accent-ink"
        >
          <ClipboardList size={21} strokeWidth={2.25} />
        </span>
        <h1 className="mt-5 font-display text-2xl font-semibold tracking-tight">AI Job Tracker</h1>
        <p className="mt-1.5 text-sm text-ink-muted">
          Every application, what it is waiting on, and an assistant that has read all of it.
        </p>

        {status.kind === 'sent' ? (
          <p className="glass mt-6 flex gap-2.5 rounded-xl p-4 text-sm text-ink-muted">
            <MailCheck size={17} aria-hidden="true" className="mt-0.5 shrink-0 text-accent" />
            <span>
              Check <span className="font-medium text-ink">{email}</span> for a sign-in link.
            </span>
          </p>
        ) : (
          <form onSubmit={sendMagicLink} className="mt-6 space-y-2.5">
            <label htmlFor="email" className="block text-xs font-medium text-ink-muted">
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
              className="well w-full rounded-xl px-3.5 py-2.5 text-sm outline-none transition placeholder:text-ink-faint focus:border-accent/40 focus:shadow-[0_0_0_3px] focus:shadow-accent/12 focus-visible:outline-none"
            />
            <button
              type="submit"
              disabled={status.kind === 'sending'}
              className="w-full cursor-pointer rounded-xl bg-accent px-3 py-2.5 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
            >
              {status.kind === 'sending' ? 'Sending…' : 'Send magic link'}
            </button>
          </form>
        )}

        <div className="my-5 flex items-center gap-3 text-xs text-ink-faint">
          <span className="h-px flex-1 bg-border-subtle" />
          or
          <span className="h-px flex-1 bg-border-subtle" />
        </div>

        <button
          type="button"
          onClick={signInWithGoogle}
          className="flex w-full cursor-pointer items-center justify-center gap-2.5 rounded-xl border border-border-subtle px-3 py-2.5 text-sm font-medium text-ink-muted transition hover:border-border-strong hover:text-ink"
        >
          <GoogleMark />
          Continue with Google
        </button>

        {status.kind === 'error' ? (
          <p className="mt-4 flex gap-2 text-sm text-danger" role="alert">
            <TriangleAlert size={15} aria-hidden="true" className="mt-0.5 shrink-0" />
            {status.message}
          </p>
        ) : null}
      </div>
    </main>
  )
}

/* Google's mark, inline. It is the one icon in the app that is not from the
   icon set, because a brand mark drawn in someone else's line weight is the
   wrong mark. */
function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-4 shrink-0" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M23.06 12.25c0-.85-.08-1.67-.22-2.45H12v4.63h6.2a5.3 5.3 0 0 1-2.3 3.48v2.89h3.72c2.18-2 3.44-4.96 3.44-8.55Z"
      />
      <path
        fill="#34A853"
        d="M12 23.5c3.11 0 5.72-1.03 7.62-2.8l-3.72-2.89c-1.03.69-2.35 1.1-3.9 1.1-3 0-5.54-2.03-6.45-4.75H1.71v2.98A11.5 11.5 0 0 0 12 23.5Z"
      />
      <path
        fill="#FBBC05"
        d="M5.55 14.16a6.9 6.9 0 0 1 0-4.4V6.78H1.71a11.5 11.5 0 0 0 0 10.36l3.84-2.98Z"
      />
      <path
        fill="#EA4335"
        d="M12 5.02c1.69 0 3.21.58 4.4 1.72l3.3-3.3C17.72 1.57 15.11.5 12 .5A11.5 11.5 0 0 0 1.71 6.78l3.84 2.98C6.46 7.04 9 5.02 12 5.02Z"
      />
    </svg>
  )
}
