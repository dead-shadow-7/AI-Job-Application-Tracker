import { useState } from 'react'
import { Link, Outlet } from 'react-router-dom'
import { useAuth } from '@/auth/authContext'
import { AgentChat } from '@/components/AgentChat'

export function AppShell() {
  const { user, signOut } = useAuth()
  const [chatOpen, setChatOpen] = useState(false)

  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-20 border-b border-border-subtle bg-surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
          <Link to="/" className="text-sm font-semibold tracking-tight">
            AI Job Tracker
          </Link>
          <div className="flex items-center gap-3">
            <Link
              to="/resumes"
              className="text-sm text-ink-muted transition hover:text-accent"
            >
              Resume
            </Link>
            <button
              type="button"
              onClick={() => setChatOpen((open) => !open)}
              aria-expanded={chatOpen}
              className="rounded-lg border border-border-subtle px-3 py-1.5 text-sm transition hover:bg-surface-muted"
            >
              Assistant
            </button>
            <span className="hidden text-xs text-ink-muted sm:inline">{user?.email}</span>
            <button
              type="button"
              onClick={signOut}
              className="rounded-lg border border-border-subtle px-3 py-1.5 text-sm transition hover:bg-surface-muted"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>

      <AgentChat open={chatOpen} onClose={() => setChatOpen(false)} />
    </div>
  )
}
