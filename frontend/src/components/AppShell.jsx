import { useQuery } from '@tanstack/react-query'
import {
  ClipboardList,
  ClipboardPaste,
  LogOut,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  PlusCircle,
  Sparkles,
  TrendingUp,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '@/auth/authContext'
import { AgentChat } from '@/components/AgentChat'
import { api } from '@/lib/api'

/**
 * Three columns, each scrolling on its own: where you are, what you are looking
 * at, and who you can ask about it.
 *
 * The assistant used to be a drawer over the page. It is now a wall of the room
 * — always mounted, always visible on a desktop width — because almost
 * everything it does refers to what is on screen beside it ("mark this one
 * rejected", "compare these two"), and a panel you have to open first turns
 * that into two steps and a lost train of thought.
 *
 * The whole shell is `h-dvh overflow-hidden` and each column owns its own
 * scrollbar. That is what stops the page scrolling the assistant out of reach
 * when the applications table is long.
 */

// Below this the three columns do not fit, and the assistant becomes an overlay
// you summon. Matches Tailwind's `xl`.
const DOCK_QUERY = '(min-width: 1280px)'

// Destinations. Each carries a `count` reader so the rail can say what is
// behind a route rather than only naming it — the point of a tracker is the
// numbers, and burying them one click deep makes you go look.
const ROUTES = [
  {
    to: '/',
    label: 'Applications',
    icon: ClipboardList,
    end: true,
    count: (stats) => stats?.active,
    tone: 'accent',
    title: (n) => `${n} active`,
  },
  { to: '/insights', label: 'Insights', icon: TrendingUp },
  { to: '/resumes', label: 'Resume', icon: Sparkles },
]

const ACTIONS = [
  { to: '/applications/paste', label: 'Paste a job', icon: ClipboardPaste },
  { to: '/applications/new', label: 'Add by hand', icon: PlusCircle },
]

export function AppShell() {
  const { user, signOut } = useAuth()
  const location = useLocation()

  // Collapsing is a preference about the shape of the workspace, so it outlives
  // the tab. The assistant's overlay state is not — that is about right now.
  const [railOpen, setRailOpen] = useState(
    () => localStorage.getItem('rail:open') !== 'false',
  )
  /* The route the overlay was opened on, rather than a boolean. Navigating
     therefore closes it by arithmetic instead of by an effect that fires after
     the new page has already painted underneath it. */
  const [openedAt, setOpenedAt] = useState(null)
  const chatOpen = openedAt === location.pathname
  const setChatOpen = (open) => setOpenedAt(open ? location.pathname : null)

  const [docked, setDocked] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(DOCK_QUERY).matches,
  )

  useEffect(() => {
    const mq = window.matchMedia(DOCK_QUERY)
    const sync = (e) => setDocked(e.matches)
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape' && chatOpen) setOpenedAt(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [chatOpen])

  // Same query keys the routes use, so the rail reads the cache rather than
  // opening a second request for numbers the page already has.
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.getApplicationStats })
  const attention = useQuery({ queryKey: ['needs-attention'], queryFn: api.needsAttention })
  const attentionCount = attention.data?.length ?? 0

  function toggleRail() {
    setRailOpen((open) => {
      localStorage.setItem('rail:open', String(!open))
      return !open
    })
  }

  return (
    <div className="relative z-10 flex h-dvh overflow-hidden p-2 sm:p-3">
      <nav
        aria-label="Main"
        className={`glass-panel hidden shrink-0 flex-col rounded-2xl transition-[width] duration-300 ease-out sm:flex ${
          railOpen ? 'w-60' : 'w-[4.5rem]'
        }`}
      >
        <div className="flex items-center gap-2.5 px-4 py-4">
          <span
            aria-hidden="true"
            className="grid size-9 shrink-0 place-items-center rounded-xl bg-accent text-accent-ink"
          >
            <ClipboardList size={18} strokeWidth={2.25} />
          </span>
          {railOpen && (
            <span className="min-w-0 truncate font-display text-[15px] font-semibold tracking-tight">
              Job Tracker
            </span>
          )}
        </div>

        <div className="mx-3 h-px bg-border-subtle/70" />

        <div className="flex-1 overflow-y-auto overflow-x-hidden px-3 py-4">
          <Section label="Track" open={railOpen} />
          <ul className="space-y-1">
            {ROUTES.map((route) => (
              <li key={route.to}>
                <RailLink route={route} open={railOpen} count={route.count?.(stats.data)} />
              </li>
            ))}
          </ul>

          <Section label="Add" open={railOpen} className="mt-6" />
          <ul className="space-y-1">
            {ACTIONS.map((route) => (
              <li key={route.to}>
                <RailLink route={route} open={railOpen} />
              </li>
            ))}
          </ul>

          {/* The one warm thing in the rail, and it only exists when it is
              true. A badge that is always there — reading zero most days —
              stops being read at all. */}
          {attentionCount > 0 && (
            <NavLink
              to="/"
              title={`${attentionCount} need attention`}
              className={`mt-6 flex items-center gap-2.5 rounded-xl border border-signal/30 bg-signal/10 py-2 text-signal transition hover:bg-signal/15 ${
                railOpen ? 'px-3' : 'justify-center px-0'
              }`}
            >
              <span className="relative grid size-5 shrink-0 place-items-center">
                <span className="absolute size-2 rounded-full bg-signal" />
                <span className="stream-glow absolute size-4 rounded-full bg-signal/40" />
              </span>
              {railOpen && (
                <span className="min-w-0 truncate text-xs font-medium">
                  {attentionCount} need attention
                </span>
              )}
            </NavLink>
          )}
        </div>

        <div className="mx-3 h-px bg-border-subtle/70" />

        <div className={`flex items-center gap-2 p-3 ${railOpen ? '' : 'flex-col'}`}>
          <span
            aria-hidden="true"
            className="grid size-9 shrink-0 place-items-center rounded-full bg-surface-muted font-mono text-xs font-medium text-ink-muted"
          >
            {(user?.email ?? '?').slice(0, 2).toUpperCase()}
          </span>
          {railOpen && (
            <span className="min-w-0 flex-1 truncate text-xs text-ink-muted" title={user?.email}>
              {user?.email}
            </span>
          )}
          <button
            type="button"
            onClick={signOut}
            aria-label="Sign out"
            title="Sign out"
            className="grid size-9 shrink-0 cursor-pointer place-items-center rounded-lg text-ink-faint transition hover:bg-surface-muted hover:text-ink"
          >
            <LogOut size={16} aria-hidden="true" />
          </button>
        </div>

        <button
          type="button"
          onClick={toggleRail}
          aria-expanded={railOpen}
          aria-label={railOpen ? 'Collapse navigation' : 'Expand navigation'}
          className="mx-3 mb-3 flex cursor-pointer items-center justify-center gap-2 rounded-lg py-2 text-xs text-ink-faint transition hover:bg-surface-muted hover:text-ink"
        >
          {railOpen ? (
            <PanelLeftClose size={16} aria-hidden="true" />
          ) : (
            <PanelLeftOpen size={16} aria-hidden="true" />
          )}
          {railOpen && 'Collapse'}
        </button>
      </nav>

      {/* The narrow-screen rail. Same destinations, laid across the bottom,
          where a thumb can reach them. */}
      <nav
        aria-label="Main"
        className="glass-panel fixed inset-x-2 bottom-2 z-30 flex items-center justify-around rounded-2xl px-2 py-1.5 sm:hidden"
      >
        {[...ROUTES, ACTIONS[0]].map((route) => (
          <RailLink key={route.to} route={route} open={false} compact />
        ))}
      </nav>

      <div className="flex min-w-0 flex-1 flex-col">
        <main className="min-w-0 flex-1 overflow-y-auto px-4 pt-4 pb-24 sm:px-6 sm:pb-6 xl:px-8">
          <div className="mx-auto max-w-5xl">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Docked: a column. Undocked: a sheet over the page, summoned by the
          button below. The component itself is identical either way — only the
          box it is poured into changes. */}
      {docked ? (
        <aside className="ml-2 hidden w-[26rem] shrink-0 xl:block 2xl:w-[30rem]">
          <AgentChat />
        </aside>
      ) : (
        <>
          {chatOpen && (
            <>
              <button
                type="button"
                aria-label="Close assistant"
                onClick={() => setChatOpen(false)}
                className="fixed inset-0 z-30 cursor-pointer bg-canvas/70 backdrop-blur-sm"
              />
              <aside className="rise fixed inset-y-2 right-2 z-40 w-[min(28rem,calc(100vw-1rem))]">
                <AgentChat onClose={() => setChatOpen(false)} />
              </aside>
            </>
          )}

          {!chatOpen && (
            <button
              type="button"
              onClick={() => setChatOpen(true)}
              className="fixed right-4 bottom-20 z-30 flex cursor-pointer items-center gap-2 rounded-full bg-accent px-4 py-3 text-sm font-semibold text-accent-ink shadow-lg shadow-accent/20 transition hover:bg-accent-hover sm:bottom-4"
            >
              <MessageSquareText size={18} aria-hidden="true" />
              Assistant
            </button>
          )}
        </>
      )}
    </div>
  )
}

function Section({ label, open, className = '' }) {
  if (!open) return <div className={`h-3 ${className}`} />
  return (
    <p
      className={`mb-2 px-3 text-[10px] font-semibold tracking-[0.14em] text-ink-faint uppercase ${className}`}
    >
      {label}
    </p>
  )
}

/**
 * A rail entry. The active route is marked three ways — a spine, a tint and the
 * accent colour — because `aria-current` alone is invisible and colour alone
 * disappears for anyone who cannot see this particular green.
 */
function RailLink({ route, open, count, compact = false }) {
  const Icon = route.icon

  return (
    <NavLink
      to={route.to}
      end={route.end}
      title={open ? undefined : route.label}
      className={({ isActive }) =>
        `group relative flex items-center rounded-xl text-sm transition-colors duration-150 ${
          compact ? 'flex-col gap-0.5 px-3 py-1.5' : 'gap-3 py-2.5'
        } ${open ? 'px-3' : compact ? '' : 'justify-center px-0'} ${
          isActive
            ? 'bg-accent/12 font-medium text-accent'
            : 'text-ink-muted hover:bg-surface-muted/60 hover:text-ink'
        }`
      }
    >
      {({ isActive }) => (
        <>
          {isActive && !compact && (
            <span
              aria-hidden="true"
              className="absolute top-1/2 -left-3 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-accent shadow-[0_0_12px] shadow-accent/70"
            />
          )}
          <Icon size={compact ? 20 : 18} strokeWidth={isActive ? 2.25 : 1.75} aria-hidden="true" />
          {compact && <span className="text-[10px] leading-none">{route.label}</span>}
          {open && <span className="min-w-0 flex-1 truncate">{route.label}</span>}
          {open && count != null && (
            <span
              title={route.title?.(count)}
              className="shrink-0 rounded-full bg-surface-muted px-1.5 py-0.5 font-mono text-[11px] text-ink-muted tabular-nums"
            >
              {count}
            </span>
          )}
        </>
      )}
    </NavLink>
  )
}
