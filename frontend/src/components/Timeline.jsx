import { EVENT_LABELS, formatDayMonth, relativeDays } from '@/lib/format'

/* The vertical rail from the original sketch:

     Aug 18   Applied
        │
     Aug 19   HR Screening
        │
     Aug 20   Waiting

   Events that move the application are filled; contact-only events
   (recruiter replies, follow-ups) are hollow, so you can see at a glance
   whether anything actually advanced. */
const ADVANCING = new Set([
  'saved',
  'applied',
  'assessment_received',
  'screening_scheduled',
  'screening_done',
  'interview_scheduled',
  'interview_done',
  'offer_received',
  'accepted',
  'rejected',
  'withdrawn',
  'marked_ghosted',
])

export function Timeline({ events = [] }) {
  if (events.length === 0) {
    return <p className="text-sm text-ink-muted">No events yet.</p>
  }

  return (
    <ol className="relative">
      {events.map((event, index) => {
        const advancing = ADVANCING.has(event.event_type)
        const last = index === events.length - 1

        return (
          <li key={event.id} className="relative flex gap-4 pb-6 last:pb-0">
            {/* Connector, drawn behind the marker and stopped on the last row. */}
            {!last && (
              <span
                aria-hidden="true"
                className="absolute left-[4.25rem] top-2 h-full w-px bg-border-subtle"
              />
            )}

            <time
              dateTime={event.occurred_at}
              className="w-16 shrink-0 pt-px text-right text-xs tabular-nums text-ink-muted"
            >
              {formatDayMonth(event.occurred_at)}
            </time>

            <span
              aria-hidden="true"
              className={`relative z-10 mt-1 size-2.5 shrink-0 rounded-full ring-4 ring-surface ${
                advancing ? 'bg-accent' : 'border border-ink-muted bg-surface'
              }`}
            />

            <div className="min-w-0 flex-1 -mt-0.5">
              <p className="text-sm font-medium">
                {EVENT_LABELS[event.event_type] ?? event.event_type}
                {event.source === 'agent' && (
                  <span className="ml-2 rounded bg-violet-50 px-1.5 py-0.5 text-[10px] font-medium text-violet-700 ring-1 ring-inset ring-violet-200">
                    agent
                  </span>
                )}
              </p>
              {event.note && <p className="mt-0.5 text-sm text-ink-muted">{event.note}</p>}
              <p className="mt-0.5 text-xs text-ink-muted">{relativeDays(event.occurred_at)}</p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
