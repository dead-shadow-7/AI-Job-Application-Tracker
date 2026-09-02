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
                className="absolute top-2 left-[4.25rem] h-full w-px bg-border-subtle/70"
              />
            )}

            <time
              dateTime={event.occurred_at}
              className="w-16 shrink-0 pt-px text-right text-xs text-ink-faint tabular-nums"
            >
              {formatDayMonth(event.occurred_at)}
            </time>

            <span
              aria-hidden="true"
              /* Ringed in the canvas colour so the marker punches a hole in
                 the connector behind it rather than sitting on top of a line
                 that runs straight through it. */
              className={`relative z-10 mt-1 size-2.5 shrink-0 rounded-full ring-4 ring-canvas ${
                advancing
                  ? 'bg-accent shadow-[0_0_10px] shadow-accent/50'
                  : 'border border-ink-faint bg-canvas'
              }`}
            />

            <div className="min-w-0 flex-1 -mt-0.5">
              <p className="text-sm font-medium">
                {EVENT_LABELS[event.event_type] ?? event.event_type}
                {event.source === 'agent' && (
                  <span className="ml-2 rounded bg-accent/14 px-1.5 py-0.5 text-[10px] font-medium text-accent ring-1 ring-accent/30 ring-inset">
                    agent
                  </span>
                )}
              </p>
              {event.note && <p className="mt-0.5 text-sm text-ink-muted">{event.note}</p>}
              <p className="mt-0.5 text-xs text-ink-faint">{relativeDays(event.occurred_at)}</p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
