import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Clock, ExternalLink, Plus, Trash2 } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { DetailsPanel } from '@/components/DetailsPanel'
import { ErrorState } from '@/components/ErrorState'
import { MatchPanel } from '@/components/MatchPanel'
import { PageHeader } from '@/components/PageHeader'
import { RolePanel } from '@/components/RolePanel'
import { Spinner } from '@/components/Spinner'
import { StatusBadge } from '@/components/StatusBadge'
import { Timeline } from '@/components/Timeline'
import { api } from '@/lib/api'
import { EVENT_LABELS, STATUS_LABELS, TERMINAL_STATUSES, WORK_MODE_LABELS } from '@/lib/format'

const LOGGABLE = [
  'applied',
  'assessment_received',
  'screening_scheduled',
  'screening_done',
  'interview_scheduled',
  'interview_done',
  'offer_received',
  'recruiter_reply',
  'follow_up_sent',
  'rejected',
  'withdrawn',
  'accepted',
  'marked_ghosted',
  'note_added',
]

/* One field style, shared. An inset well rather than an outlined box: on the
   dark canvas an outline reads as a card, and a card is not something you type
   into. */
const FIELD =
  'well w-full rounded-xl px-3 py-2.5 text-sm outline-none transition placeholder:text-ink-faint focus:border-accent/40 focus:shadow-[0_0_0_3px] focus:shadow-accent/12 focus-visible:outline-none'

export function ApplicationDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data, isPending, error, refetch } = useQuery({
    queryKey: ['application', id],
    queryFn: () => api.getApplication(id),
  })

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['application', id] })
    queryClient.invalidateQueries({ queryKey: ['applications'] })
    queryClient.invalidateQueries({ queryKey: ['stats'] })
  }

  const addEvent = useMutation({
    mutationFn: (body) => api.addEvent(id, body),
    onSuccess: invalidate,
  })
  const remove = useMutation({
    mutationFn: () => api.deleteApplication(id),
    onSuccess: () => {
      invalidate()
      navigate('/')
    },
  })

  if (isPending) return <Spinner label="Loading application" />
  if (error) return <ErrorState error={error} onRetry={refetch} />

  const { job } = data
  const stale = !TERMINAL_STATUSES.has(data.current_status)

  return (
    <div className="space-y-6">
      <PageHeader
        back={{ to: '/', label: 'Applications' }}
        title={job.title}
        subtitle={[
          job.company.name,
          job.location,
          job.work_mode && WORK_MODE_LABELS[job.work_mode],
        ]
          .filter(Boolean)
          .join(' · ')}
        actions={
          <>
            <StatusBadge status={data.current_status} />
            {job.url && (
              <a
                href={job.url}
                target="_blank"
                rel="noreferrer noopener"
                className="inline-flex items-center gap-1.5 rounded-xl border border-border-subtle px-3 py-2 text-sm text-ink-muted transition hover:border-border-strong hover:text-ink"
              >
                Posting
                <ExternalLink size={14} aria-hidden="true" />
              </a>
            )}
          </>
        }
      />

      {/* The line the Phase 4 agent will eventually write for you. Showing it
          now proves the event log already carries what that will need. */}
      {stale && (
        <IdleNotice status={data.current_status} days={data.days_since_activity} />
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_20rem]">
        <div className="space-y-6">
          <section className="glass rounded-2xl p-5">
            <h2 className="text-sm font-semibold">Timeline</h2>
            <div className="mt-4">
              <Timeline events={data.events} />
            </div>

            <form
              className="mt-6 flex flex-wrap gap-2 border-t border-border-subtle/70 pt-4"
              onSubmit={(e) => {
                e.preventDefault()
                const form = new FormData(e.currentTarget)
                addEvent.mutate({
                  event_type: form.get('event_type'),
                  occurred_at: toIso(form.get('occurred_on')),
                  note: form.get('note') || null,
                })
                e.currentTarget.reset()
              }}
            >
              <select name="event_type" required className={`${FIELD} w-auto flex-1`}>
                {LOGGABLE.map((value) => (
                  <option key={value} value={value}>
                    {EVENT_LABELS[value]}
                  </option>
                ))}
              </select>
              <input
                type="date"
                name="occurred_on"
                defaultValue={new Date().toISOString().slice(0, 10)}
                max={new Date().toISOString().slice(0, 10)}
                aria-label="Date it happened"
                className={`${FIELD} w-auto`}
              />
              <input name="note" placeholder="Note (optional)" className={`${FIELD} w-auto flex-1`} />
              <button
                type="submit"
                disabled={addEvent.isPending}
                className="inline-flex cursor-pointer items-center gap-1.5 rounded-xl bg-accent px-3.5 py-2 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Plus size={15} aria-hidden="true" />
                Log
              </button>
            </form>
            {addEvent.error && (
              <p className="mt-2 text-sm text-danger" role="alert">
                {addEvent.error.message}
              </p>
            )}
          </section>

          <RolePanel job={job} />
        </div>

        <aside className="space-y-6">
          <DetailsPanel application={data} />

          <MatchPanel applicationId={id} />

          <button
            type="button"
            onClick={() => {
              if (confirm('Stop tracking this application? Its timeline is deleted too.')) {
                remove.mutate()
              }
            }}
            className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-danger/30 px-3 py-2.5 text-sm font-medium text-danger transition hover:bg-danger/10"
          >
            <Trash2 size={15} aria-hidden="true" />
            Stop tracking
          </button>
        </aside>
      </div>
    </div>
  )
}

function IdleNotice({ status, days }) {
  // `days` comes from the server so the badge here and the "Idle" column on the
  // dashboard can never disagree, and render stays free of clock reads.
  if (days < 7) return null

  return (
    <div className="flex gap-3 rounded-2xl border border-signal/30 bg-signal/8 px-4 py-3.5">
      <Clock size={17} aria-hidden="true" className="mt-0.5 shrink-0 text-signal" />
      <div>
        <p className="text-sm text-signal">
          No activity for <strong className="font-semibold">{days} days</strong> since this moved to{' '}
          {STATUS_LABELS[status] ?? status}. A follow-up may be due.
        </p>
        <p className="mt-0.5 text-xs text-signal/70">
          Phase 4 turns this into a rule you configure, and drafts the email.
        </p>
      </div>
    </div>
  )
}


function toIso(value) {
  if (!value) return null
  const date = new Date(`${value}T12:00:00`)
  return (date > new Date() ? new Date() : date).toISOString()
}
