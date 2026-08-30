import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ErrorState } from '@/components/ErrorState'
import { Spinner } from '@/components/Spinner'
import { StatusBadge } from '@/components/StatusBadge'
import { Timeline } from '@/components/Timeline'
import { api } from '@/lib/api'
import {
  EVENT_LABELS,
  STATUS_LABELS,
  TERMINAL_STATUSES,
  WORK_MODE_LABELS,
  formatDate,
  formatSalary,
  relativeDays,
} from '@/lib/format'

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

const FIELD =
  'w-full rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

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
  const salary = formatSalary(job)
  const stale = !TERMINAL_STATUSES.has(data.current_status)

  return (
    <div className="space-y-6">
      <div>
        <Link to="/" className="text-sm text-ink-muted hover:text-accent">
          ← Applications
        </Link>
        <div className="mt-2 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">{job.title}</h1>
            <p className="mt-0.5 text-sm text-ink-muted">
              {job.company.name}
              {job.location && ` · ${job.location}`}
              {job.work_mode && ` · ${WORK_MODE_LABELS[job.work_mode]}`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={data.current_status} />
            {job.url && (
              <a
                href={job.url}
                target="_blank"
                rel="noreferrer noopener"
                className="rounded-lg border border-border-subtle px-3 py-1.5 text-sm transition hover:bg-surface-muted"
              >
                Posting ↗
              </a>
            )}
          </div>
        </div>
      </div>

      {/* The line the Phase 4 agent will eventually write for you. Showing it
          now proves the event log already carries what that will need. */}
      {stale && (
        <IdleNotice status={data.current_status} days={data.days_since_activity} />
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_20rem]">
        <div className="space-y-6">
          <section className="rounded-xl border border-border-subtle bg-surface p-5">
            <h2 className="text-sm font-medium">Timeline</h2>
            <div className="mt-4">
              <Timeline events={data.events} />
            </div>

            <form
              className="mt-6 flex flex-wrap gap-2 border-t border-border-subtle pt-4"
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
                className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
              >
                Log
              </button>
            </form>
            {addEvent.error && (
              <p className="mt-2 text-sm text-rose-700" role="alert">
                {addEvent.error.message}
              </p>
            )}
          </section>

          {(job.requirements.length > 0 || job.description) && (
            <section className="rounded-xl border border-border-subtle bg-surface p-5">
              <h2 className="text-sm font-medium">The role</h2>
              {job.requirements.length > 0 && (
                <div className="mt-3 grid gap-4 sm:grid-cols-2">
                  <RequirementList
                    title="Must have"
                    items={job.requirements.filter((r) => r.kind === 'must')}
                  />
                  <RequirementList
                    title="Nice to have"
                    items={job.requirements.filter((r) => r.kind === 'nice')}
                  />
                </div>
              )}
              {job.description && (
                <p className="mt-4 whitespace-pre-wrap text-sm text-ink-muted">{job.description}</p>
              )}
            </section>
          )}
        </div>

        <aside className="space-y-6">
          <section className="rounded-xl border border-border-subtle bg-surface p-5">
            <h2 className="text-sm font-medium">Details</h2>
            <dl className="mt-3 space-y-2.5 text-sm">
              <Detail label="Applied" value={formatDate(data.applied_at)} />
              <Detail label="Last activity" value={relativeDays(data.last_activity_at)} />
              <Detail label="Salary" value={salary ?? '—'} />
              <Detail label="Seniority" value={job.seniority ?? '—'} />
              <Detail label="Source" value={job.source_platform ?? '—'} />
              <Detail label="Priority" value={data.priority} />
            </dl>
          </section>

          {job.skills.length > 0 && (
            <section className="rounded-xl border border-border-subtle bg-surface p-5">
              <h2 className="text-sm font-medium">Skills</h2>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {job.skills.map(({ skill }) => (
                  <span
                    key={skill.id}
                    className="rounded-full bg-surface-muted px-2.5 py-1 text-xs text-ink-muted"
                  >
                    {skill.name}
                  </span>
                ))}
              </div>
            </section>
          )}

          <section className="rounded-xl border border-border-subtle bg-surface p-5">
            <h2 className="text-sm font-medium">Match</h2>
            <p className="mt-2 text-sm text-ink-muted">
              Resume match scoring arrives in Phase 3.
            </p>
          </section>

          <button
            type="button"
            onClick={() => {
              if (confirm('Stop tracking this application? Its timeline is deleted too.')) {
                remove.mutate()
              }
            }}
            className="w-full rounded-lg border border-rose-200 px-3 py-2 text-sm font-medium text-rose-700 transition hover:bg-rose-50"
          >
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
    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
      <p className="text-sm text-amber-900">
        No activity for <strong>{days} days</strong> since this moved to {STATUS_LABELS[status] ?? status}.
        A follow-up may be due.
      </p>
      <p className="mt-0.5 text-xs text-amber-700">
        Phase 4 turns this into a rule you configure, and drafts the email.
      </p>
    </div>
  )
}

function RequirementList({ title, items }) {
  if (items.length === 0) return null
  return (
    <div>
      <h3 className="text-xs font-medium text-ink-muted">{title}</h3>
      <ul className="mt-1.5 space-y-1 text-sm">
        {items.map((item) => (
          <li key={item.id} className="flex gap-2">
            <span aria-hidden="true" className="text-ink-muted">
              ·
            </span>
            {item.text}
          </li>
        ))}
      </ul>
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-ink-muted">{label}</dt>
      <dd className="text-right font-medium">{value}</dd>
    </div>
  )
}

function toIso(value) {
  if (!value) return null
  const date = new Date(`${value}T12:00:00`)
  return (date > new Date() ? new Date() : date).toISOString()
}
