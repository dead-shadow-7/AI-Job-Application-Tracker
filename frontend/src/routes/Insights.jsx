import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ErrorState } from '@/components/ErrorState'
import { Spinner } from '@/components/Spinner'
import { api } from '@/lib/api'
import { STATUS_LABELS } from '@/lib/format'

/* Same order as STATUS_ORDER on the applications table, and the same order the
   backend emits. Rendered whole, including the zeros: a funnel with stages
   missing reads as data loss rather than as a stage nobody reached. */
const FUNNEL_TONE = {
  saved: 'bg-slate-300',
  applied: 'bg-blue-400',
  screening: 'bg-indigo-400',
  interviewing: 'bg-violet-400',
  offer: 'bg-amber-400',
  accepted: 'bg-emerald-500',
  rejected: 'bg-rose-400',
  withdrawn: 'bg-slate-300',
  ghosted: 'bg-orange-300',
}

export function Insights() {
  const analytics = useQuery({ queryKey: ['analytics'], queryFn: api.getAnalytics })

  if (analytics.isPending) return <Spinner label="Reading your history" />
  if (analytics.error) return <ErrorState error={analytics.error} onRetry={analytics.refetch} />

  const data = analytics.data

  if (data.total === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border-subtle bg-surface p-10 text-center">
        <p className="text-sm font-medium">Nothing to analyse yet</p>
        <p className="mt-1 text-sm text-ink-muted">
          These figures come from your own application history.{' '}
          <Link to="/" className="font-medium text-accent hover:underline">
            Track a job
          </Link>{' '}
          to start building it.
        </p>
      </div>
    )
  }

  const peak = Math.max(1, ...data.funnel.map((stage) => stage.count))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">Insights</h1>
        <p className="mt-0.5 text-sm text-ink-muted">
          Computed from your timeline events — nothing here is a model&rsquo;s opinion.
        </p>
      </div>

      {/* The caveat leads rather than sits in a footnote. A response rate over
          three applications is noise, and a dashboard that presents noise
          confidently gets believed and then acted on. */}
      {data.caveat && (
        <p className="rounded-xl border border-amber-200 bg-amber-50/60 px-4 py-3 text-sm text-amber-900">
          {data.caveat}
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-3">
        <Metric
          label="Response rate"
          value={data.response_rate == null ? null : `${Math.round(data.response_rate * 100)}%`}
          // The denominator, always. Not `total` — a job you saved but never
          // applied to has not gone unanswered.
          detail={
            data.submitted === 0
              ? 'No applications sent yet'
              : `${data.responses} of ${data.submitted} sent`
          }
          muted={data.sample_is_small}
        />
        <Metric
          label="Typical wait for a reply"
          value={
            data.median_days_to_response == null
              ? null
              : `${data.median_days_to_response} ${data.median_days_to_response === 1 ? 'day' : 'days'}`
          }
          /* Median, not mean: one company replying after four months would drag
             an average somewhere you should not plan around. */
          detail={data.responses === 0 ? 'Nobody has replied yet' : 'Median, across replies'}
          muted={data.sample_is_small}
        />
        <Metric label="Tracked" value={String(data.total)} detail={`${data.submitted} applied`} />
      </div>

      <section className="rounded-xl border border-border-subtle bg-surface p-5">
        <h2 className="text-sm font-medium">Where they stand</h2>
        <p className="mt-0.5 text-xs text-ink-muted">
          Current status of every tracked application.
        </p>

        <dl className="mt-4 space-y-2">
          {data.funnel.map((stage) => (
            <div
              key={stage.status}
              className="grid grid-cols-[7.5rem_1fr_2.5rem] items-center gap-3 text-xs"
            >
              <dt className={stage.count === 0 ? 'text-ink-muted opacity-60' : 'text-ink-muted'}>
                {STATUS_LABELS[stage.status] ?? stage.status}
              </dt>
              <div className="h-2 rounded-full bg-surface-muted" aria-hidden="true">
                {stage.count > 0 && (
                  <div
                    className={`h-2 rounded-full ${FUNNEL_TONE[stage.status] ?? 'bg-slate-300'}`}
                    style={{ width: `${Math.max(3, (stage.count / peak) * 100)}%` }}
                  />
                )}
              </div>
              <dd
                className={`text-right tabular-nums ${
                  stage.count === 0 ? 'text-ink-muted opacity-60' : 'font-medium'
                }`}
              >
                {stage.count}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="rounded-xl border border-border-subtle bg-surface p-5">
        <h2 className="text-sm font-medium">By platform</h2>
        <p className="mt-0.5 text-xs text-ink-muted">
          Which sources actually get you a reply. Only applications you sent are counted.
        </p>

        {data.by_platform.length === 0 ? (
          <p className="mt-4 text-sm text-ink-muted">
            No applications sent yet, so there is nothing to compare.
          </p>
        ) : (
          <table className="mt-4 w-full text-sm">
            <thead className="border-b border-border-subtle text-left text-xs text-ink-muted">
              <tr>
                <th scope="col" className="py-2 font-medium">Platform</th>
                <th scope="col" className="py-2 text-right font-medium">Applied</th>
                <th scope="col" className="py-2 text-right font-medium">Replied</th>
                <th scope="col" className="py-2 text-right font-medium">Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {data.by_platform.map((row) => (
                <tr key={row.platform}>
                  <td className="py-2">
                    {row.platform === 'unknown' ? (
                      <span
                        className="text-ink-muted"
                        title="Added by hand, or pasted without naming a source."
                      >
                        Not recorded
                      </span>
                    ) : (
                      row.platform
                    )}
                  </td>
                  <td className="py-2 text-right tabular-nums text-ink-muted">
                    {row.applications}
                  </td>
                  <td className="py-2 text-right tabular-nums text-ink-muted">{row.responses}</td>
                  <td className="py-2 text-right tabular-nums">
                    {row.response_rate == null ? '—' : `${Math.round(row.response_rate * 100)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Named rather than silently absent, so the gap does not read as a bug.
          These need a corpus to say anything true; below roughly fifty
          applications they would produce confident nonsense. */}
      <p className="text-xs text-ink-muted">
        Rejection patterns, skill trends and recommendations are deliberately not here yet — they
        need far more history than {data.total} {data.total === 1 ? 'application' : 'applications'}{' '}
        before they would say anything true.
      </p>
    </div>
  )
}

function Metric({ label, value, detail, muted = false }) {
  return (
    <div className="rounded-xl border border-border-subtle bg-surface p-4">
      <p className="text-xs text-ink-muted">{label}</p>
      <p
        className={`mt-1 text-2xl font-semibold tabular-nums ${
          value == null ? 'text-ink-muted' : muted ? 'text-ink opacity-70' : 'text-ink'
        }`}
      >
        {value ?? '—'}
      </p>
      <p className="mt-0.5 text-xs text-ink-muted">{detail}</p>
    </div>
  )
}
