import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ErrorState } from '@/components/ErrorState'
import { Spinner } from '@/components/Spinner'
import { StatusBadge } from '@/components/StatusBadge'
import { api } from '@/lib/api'
import { STATUS_LABELS, WORK_MODE_LABELS, formatDate, formatSalary } from '@/lib/format'

const STATUS_ORDER = [
  'saved',
  'applied',
  'screening',
  'interviewing',
  'offer',
  'accepted',
  'rejected',
  'withdrawn',
  'ghosted',
]

const PAGE_SIZE = 25

export function Applications() {
  // Filters live in the URL so a filtered view is linkable and survives reload.
  const [params, setParams] = useSearchParams()
  const [searchDraft, setSearchDraft] = useState(params.get('search') ?? '')

  const status = params.getAll('status')
  const search = params.get('search') ?? ''
  const activeOnly = params.get('active_only') === 'true'
  const sort = params.get('sort') ?? 'last_activity_at'
  const page = Number(params.get('page') ?? 0)

  const filters = useMemo(
    () => ({
      status,
      search,
      active_only: activeOnly,
      sort,
      order: 'desc',
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [params.toString()],
  )

  const applications = useQuery({
    queryKey: ['applications', filters],
    queryFn: () => api.listApplications(filters),
  })
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.getApplicationStats })

  function update(changes) {
    const next = new URLSearchParams(params)
    for (const [key, value] of Object.entries(changes)) {
      next.delete(key)
      if (Array.isArray(value)) value.forEach((v) => next.append(key, v))
      else if (value) next.set(key, String(value))
    }
    // Any filter change invalidates the current page offset.
    if (!('page' in changes)) next.delete('page')
    setParams(next, { replace: true })
  }

  function toggleStatus(value) {
    update({ status: status.includes(value) ? status.filter((s) => s !== value) : [...status, value] })
  }

  const total = applications.data?.total ?? 0
  const items = applications.data?.items ?? []

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Applications</h1>
          <p className="mt-0.5 text-sm text-ink-muted">
            {stats.data
              ? `${stats.data.total} tracked · ${stats.data.active} active · ${stats.data.needs_attention} need attention`
              : ' '}
          </p>
        </div>
        <Link
          to="/applications/new"
          className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-accent-hover"
        >
          Track a job
        </Link>
      </div>

      <div className="space-y-3 rounded-xl border border-border-subtle bg-surface p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            update({ search: searchDraft })
          }}
          className="flex gap-2"
        >
          <input
            type="search"
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            placeholder="Search by role or company…"
            aria-label="Search applications"
            className="min-w-0 flex-1 rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
          />
          <button
            type="submit"
            className="rounded-lg border border-border-subtle px-3 py-2 text-sm font-medium transition hover:bg-surface-muted"
          >
            Search
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-1.5">
          {STATUS_ORDER.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => toggleStatus(value)}
              aria-pressed={status.includes(value)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset transition ${
                status.includes(value)
                  ? 'bg-accent text-white ring-accent'
                  : 'bg-surface text-ink-muted ring-border-subtle hover:bg-surface-muted'
              }`}
            >
              {STATUS_LABELS[value]}
            </button>
          ))}

          <label className="ml-auto flex items-center gap-2 text-xs text-ink-muted">
            <input
              type="checkbox"
              checked={activeOnly}
              onChange={(e) => update({ active_only: e.target.checked ? 'true' : '' })}
              className="rounded border-border-subtle"
            />
            Active only
          </label>

          <select
            value={sort}
            onChange={(e) => update({ sort: e.target.value })}
            aria-label="Sort by"
            className="rounded-lg border border-border-subtle px-2 py-1 text-xs outline-none focus:border-accent"
          >
            <option value="last_activity_at">Last activity</option>
            <option value="created_at">Recently added</option>
            <option value="applied_at">Applied date</option>
            <option value="company">Company</option>
            <option value="title">Role</option>
          </select>
        </div>
      </div>

      {applications.isPending && <Spinner label="Loading applications" />}
      {applications.error && (
        <ErrorState error={applications.error} onRetry={applications.refetch} />
      )}

      {applications.data && items.length === 0 && (
        <div className="rounded-xl border border-dashed border-border-subtle bg-surface p-10 text-center">
          <p className="text-sm font-medium">
            {status.length || search ? 'Nothing matches those filters' : 'No applications yet'}
          </p>
          <p className="mt-1 text-sm text-ink-muted">
            {status.length || search
              ? 'Try clearing a filter.'
              : 'Track your first job to start building the timeline.'}
          </p>
        </div>
      )}

      {items.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-border-subtle bg-surface">
          <table className="w-full min-w-[52rem] text-sm">
            <thead className="border-b border-border-subtle text-left text-xs text-ink-muted">
              <tr>
                <th scope="col" className="px-4 py-2.5 font-medium">Role</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Company</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Status</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Salary</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Applied</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Idle</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {items.map((item) => {
                const salary = formatSalary(item.job)
                // Surfaces the Phase 4 signal early: a week of silence on an
                // open application is what the follow-up agent will act on.
                const stale = item.days_since_activity >= 7 && !isTerminal(item.current_status)

                return (
                  <tr key={item.id} className="transition hover:bg-surface-muted">
                    <td className="px-4 py-3">
                      <Link
                        to={`/applications/${item.id}`}
                        className="font-medium text-ink hover:text-accent"
                      >
                        {item.job.title}
                      </Link>
                      {item.job.work_mode && (
                        <span className="ml-2 text-xs text-ink-muted">
                          {WORK_MODE_LABELS[item.job.work_mode]}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-ink-muted">{item.job.company.name}</td>
                    <td className="px-4 py-3">
                      <StatusBadge status={item.current_status} />
                    </td>
                    <td className="px-4 py-3 tabular-nums text-ink-muted">{salary ?? '—'}</td>
                    <td className="px-4 py-3 text-ink-muted">{formatDate(item.applied_at)}</td>
                    <td
                      className={`px-4 py-3 text-right tabular-nums ${
                        stale ? 'font-medium text-amber-700' : 'text-ink-muted'
                      }`}
                    >
                      {item.days_since_activity}d
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between text-sm">
          <p className="text-ink-muted">
            {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={page === 0}
              onClick={() => update({ page: String(page - 1) })}
              className="rounded-lg border border-border-subtle px-3 py-1.5 transition hover:bg-surface-muted disabled:opacity-40"
            >
              Previous
            </button>
            <button
              type="button"
              disabled={(page + 1) * PAGE_SIZE >= total}
              onClick={() => update({ page: String(page + 1) })}
              className="rounded-lg border border-border-subtle px-3 py-1.5 transition hover:bg-surface-muted disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function isTerminal(status) {
  return ['accepted', 'rejected', 'withdrawn', 'ghosted'].includes(status)
}
