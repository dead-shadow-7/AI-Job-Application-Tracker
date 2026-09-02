import { useQuery } from '@tanstack/react-query'
import {
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  ClipboardPaste,
  Inbox,
  PlusCircle,
  Search,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ErrorState } from '@/components/ErrorState'
import { MeaningResults } from '@/components/MeaningResults'
import { NeedsAttention } from '@/components/NeedsAttention'
import { PageHeader } from '@/components/PageHeader'
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

/* Two searches that cannot be merged, so they are offered as a choice rather
   than silently blended.
     filter  — ILIKE over title and company. Narrows the table, combines with
               the status chips and the sort, and paginates.
     meaning — pgvector over the job descriptions. Returns its own ranking, so
               the chips and the sort have nothing to act on and are hidden. */
const MODES = [
  ['filter', 'Filter', 'Match the role or company name'],
  ['meaning', 'By meaning', 'Search what the postings are about, not the words they use'],
]

export function Applications() {
  // Filters live in the URL so a filtered view is linkable and survives reload.
  const [params, setParams] = useSearchParams()
  const [searchDraft, setSearchDraft] = useState(params.get('search') ?? '')

  const status = params.getAll('status')
  const search = params.get('search') ?? ''
  const activeOnly = params.get('active_only') === 'true'
  const sort = params.get('sort') ?? 'last_activity_at'
  const page = Number(params.get('page') ?? 0)
  const mode = params.get('mode') === 'meaning' ? 'meaning' : 'filter'
  const searching = mode === 'meaning' && search.trim().length >= 2

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
    // Nothing renders it while semantic results are showing, and it would
    // otherwise fire a request per keystroke-submit for a list nobody sees.
    enabled: !searching,
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
  const filtered = status.length > 0 || Boolean(search) || activeOnly

  return (
    <div className="space-y-5">
      <PageHeader
        title="Applications"
        subtitle="Every role you are tracking, and how long each has been quiet."
        actions={
          <>
            <Link
              to="/applications/new"
              className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-border-subtle px-3.5 py-2.5 text-sm font-medium text-ink-muted transition hover:border-border-strong hover:text-ink"
            >
              <PlusCircle size={16} aria-hidden="true" />
              Add by hand
            </Link>
            <Link
              to="/applications/paste"
              className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-accent px-3.5 py-2.5 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover"
            >
              <ClipboardPaste size={16} aria-hidden="true" />
              Paste a job
            </Link>
          </>
        }
      />

      {/* Three counters, narrowing left to right: everything, everything still
          open, and the ones where somebody is actually talking to you. The
          needs-attention count is deliberately not here — the panel below says
          the same number and can name the applications, and a tile that
          duplicates the thing directly under it is just noise. */}
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Tracked" value={stats.data?.total} />
        <Stat label="Active" value={stats.data?.active} />
        <Stat label="In conversation" value={inConversation(stats.data)} tone="accent" />
      </div>

      <NeedsAttention />

      <div className="glass space-y-3.5 rounded-2xl p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            update({ search: searchDraft, mode })
          }}
          className="flex gap-2"
        >
          <div className="well flex min-w-0 flex-1 items-center gap-2.5 rounded-xl px-3 transition focus-within:border-accent/40 focus-within:shadow-[0_0_0_3px] focus-within:shadow-accent/12">
            <Search size={16} aria-hidden="true" className="shrink-0 text-ink-faint" />
            <input
              type="search"
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
              placeholder={
                mode === 'meaning'
                  ? 'Describe the kind of role — “retrieval and agents”…'
                  : 'Search by role or company…'
              }
              aria-label="Search applications"
              className="min-w-0 flex-1 bg-transparent py-2.5 text-sm outline-none placeholder:text-ink-faint focus-visible:outline-none"
            />
          </div>
          <button
            type="submit"
            className="shrink-0 cursor-pointer rounded-xl border border-border-subtle px-4 py-2.5 text-sm font-medium text-ink-muted transition hover:border-border-strong hover:text-ink"
          >
            Search
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-2.5">
          <div
            className="well flex rounded-lg p-0.5"
            role="group"
            aria-label="Search mode"
          >
            {MODES.map(([value, label, hint]) => (
              <button
                key={value}
                type="button"
                title={hint}
                aria-pressed={mode === value}
                /* Carries whatever is typed but not yet submitted. Switching
                   mode is itself an intent to search, and dropping the draft
                   would silently clear a box that still displays it. */
                onClick={() => update({ mode: value === 'filter' ? '' : value, search: searchDraft })}
                className={`cursor-pointer rounded-md px-2.5 py-1.5 text-xs font-medium transition ${
                  mode === value
                    ? 'bg-accent/15 text-accent'
                    : 'text-ink-faint hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {mode === 'meaning' && (
            <p className="text-xs text-ink-faint">
              Ranked by what the descriptions are about, so status filters and sorting do not apply.
            </p>
          )}
        </div>

        <div className={`flex flex-wrap items-center gap-1.5 ${searching ? 'hidden' : ''}`}>
          {STATUS_ORDER.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => toggleStatus(value)}
              aria-pressed={status.includes(value)}
              className={`cursor-pointer rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset transition ${
                status.includes(value)
                  ? 'bg-accent text-accent-ink ring-accent'
                  : 'text-ink-muted ring-border-subtle hover:bg-surface-muted/60 hover:text-ink'
              }`}
            >
              {STATUS_LABELS[value]}
            </button>
          ))}

          <label className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-ink-muted">
            <input
              type="checkbox"
              checked={activeOnly}
              onChange={(e) => update({ active_only: e.target.checked ? 'true' : '' })}
              className="size-3.5 cursor-pointer accent-accent"
            />
            Active only
          </label>

          <div className="flex items-center gap-1.5 text-ink-faint">
            <ArrowUpDown size={13} aria-hidden="true" />
            <select
              value={sort}
              onChange={(e) => update({ sort: e.target.value })}
              aria-label="Sort by"
              className="cursor-pointer rounded-lg border border-border-subtle bg-surface px-2 py-1 text-xs text-ink-muted outline-none transition hover:text-ink focus:border-accent"
            >
              <option value="last_activity_at">Last activity</option>
              <option value="created_at">Recently added</option>
              <option value="applied_at">Applied date</option>
              <option value="company">Company</option>
              <option value="title">Role</option>
            </select>
          </div>
        </div>
      </div>

      {searching && <MeaningResults query={search} />}

      {!searching && applications.isPending && <Spinner label="Loading applications" />}
      {!searching && applications.error && (
        <ErrorState error={applications.error} onRetry={applications.refetch} />
      )}

      {!searching && applications.data && items.length === 0 && (
        <div className="glass flex flex-col items-center rounded-2xl border-dashed px-6 py-14 text-center">
          <span
            aria-hidden="true"
            className="grid size-12 place-items-center rounded-2xl bg-surface-muted/60 text-ink-faint"
          >
            <Inbox size={22} />
          </span>
          <p className="mt-4 font-display text-base font-semibold">
            {filtered ? 'Nothing matches those filters' : 'No applications yet'}
          </p>
          <p className="mt-1 max-w-sm text-sm text-ink-muted">
            {filtered
              ? 'Try clearing a filter, or search by meaning instead.'
              : 'Track your first job to start building the timeline.'}
          </p>
          {!filtered && (
            <Link
              to="/applications/paste"
              className="mt-5 inline-flex cursor-pointer items-center gap-2 rounded-xl bg-accent px-3.5 py-2.5 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover"
            >
              <ClipboardPaste size={16} aria-hidden="true" />
              Paste a job description
            </Link>
          )}
        </div>
      )}

      {!searching && items.length > 0 && (
        <div className="glass overflow-hidden rounded-2xl">
          <div className="overflow-x-auto">
            <table className="w-full min-w-208 text-sm">
              <thead>
                <tr className="border-b border-border-subtle/70 text-left text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
                  <th scope="col" className="px-4 py-3">Role</th>
                  <th scope="col" className="px-4 py-3">Company</th>
                  <th scope="col" className="px-4 py-3">Status</th>
                  <th scope="col" className="px-4 py-3">Salary</th>
                  <th scope="col" className="px-4 py-3">Applied</th>
                  <th scope="col" className="px-4 py-3 text-right">Idle</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle/50">
                {items.map((item) => {
                  const salary = formatSalary(item.job)
                  // Surfaces the Phase 4 signal early: a week of silence on an
                  // open application is what the follow-up agent will act on.
                  const stale = item.days_since_activity >= 7 && !isTerminal(item.current_status)

                  return (
                    <tr key={item.id} className="group transition hover:bg-white/4">
                      <td className="px-4 py-3.5">
                        <Link
                          to={`/applications/${item.id}`}
                          className="font-medium text-ink transition group-hover:text-accent"
                        >
                          {item.job.title}
                        </Link>
                        {item.job.work_mode && (
                          <span className="ml-2 text-xs text-ink-faint">
                            {WORK_MODE_LABELS[item.job.work_mode]}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3.5 text-ink-muted">{item.job.company.name}</td>
                      <td className="px-4 py-3.5">
                        <StatusBadge status={item.current_status} />
                      </td>
                      <td className="px-4 py-3.5 text-ink-muted tabular-nums">{salary ?? '—'}</td>
                      <td className="px-4 py-3.5 text-ink-muted tabular-nums">
                        {formatDate(item.applied_at)}
                      </td>
                      <td
                        className={`px-4 py-3.5 text-right tabular-nums ${
                          stale ? 'font-medium text-signal' : 'text-ink-faint'
                        }`}
                        title={stale ? 'No activity for a week or more' : undefined}
                      >
                        {item.days_since_activity}d
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!searching && total > PAGE_SIZE && (
        <div className="flex items-center justify-between text-sm">
          <p className="font-mono text-xs text-ink-faint tabular-nums">
            {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={page === 0}
              onClick={() => update({ page: String(page - 1) })}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border-subtle px-3 py-1.5 text-sm text-ink-muted transition hover:border-border-strong hover:text-ink disabled:cursor-not-allowed disabled:opacity-35"
            >
              <ChevronLeft size={15} aria-hidden="true" />
              Previous
            </button>
            <button
              type="button"
              disabled={(page + 1) * PAGE_SIZE >= total}
              onClick={() => update({ page: String(page + 1) })}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border-subtle px-3 py-1.5 text-sm text-ink-muted transition hover:border-border-strong hover:text-ink disabled:cursor-not-allowed disabled:opacity-35"
            >
              Next
              <ChevronRight size={15} aria-hidden="true" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/* The stages where a human is on the other end. `by_status` is the only place
   the breakdown exists, so this is summed here rather than asking the API for a
   fourth counter it does not have. */
function inConversation(stats) {
  if (!stats) return undefined
  return stats.by_status
    .filter(({ status }) => ['screening', 'interviewing', 'offer'].includes(status))
    .reduce((total, { count }) => total + count, 0)
}

/* A counter. Renders an em dash rather than a zero while the request is in
   flight — a stat row that flashes three zeroes on every load reads, for the
   half second it is up, as an empty account. */
function Stat({ label, value, tone }) {
  const colour = tone === 'accent' ? 'text-accent' : 'text-ink'

  return (
    <div className="glass rounded-2xl px-4 py-3.5">
      <p className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
        {label}
      </p>
      <p className={`mt-1.5 font-mono text-2xl leading-none font-medium tabular-nums ${colour}`}>
        {value ?? '—'}
      </p>
    </div>
  )
}

function isTerminal(status) {
  return ['accepted', 'rejected', 'withdrawn', 'ghosted'].includes(status)
}
