import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ErrorState } from '@/components/ErrorState'
import { Spinner } from '@/components/Spinner'
import { StatusBadge } from '@/components/StatusBadge'
import { api } from '@/lib/api'
import { WORK_MODE_LABELS, formatSalary } from '@/lib/format'

/**
 * Semantic search results.
 *
 * Rendered as a ranked list rather than reusing the table on purpose. The
 * table's columns exist to be scanned and sorted; these rows have exactly one
 * meaningful ordering — closeness to what was asked — and re-sorting them by
 * salary would throw away the only thing the query established.
 *
 * The similarity figure is shown because the results are never empty-or-right:
 * the backend returns anything under 0.65 cosine distance, which is deliberately
 * permissive so a search rarely dead-ends. A weak match is therefore expected,
 * and saying how weak is what stops it reading as a confident answer.
 */
export function MeaningResults({ query }) {
  const results = useQuery({
    queryKey: ['search', query],
    queryFn: () => api.searchByMeaning(query, 20),
    enabled: query.trim().length >= 2,
  })

  if (results.isPending) return <Spinner label={`Searching for “${query}”`} />
  if (results.error) return <ErrorState error={results.error} onRetry={results.refetch} />

  const hits = results.data ?? []

  if (hits.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border-subtle bg-surface p-10 text-center">
        <p className="text-sm font-medium">Nothing you track resembles “{query}”</p>
        <p className="mt-1 text-sm text-ink-muted">
          This searches the job descriptions by meaning, so it only covers jobs added by pasting
          a description — one entered by hand has no text to match against.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-ink-muted">
        {hits.length} {hits.length === 1 ? 'match' : 'matches'}, closest first.
      </p>

      <ul className="divide-y divide-border-subtle overflow-hidden rounded-xl border border-border-subtle bg-surface">
        {hits.map((hit) => {
          const salary = formatSalary(hit.job)

          return (
            <li
              key={hit.application_id}
              className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 transition hover:bg-surface-muted"
            >
              <div className="min-w-0 flex-1">
                <Link
                  to={`/applications/${hit.application_id}`}
                  className="text-sm font-medium hover:text-accent"
                >
                  {hit.job.title}
                </Link>
                <span className="ml-2 text-sm text-ink-muted">{hit.job.company.name}</span>
                <p className="mt-0.5 text-xs text-ink-muted">
                  {[
                    hit.job.location,
                    hit.job.work_mode && WORK_MODE_LABELS[hit.job.work_mode],
                    salary,
                  ]
                    .filter(Boolean)
                    .join(' · ') || 'No location or salary recorded'}
                </p>
              </div>

              <StatusBadge status={hit.current_status} />

              <div className="w-24 shrink-0" title={`Cosine similarity ${hit.similarity}`}>
                <div className="flex items-baseline justify-end gap-1 text-xs tabular-nums text-ink-muted">
                  {Math.round(hit.similarity * 100)}%
                </div>
                <div className="mt-1 h-1 rounded-full bg-surface-muted">
                  <div
                    className="h-1 rounded-full bg-accent"
                    style={{ width: `${Math.max(4, hit.similarity * 100)}%` }}
                  />
                </div>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
