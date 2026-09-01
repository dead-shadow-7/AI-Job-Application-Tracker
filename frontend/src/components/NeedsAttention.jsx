import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '@/lib/api'

/**
 * The panel this whole project was described to produce:
 *
 *   "Amazon has had no activity for 9 days since it moved to applied."
 *
 * Every row states which rule fired and at what threshold, so the number is
 * traceable rather than something the dashboard merely asserts. A suggestion
 * you cannot explain is one you stop trusting.
 */
export function NeedsAttention() {
  const queryClient = useQueryClient()

  const attention = useQuery({ queryKey: ['needs-attention'], queryFn: api.needsAttention })

  const closeGhosted = useMutation({
    mutationFn: api.closeGhosted,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['needs-attention'] })
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
    },
  })

  const items = attention.data ?? []
  if (attention.isPending || items.length === 0) return null

  const ghostable = items.filter((i) => i.rule_action === 'mark_ghosted')

  return (
    <section className="rounded-xl border border-amber-200 bg-amber-50/60 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-medium text-amber-900">
          {items.length === 1
            ? 'One application needs attention'
            : `${items.length} applications need attention`}
        </h2>
        {ghostable.length > 0 && (
          <button
            type="button"
            onClick={() => {
              if (
                confirm(
                  `Close ${ghostable.length} application(s) with no response? ` +
                    'This is logged on each timeline and can be undone by adding an event.',
                )
              ) {
                closeGhosted.mutate()
              }
            }}
            disabled={closeGhosted.isPending}
            className="rounded-lg border border-amber-300 bg-surface px-3 py-1.5 text-xs font-medium text-amber-900 transition hover:bg-amber-100 disabled:opacity-60"
          >
            {closeGhosted.isPending
              ? 'Closing…'
              : `Close ${ghostable.length} unanswered`}
          </button>
        )}
      </div>

      <ul className="mt-3 space-y-2">
        {items.map((item) => (
          <li
            key={item.application_id}
            className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 rounded-lg bg-surface px-3 py-2"
          >
            <div className="min-w-0 flex-1">
              <Link
                to={`/applications/${item.application_id}`}
                className="text-sm font-medium hover:text-accent"
              >
                {item.job.company.name}
              </Link>
              <span className="ml-2 text-sm text-ink-muted">{item.job.title}</span>
              <p className="mt-0.5 text-xs text-ink-muted">{item.reason}</p>
            </div>

            <span
              className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
                item.rule_action === 'mark_ghosted'
                  ? 'bg-orange-50 text-orange-800 ring-orange-200'
                  : 'bg-amber-100 text-amber-900 ring-amber-300'
              }`}
              title={`Rule: ${item.rule_threshold} days in ${item.current_status}`}
            >
              {item.rule_action === 'mark_ghosted'
                ? 'no response'
                : 'follow up'}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
