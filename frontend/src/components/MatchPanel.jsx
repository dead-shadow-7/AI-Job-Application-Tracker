import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '@/lib/api'

/* Weights mirror WEIGHTS in backend/app/services/matching.py. Shown rather
   than hidden: a score you cannot decompose is a score you cannot argue with,
   and the whole design rests on this number being explicable. */
const COMPONENTS = [
  ['must_have_skills', 'Must-have skills', 45],
  ['nice_to_have_skills', 'Nice-to-have skills', 15],
  ['experience', 'Experience fit', 15],
  ['seniority', 'Seniority fit', 10],
  ['rubric', 'Evidence review', 15],
]

function tone(score) {
  if (score >= 75) return { text: 'text-emerald-700', bar: 'bg-emerald-500', ring: 'ring-emerald-200', bg: 'bg-emerald-50' }
  if (score >= 50) return { text: 'text-amber-800', bar: 'bg-amber-500', ring: 'ring-amber-200', bg: 'bg-amber-50' }
  return { text: 'text-rose-700', bar: 'bg-rose-500', ring: 'ring-rose-200', bg: 'bg-rose-50' }
}

export function MatchPanel({ applicationId }) {
  const queryClient = useQueryClient()

  const match = useQuery({
    queryKey: ['match', applicationId],
    queryFn: () => api.getMatch(applicationId),
  })
  const resumes = useQuery({ queryKey: ['resumes'], queryFn: api.listResumes })

  const compute = useMutation({
    mutationFn: () => api.computeMatch(applicationId),
    onSuccess: (result) => {
      queryClient.setQueryData(['match', applicationId], result)
      queryClient.invalidateQueries({ queryKey: ['applications'] })
    },
  })

  const hasResume = (resumes.data?.length ?? 0) > 0
  const data = match.data

  return (
    <section className="rounded-xl border border-border-subtle bg-surface p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium">Resume match</h2>
        {hasResume && (
          <button
            type="button"
            onClick={() => compute.mutate()}
            disabled={compute.isPending}
            className="rounded-lg border border-border-subtle px-3 py-1.5 text-xs transition hover:bg-surface-muted disabled:opacity-60"
          >
            {compute.isPending ? 'Scoring…' : data ? 'Recompute' : 'Score this job'}
          </button>
        )}
      </div>

      {!hasResume && !resumes.isPending && (
        <p className="mt-3 text-sm text-ink-muted">
          <Link to="/resumes" className="font-medium text-accent hover:underline">
            Add a resume
          </Link>{' '}
          to score how well this job fits you.
        </p>
      )}

      {compute.error && (
        <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700" role="alert">
          {compute.error.message}
        </p>
      )}

      {hasResume && !data && !compute.isPending && (
        <p className="mt-3 text-sm text-ink-muted">Not scored yet.</p>
      )}

      {data && (
        <div className="mt-4 space-y-4">
          <div className="flex items-baseline gap-2">
            <span className={`text-3xl font-semibold tabular-nums ${tone(data.overall_score).text}`}>
              {data.overall_score}
            </span>
            <span className="text-sm text-ink-muted">/ 100</span>
          </div>

          <dl className="space-y-2">
            {COMPONENTS.map(([key, label, weight]) => {
              const value = data.subscores?.[key]
              if (value === undefined) return null
              return (
                <div key={key} className="grid grid-cols-[1fr_auto] items-center gap-x-3 text-xs">
                  <dt className="text-ink-muted">
                    {label}
                    <span className="ml-1.5 text-[10px] opacity-70">{weight}%</span>
                  </dt>
                  <dd className="tabular-nums">{Math.round(value * 100)}%</dd>
                  <div className="col-span-2 h-1 rounded-full bg-surface-muted">
                    <div
                      className={`h-1 rounded-full ${tone(value * 100).bar}`}
                      style={{ width: `${Math.max(2, value * 100)}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </dl>

          {data.narrative && (
            <p className={`rounded-lg p-3 text-sm ${tone(data.overall_score).bg}`}>
              {data.narrative}
            </p>
          )}

          {data.missing_skills?.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-ink-muted">Required but not in your resume</h3>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {data.missing_skills.map((skill) => (
                  <span
                    key={skill}
                    className="rounded-full bg-rose-50 px-2 py-0.5 text-xs text-rose-700 ring-1 ring-inset ring-rose-200"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {data.matched_skills?.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-ink-muted">Matched</h3>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {data.matched_skills.map((skill) => (
                  <span
                    key={skill}
                    className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700 ring-1 ring-inset ring-emerald-200"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {data.gaps?.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-ink-muted">Gaps</h3>
              <ul className="mt-1 space-y-1 text-sm">
                {data.gaps.map((gap) => (
                  <li key={gap} className="flex gap-2 text-ink-muted">
                    <span aria-hidden="true">·</span>
                    {gap}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!data.model && (
            <p className="text-xs text-ink-muted">
              Scored without the evidence review — the model was unavailable, so this reflects
              the other 85% only.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
