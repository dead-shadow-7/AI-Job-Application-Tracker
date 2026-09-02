import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
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

/* Three bands, and the same three everywhere the score appears: the ring, the
   subscore bars and the narrative all agree on what 62 means. `stroke` is the
   raw value rather than a class because an SVG stroke cannot read a Tailwind
   text colour. */
function tone(score) {
  if (score >= 75) {
    return {
      text: 'text-positive',
      bar: 'bg-positive',
      wash: 'border-positive/25 bg-positive/8',
      stroke: 'oklch(78% 0.15 158)',
    }
  }
  if (score >= 50) {
    return {
      text: 'text-signal',
      bar: 'bg-signal',
      wash: 'border-signal/25 bg-signal/8',
      stroke: 'oklch(82% 0.13 72)',
    }
  }
  return {
    text: 'text-danger',
    bar: 'bg-danger',
    wash: 'border-danger/25 bg-danger/8',
    stroke: 'oklch(71% 0.17 18)',
  }
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
    <section className="glass rounded-2xl p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold">Resume match</h2>
        {hasResume && (
          <button
            type="button"
            onClick={() => compute.mutate()}
            disabled={compute.isPending}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border-subtle px-2.5 py-1.5 text-xs text-ink-muted transition hover:border-border-strong hover:text-ink disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw
              size={12}
              aria-hidden="true"
              className={compute.isPending ? 'animate-spin' : ''}
            />
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
        <p
          className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger"
          role="alert"
        >
          {compute.error.message}
        </p>
      )}

      {hasResume && !data && !compute.isPending && (
        <p className="mt-3 text-sm text-ink-faint">Not scored yet.</p>
      )}

      {data && (
        <div className="mt-4 space-y-4">
          <ScoreRing score={data.overall_score} />

          <dl className="space-y-2">
            {COMPONENTS.map(([key, label, weight]) => {
              const value = data.subscores?.[key]
              if (value === undefined) return null
              return (
                <div key={key} className="grid grid-cols-[1fr_auto] items-center gap-x-3 text-xs">
                  <dt className="text-ink-muted">
                    {label}
                    <span className="ml-1.5 font-mono text-[10px] text-ink-faint">{weight}%</span>
                  </dt>
                  <dd className="font-medium tabular-nums">{Math.round(value * 100)}%</dd>
                  <div className="col-span-2 mt-1 h-1 overflow-hidden rounded-full bg-canvas/60">
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
            <p className={`rounded-xl border p-3 text-sm ${tone(data.overall_score).wash}`}>
              {data.narrative}
            </p>
          )}

          {data.missing_skills?.length > 0 && (
            <div>
              <h3 className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
                Required but not in your resume
              </h3>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {data.missing_skills.map((skill) => (
                  <span
                    key={skill}
                    className="rounded-full bg-danger/12 px-2 py-0.5 text-xs text-danger ring-1 ring-danger/28 ring-inset"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {data.matched_skills?.length > 0 && (
            <div>
              <h3 className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
                Matched
              </h3>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {data.matched_skills.map((skill) => (
                  <span
                    key={skill}
                    className="rounded-full bg-positive/12 px-2 py-0.5 text-xs text-positive ring-1 ring-positive/28 ring-inset"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {data.gaps?.length > 0 && (
            <div>
              <h3 className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
                Gaps
              </h3>
              <ul className="mt-1.5 space-y-1.5 text-sm">
                {data.gaps.map((gap) => (
                  <li key={gap} className="flex gap-2 text-ink-muted">
                    <span
                      aria-hidden="true"
                      className="mt-2 size-1 shrink-0 rounded-full bg-ink-faint"
                    />
                    <span className="min-w-0">{gap}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!data.model && (
            <p className="text-xs text-ink-faint">
              Scored without the evidence review — the model was unavailable, so this reflects
              the other 85% only.
            </p>
          )}
        </div>
      )}
    </section>
  )
}

/**
 * The overall score, as a ring.
 *
 * A bare number out of 100 makes you do the division; the arc does it for you,
 * and the band colour is the same one the subscore bars and the narrative use,
 * so the panel says one thing three times rather than three things once. The
 * number is still there, and still the thing a screen reader reads — the ring
 * is `aria-hidden` scenery around it.
 */
const RADIUS = 30
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

function ScoreRing({ score }) {
  const { text, stroke } = tone(score)

  return (
    <div className="flex items-center gap-4">
      <div className="relative size-[72px] shrink-0">
        <svg viewBox="0 0 72 72" className="size-full -rotate-90" aria-hidden="true">
          <circle
            cx="36"
            cy="36"
            r={RADIUS}
            fill="none"
            strokeWidth="6"
            className="stroke-canvas/70"
          />
          <circle
            cx="36"
            cy="36"
            r={RADIUS}
            fill="none"
            strokeWidth="6"
            strokeLinecap="round"
            stroke={stroke}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE * (1 - Math.min(100, Math.max(0, score)) / 100)}
            style={{ transition: 'stroke-dashoffset 700ms cubic-bezier(0.16, 1, 0.3, 1)' }}
          />
        </svg>
        <span
          className={`absolute inset-0 grid place-items-center font-mono text-xl font-medium tabular-nums ${text}`}
        >
          {score}
        </span>
      </div>
      <p className="text-xs text-ink-faint">
        out of 100, weighted
        <br />
        across the five parts below
      </p>
    </div>
  )
}
