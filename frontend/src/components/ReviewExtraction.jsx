import { useState } from 'react'
import { formatSalary } from '@/lib/format'

const FIELD =
  'w-full rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

/**
 * Review before save.
 *
 * Nothing is written until the user confirms here. Extraction is good but not
 * perfect, and a wrong row saved silently is far more expensive to find later
 * than an edit made now — especially salary, which is the field most likely to
 * be wrong and least likely to be re-checked once it is in the table.
 *
 * Fields the backend discarded as unverifiable are highlighted rather than
 * hidden: the user knows what the posting said and can supply it, whereas a
 * blank with no explanation just looks like the extractor failed.
 */
export function ReviewExtraction({ preview, onBack, onSave, saving, error }) {
  const [job, setJob] = useState(preview.job)
  const [initialEvent, setInitialEvent] = useState('applied')
  const [occurredOn, setOccurredOn] = useState(new Date().toISOString().slice(0, 10))

  const set = (key) => (e) => {
    const value = e.target.value
    setJob((j) => ({ ...j, [key]: value === '' ? null : value }))
  }

  const salaryDropped = preview.dropped_fields.includes('salary')
  const salaryPreview = formatSalary(job)

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <button type="button" onClick={onBack} className="text-sm text-ink-muted hover:text-accent">
          ← Paste a different posting
        </button>
        <h1 className="mt-2 text-lg font-semibold tracking-tight">Review before saving</h1>
        <p className="mt-0.5 text-sm text-ink-muted">
          Extracted by {preview.model} in {(preview.latency_ms / 1000).toFixed(1)}s using{' '}
          {preview.tokens_used.toLocaleString()} tokens. Nothing has been saved yet.
        </p>
      </div>

      {preview.duplicate_of && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-sm text-amber-900">
            You already track a job with this exact description.{' '}
            <a href={`/applications/${preview.duplicate_of}`} className="font-medium underline">
              Open it
            </a>{' '}
            instead of creating a duplicate.
          </p>
        </div>
      )}

      {preview.warnings.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
          <h2 className="text-sm font-medium text-amber-900">
            {preview.warnings.length === 1 ? 'One thing to check' : `${preview.warnings.length} things to check`}
          </h2>
          <ul className="mt-2 space-y-1 text-sm text-amber-800">
            {preview.warnings.map((warning) => (
              <li key={warning} className="flex gap-2">
                <span aria-hidden="true">·</span>
                {warning}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium">The role</h2>
          <ConfidenceBadge value={Number(preview.confidence)} />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Company">
            <input value={job.company_name ?? ''} onChange={set('company_name')} className={FIELD} />
          </Field>
          <Field label="Role">
            <input value={job.title ?? ''} onChange={set('title')} className={FIELD} />
          </Field>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Location">
            <input value={job.location ?? ''} onChange={set('location')} className={FIELD} />
          </Field>
          <Field label="Work mode">
            <select value={job.work_mode ?? ''} onChange={set('work_mode')} className={FIELD}>
              <option value="">—</option>
              <option value="remote">Remote</option>
              <option value="hybrid">Hybrid</option>
              <option value="onsite">On-site</option>
            </select>
          </Field>
          <Field label="Seniority">
            <select value={job.seniority ?? ''} onChange={set('seniority')} className={FIELD}>
              <option value="">—</option>
              {['intern', 'junior', 'mid', 'senior', 'staff', 'lead', 'principal'].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </Field>
        </div>
      </div>

      <div
        className={`space-y-4 rounded-xl border bg-surface p-5 ${
          salaryDropped ? 'border-amber-300' : 'border-border-subtle'
        }`}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium">Compensation</h2>
          {salaryPreview && <span className="text-sm tabular-nums text-ink-muted">{salaryPreview}</span>}
        </div>

        {salaryDropped && (
          <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
            A salary was returned but could not be found in the posting, so it was discarded
            rather than saved. If the posting does state one, enter it here.
          </p>
        )}

        <div className="grid gap-4 sm:grid-cols-4">
          <Field label="Min">
            <input type="number" min="0" value={job.salary_min ?? ''} onChange={set('salary_min')} className={FIELD} />
          </Field>
          <Field label="Max">
            <input type="number" min="0" value={job.salary_max ?? ''} onChange={set('salary_max')} className={FIELD} />
          </Field>
          <Field label="Currency">
            <select value={job.salary_currency ?? ''} onChange={set('salary_currency')} className={FIELD}>
              <option value="">—</option>
              {['INR', 'USD', 'EUR', 'GBP'].map((c) => <option key={c}>{c}</option>)}
            </select>
          </Field>
          <Field label="Per">
            <select value={job.salary_period ?? ''} onChange={set('salary_period')} className={FIELD}>
              <option value="">—</option>
              <option value="year">year</option>
              <option value="month">month</option>
              <option value="hour">hour</option>
            </select>
          </Field>
        </div>
      </div>

      <div className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
        <h2 className="text-sm font-medium">
          Requirements
          <span className="ml-2 font-normal text-ink-muted">
            {job.requirements.filter((r) => r.kind === 'must').length} must,{' '}
            {job.requirements.filter((r) => r.kind === 'nice').length} nice
          </span>
        </h2>
        <ul className="space-y-1.5 text-sm">
          {job.requirements.map((req, index) => (
            <li key={`${req.kind}-${index}`} className="flex items-start gap-2">
              <span
                className={`mt-0.5 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  req.kind === 'must'
                    ? 'bg-slate-100 text-slate-700'
                    : 'bg-surface-muted text-ink-muted'
                }`}
              >
                {req.kind}
              </span>
              <span className="flex-1">{req.text}</span>
            </li>
          ))}
          {job.requirements.length === 0 && (
            <li className="text-ink-muted">None found in the posting.</li>
          )}
        </ul>

        <div>
          <h3 className="text-xs font-medium text-ink-muted">
            Skills matched to the taxonomy ({job.skill_slugs.length})
          </h3>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {job.skill_slugs.map((slug) => (
              <span key={slug} className="rounded-full bg-surface-muted px-2.5 py-1 text-xs">
                {slug}
              </span>
            ))}
            {job.skill_slugs.length === 0 && <span className="text-sm text-ink-muted">None.</span>}
          </div>
        </div>

        {preview.unmatched_skills.length > 0 && (
          <p className="text-xs text-ink-muted">
            Not in the taxonomy, so not attached:{' '}
            <span className="font-medium">{preview.unmatched_skills.join(', ')}</span>. New skills
            are added deliberately rather than automatically, so a typo cannot fragment scoring
            later.
          </p>
        )}
      </div>

      <div className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
        <h2 className="text-sm font-medium">Status</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Starting point">
            <select value={initialEvent} onChange={(e) => setInitialEvent(e.target.value)} className={FIELD}>
              <option value="applied">Applied</option>
              <option value="saved">Saved (not applied yet)</option>
              <option value="screening_scheduled">Screening scheduled</option>
              <option value="interview_scheduled">Interview scheduled</option>
              <option value="rejected">Rejected</option>
            </select>
          </Field>
          <Field label="On" hint="Backdate freely — this drives follow-up timing.">
            <input
              type="date"
              value={occurredOn}
              max={new Date().toISOString().slice(0, 10)}
              onChange={(e) => setOccurredOn(e.target.value)}
              className={FIELD}
            />
          </Field>
        </div>
      </div>

      {error && (
        <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700" role="alert">
          {error.message}
        </p>
      )}

      <div className="flex justify-end gap-3">
        <button
          type="button"
          onClick={onBack}
          className="rounded-lg border border-border-subtle px-4 py-2 text-sm font-medium transition hover:bg-surface-muted"
        >
          Discard
        </button>
        <button
          type="button"
          disabled={saving}
          onClick={() =>
            onSave({
              job: {
                ...job,
                salary_min: job.salary_min || null,
                salary_max: job.salary_max || null,
              },
              initialEvent,
              occurredAt: toIso(occurredOn),
            })
          }
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
        >
          {saving ? 'Saving…' : 'Save application'}
        </button>
      </div>
    </div>
  )
}

function ConfidenceBadge({ value }) {
  const high = value >= 0.75
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
        high
          ? 'bg-emerald-50 text-emerald-700 ring-emerald-200'
          : 'bg-amber-50 text-amber-800 ring-amber-200'
      }`}
      title="The model's own estimate of how complete and unambiguous the posting was"
    >
      {Math.round(value * 100)}% confident
    </span>
  )
}

function Field({ label, hint, children }) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-sm font-medium">{label}</span>
      {children}
      {hint && <span className="block text-xs text-ink-muted">{hint}</span>}
    </label>
  )
}

function toIso(value) {
  if (!value) return null
  const date = new Date(`${value}T12:00:00`)
  return (date > new Date() ? new Date() : date).toISOString()
}
