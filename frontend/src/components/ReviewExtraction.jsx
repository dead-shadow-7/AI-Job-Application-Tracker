import { ArrowLeft, Check, CopyCheck, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { formatSalary } from '@/lib/format'

const FIELD =
  'well w-full rounded-xl px-3 py-2.5 text-sm outline-none transition placeholder:text-ink-faint focus:border-accent/40 focus:shadow-[0_0_0_3px] focus:shadow-accent/12 focus-visible:outline-none'

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
    <div className="mx-auto max-w-3xl space-y-5">
      <div>
        <button
          type="button"
          onClick={onBack}
          className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-ink-faint transition hover:text-accent"
        >
          <ArrowLeft size={13} aria-hidden="true" />
          Paste a different posting
        </button>
        <h1 className="mt-2 font-display text-2xl leading-tight font-semibold tracking-tight">
          Review before saving
        </h1>
        <p className="mt-1 text-sm text-ink-muted">
          Extracted by <span className="font-mono text-xs">{preview.model}</span> in{' '}
          {(preview.latency_ms / 1000).toFixed(1)}s using {preview.tokens_used.toLocaleString()}{' '}
          tokens. Nothing has been saved yet.
        </p>
      </div>

      {/* Two different claims, deliberately worded differently. An identical
          description is a fact; a near match is the embedding's judgement, and
          two genuinely separate openings at one company would trip it. Saving
          stays enabled either way — this warns, it does not block. */}
      {preview.duplicate_of && (
        <div className="flex gap-3 rounded-2xl border border-signal/30 bg-signal/8 px-4 py-3.5">
          <CopyCheck size={17} aria-hidden="true" className="mt-0.5 shrink-0 text-signal" />
          <p className="text-sm text-signal">
            {preview.duplicate_of.is_exact
              ? 'You already track this exact posting: '
              : 'This looks like a posting you already track: '}
            <Link
              to={`/applications/${preview.duplicate_of.application_id}`}
              className="font-semibold underline underline-offset-2"
            >
              {preview.duplicate_of.label}
            </Link>
            {preview.duplicate_of.is_exact
              ? '. Open it instead of starting a second timeline.'
              : '. Check before saving — if it is a different opening, carry on.'}
          </p>
        </div>
      )}

      {preview.warnings.length > 0 && (
        <div className="rounded-2xl border border-signal/30 bg-signal/8 p-4">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-signal">
            <TriangleAlert size={15} aria-hidden="true" className="shrink-0" />
            {preview.warnings.length === 1
              ? 'One thing to check'
              : `${preview.warnings.length} things to check`}
          </h2>
          <ul className="mt-2.5 space-y-1.5 text-sm text-signal/85">
            {preview.warnings.map((warning) => (
              <li key={warning} className="flex gap-2.5">
                <span
                  aria-hidden="true"
                  className="mt-2 size-1 shrink-0 rounded-full bg-signal/60"
                />
                <span className="min-w-0">{warning}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="glass space-y-4 rounded-2xl p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">The role</h2>
          <ConfidenceBadge value={Number(preview.confidence)} />
        </div>

        {/* Highlighted when absent rather than merely blank. A posting that
            never names its employer is common — on LinkedIn the company sits
            above the description, not inside it — and the model is told not to
            guess, so this is the one thing it routinely cannot supply. */}
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Company" required missing={!job.company_name}>
            <input
              required
              value={job.company_name ?? ''}
              onChange={set('company_name')}
              placeholder="Not named in the posting — add it"
              className={`${FIELD} ${!job.company_name ? 'border-signal/50 bg-signal/8' : ''}`}
            />
          </Field>
          <Field label="Role" required missing={!job.title}>
            <input
              required
              value={job.title ?? ''}
              onChange={set('title')}
              placeholder="Not found — add it"
              className={`${FIELD} ${!job.title ? 'border-signal/50 bg-signal/8' : ''}`}
            />
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

      <div className={`glass space-y-4 rounded-2xl p-5 ${salaryDropped ? 'border-signal/40' : ''}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Compensation</h2>
          {salaryPreview && (
            <span className="text-sm text-ink-muted tabular-nums">{salaryPreview}</span>
          )}
        </div>

        {salaryDropped && (
          <p className="rounded-xl border border-signal/25 bg-signal/8 px-3 py-2.5 text-xs text-signal">
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

      <div className="glass space-y-4 rounded-2xl p-5">
        <h2 className="text-sm font-semibold">
          Requirements
          <span className="ml-2 font-mono text-xs font-normal text-ink-faint">
            {job.requirements.filter((r) => r.kind === 'must').length} must,{' '}
            {job.requirements.filter((r) => r.kind === 'nice').length} nice
          </span>
        </h2>
        <ul className="space-y-1.5 text-sm">
          {job.requirements.map((req, index) => (
            <li key={`${req.kind}-${index}`} className="flex items-start gap-2">
              <span
                className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  req.kind === 'must'
                    ? 'bg-accent/14 text-accent ring-1 ring-accent/25 ring-inset'
                    : 'bg-surface-muted/70 text-ink-faint'
                }`}
              >
                {req.kind}
              </span>
              <span className="flex-1">{req.text}</span>
            </li>
          ))}
          {job.requirements.length === 0 && (
            <li className="text-ink-faint">None found in the posting.</li>
          )}
        </ul>

        <div>
          <h3 className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
            Skills matched to the taxonomy ({job.skill_slugs.length})
          </h3>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {job.skill_slugs.map((slug) => (
              <span
                key={slug}
                className="rounded-full bg-surface-muted/70 px-2.5 py-1 text-xs text-ink-muted ring-1 ring-border-subtle/60 ring-inset"
              >
                {slug}
              </span>
            ))}
            {job.skill_slugs.length === 0 && <span className="text-sm text-ink-faint">None.</span>}
          </div>
        </div>

        {preview.unmatched_skills.length > 0 && (
          <p className="text-xs text-ink-faint">
            Not in the taxonomy, so not attached:{' '}
            <span className="font-medium">{preview.unmatched_skills.join(', ')}</span>. New skills
            are added deliberately rather than automatically, so a typo cannot fragment scoring
            later.
          </p>
        )}
      </div>

      <div className="glass space-y-4 rounded-2xl p-5">
        <h2 className="text-sm font-semibold">Status</h2>
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
        <p
          className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
          role="alert"
        >
          {error.message}
        </p>
      )}

      <div className="flex justify-end gap-3">
        <button
          type="button"
          onClick={onBack}
          className="cursor-pointer rounded-xl border border-border-subtle px-4 py-2.5 text-sm font-medium text-ink-muted transition hover:border-border-strong hover:text-ink"
        >
          Discard
        </button>
        <button
          type="button"
          disabled={saving || !job.company_name?.trim() || !job.title?.trim()}
          title={
            !job.company_name?.trim() || !job.title?.trim()
              ? 'Company and role are required to save'
              : undefined
          }
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
          className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Check size={16} aria-hidden="true" />
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
      className={`rounded-full px-2 py-0.5 font-mono text-xs font-medium ring-1 ring-inset tabular-nums ${
        high
          ? 'bg-accent/14 text-accent ring-accent/30'
          : 'bg-signal/14 text-signal ring-signal/30'
      }`}
      title="The model's own estimate of how complete and unambiguous the posting was"
    >
      {Math.round(value * 100)}% confident
    </span>
  )
}

function Field({ label, hint, children, required, missing }) {
  return (
    <label className="block space-y-2">
      <span className="block text-sm font-medium">
        {label}
        {required && (
          <span className="ml-1 text-danger" aria-label="required">
            *
          </span>
        )}
        {missing && (
          <span className="ml-2 text-xs font-normal text-signal">needs your input</span>
        )}
      </span>
      {children}
      {hint && <span className="block text-xs text-ink-faint">{hint}</span>}
    </label>
  )
}

function toIso(value) {
  if (!value) return null
  const date = new Date(`${value}T12:00:00`)
  return (date > new Date() ? new Date() : date).toISOString()
}
