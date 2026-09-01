import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '@/lib/api'
import { formatDate, formatSalary, relativeDays } from '@/lib/format'

/* Two records behind one panel. Everything about the *posting* lives on the job
   and is shared; priority and notes are yours alone and live on the
   application. The split is invisible here on purpose — you are editing "this
   job", not two tables — but it means a save can touch two endpoints. */
const JOB_FIELDS = [
  { key: 'title', label: 'Role title', type: 'text' },
  { key: 'location', label: 'Location', type: 'text' },
  {
    key: 'work_mode',
    label: 'Work mode',
    options: [
      ['onsite', 'On-site'],
      ['hybrid', 'Hybrid'],
      ['remote', 'Remote'],
    ],
  },
  {
    key: 'seniority',
    label: 'Seniority',
    options: [
      ['intern', 'Intern'],
      ['junior', 'Junior'],
      ['mid', 'Mid'],
      ['senior', 'Senior'],
      ['staff', 'Staff'],
      ['lead', 'Lead'],
      ['principal', 'Principal'],
    ],
  },
  {
    key: 'employment_type',
    label: 'Employment',
    options: [
      ['full_time', 'Full time'],
      ['part_time', 'Part time'],
      ['contract', 'Contract'],
      ['internship', 'Internship'],
    ],
  },
  { half: true, key: 'salary_min', label: 'Salary from', type: 'number' },
  { half: true, key: 'salary_max', label: 'Salary to', type: 'number' },
  {
    half: true,
    key: 'salary_currency',
    label: 'Currency',
    options: [
      ['INR', 'INR'],
      ['USD', 'USD'],
      ['EUR', 'EUR'],
      ['GBP', 'GBP'],
    ],
  },
  {
    half: true,
    key: 'salary_period',
    label: 'Per',
    options: [
      ['year', 'Year'],
      ['month', 'Month'],
      ['hour', 'Hour'],
    ],
  },
  { half: true, key: 'years_experience_min', label: 'Years from', type: 'number' },
  { half: true, key: 'years_experience_max', label: 'Years to', type: 'number' },
  { key: 'source_platform', label: 'Found on', type: 'text' },
  { key: 'url', label: 'Posting link', type: 'text' },
]

const APPLICATION_FIELDS = [
  {
    key: 'priority',
    label: 'Priority',
    options: [
      ['low', 'Low'],
      ['medium', 'Medium'],
      ['high', 'High'],
    ],
  },
  { key: 'notes', label: 'Your notes', type: 'textarea' },
]

const INPUT =
  'w-full rounded-lg border border-border-subtle px-2.5 py-1.5 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

/**
 * The details of one application, readable and editable.
 *
 * Editable because extraction is not the only way a record gets filled in. A
 * posting that never states a salary, a job added by hand or through the
 * assistant, a title the extractor trimmed too far — all of them leave gaps
 * that only the person applying can close, and a dash you cannot click is a
 * dead end.
 */
export function DetailsPanel({ application }) {
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState({})
  const [error, setError] = useState(null)

  const job = application.job

  function open() {
    setDraft({
      ...Object.fromEntries(JOB_FIELDS.map((f) => [f.key, job[f.key] ?? ''])),
      priority: application.priority ?? 'medium',
      notes: application.notes ?? '',
    })
    setError(null)
    setEditing(true)
  }

  const save = useMutation({
    mutationFn: async () => {
      /* Only what changed. Both endpoints apply exactly the keys they are sent,
         so posting the whole form would rewrite untouched fields with whatever
         the form happened to render — and clearing a field is a real edit, not
         an accident, so an explicit null still has to get through. */
      const jobChanges = diff(JOB_FIELDS, draft, job)
      const appChanges = diff(APPLICATION_FIELDS, draft, application)

      if (Object.keys(jobChanges).length) await api.updateJob(job.id, jobChanges)
      if (Object.keys(appChanges).length) await api.updateApplication(application.id, appChanges)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['application', application.id] })
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      setEditing(false)
    },
    onError: (e) => setError(e.message),
  })

  if (!editing) {
    return (
      <section className="rounded-xl border border-border-subtle bg-surface p-5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-medium">Details</h2>
          <button
            type="button"
            onClick={open}
            className="rounded-lg border border-border-subtle px-2.5 py-1 text-xs transition hover:bg-surface-muted"
          >
            Edit
          </button>
        </div>

        <dl className="mt-3 space-y-2.5 text-sm">
          <Row label="Applied" value={formatDate(application.applied_at)} />
          <Row label="Last activity" value={relativeDays(application.last_activity_at)} />
          <Row label="Salary" value={formatSalary(job)} />
          <Row label="Location" value={job.location} />
          <Row label="Seniority" value={job.seniority} />
          <Row
            label="Experience"
            value={job.years_experience_min != null ? `${job.years_experience_min}+ years` : null}
          />
          <Row label="Source" value={job.source_platform} />
          <Row label="Priority" value={application.priority} />
        </dl>

        {application.notes && (
          <p className="mt-3 whitespace-pre-wrap border-t border-border-subtle pt-3 text-sm text-ink-muted">
            {application.notes}
          </p>
        )}
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-border-subtle bg-surface p-5">
      <h2 className="text-sm font-medium">Edit details</h2>
      <p className="mt-0.5 text-xs text-ink-muted">
        Blank clears a field. Salary and skills from a pasted posting were checked against its
        text; anything you type here is taken as given.
      </p>

      <form
        className="mt-4 grid grid-cols-2 gap-3"
        onSubmit={(e) => {
          e.preventDefault()
          save.mutate()
        }}
      >
        {/* Paired where the values are naturally short — a from/to range reads
            as one field, and fifteen full-width boxes in a narrow column is a
            lot of scrolling for what is mostly two numbers. */}
        {[...JOB_FIELDS, ...APPLICATION_FIELDS].map((field) => (
          <label key={field.key} className={field.half ? 'block' : 'col-span-2 block'}>
            <span className="text-xs text-ink-muted">{field.label}</span>
            {field.options ? (
              <select
                value={draft[field.key] ?? ''}
                onChange={(e) => setDraft((d) => ({ ...d, [field.key]: e.target.value }))}
                className={`${INPUT} mt-1`}
              >
                <option value="">—</option>
                {field.options.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            ) : field.type === 'textarea' ? (
              <textarea
                rows={3}
                value={draft[field.key] ?? ''}
                onChange={(e) => setDraft((d) => ({ ...d, [field.key]: e.target.value }))}
                className={`${INPUT} mt-1`}
              />
            ) : (
              <input
                type={field.type}
                inputMode={field.type === 'number' ? 'decimal' : undefined}
                value={draft[field.key] ?? ''}
                onChange={(e) => setDraft((d) => ({ ...d, [field.key]: e.target.value }))}
                className={`${INPUT} mt-1`}
              />
            )}
          </label>
        ))}

        {error && (
          <p className="col-span-2 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700" role="alert">
            {error}
          </p>
        )}

        <div className="col-span-2 flex gap-2 pt-1">
          <button
            type="submit"
            disabled={save.isPending}
            className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
          >
            {save.isPending ? 'Saving…' : 'Save'}
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="rounded-lg border border-border-subtle px-3 py-1.5 text-sm transition hover:bg-surface-muted"
          >
            Cancel
          </button>
        </div>
      </form>
    </section>
  )
}

/** Changed fields only, normalised for the API. */
function diff(fields, draft, original) {
  const changes = {}
  for (const { key, type } of fields) {
    const typed = draft[key]
    // An empty box means "no value", which the API expresses as null. Sending
    // "" instead fails validation on the currency field and stores a blank
    // string everywhere else, which reads as a value but is not one.
    const next = typed === '' || typed == null ? null : type === 'number' ? Number(typed) : typed
    const before = original[key] ?? null

    if (next === null && before === null) continue
    // Numeric columns come back as strings ("4500000.00"), so compare by value
    // or every save would rewrite the salary it just read.
    if (next != null && before != null && Number(next) === Number(before)) continue
    if (String(next) === String(before)) continue

    changes[key] = next
  }
  return changes
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-ink-muted">{label}</dt>
      <dd className="text-right font-medium">{value || '—'}</dd>
    </div>
  )
}
