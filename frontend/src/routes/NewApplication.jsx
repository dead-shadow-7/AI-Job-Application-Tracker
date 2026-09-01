import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '@/lib/api'

const FIELD =
  'w-full rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

function Field({ label, hint, children }) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-sm font-medium">{label}</span>
      {children}
      {hint && <span className="block text-xs text-ink-muted">{hint}</span>}
    </label>
  )
}

export function NewApplication() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const skills = useQuery({ queryKey: ['skills'], queryFn: () => api.listSkills({ limit: 500 }) })

  const [form, setForm] = useState({
    company_name: '',
    title: '',
    url: '',
    source_platform: '',
    location: '',
    work_mode: '',
    seniority: '',
    employment_type: '',
    salary_min: '',
    salary_max: '',
    salary_currency: 'INR',
    salary_period: 'year',
    description: '',
    initial_event: 'applied',
    applied_on: new Date().toISOString().slice(0, 10),
    must_haves: '',
    nice_to_haves: '',
  })
  const [selectedSkills, setSelectedSkills] = useState([])
  const [skillFilter, setSkillFilter] = useState('')

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))

  const create = useMutation({
    mutationFn: (body) => api.createApplication(body),
    onSuccess: (application) => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      navigate(`/applications/${application.id}`)
    },
  })

  function submit(event) {
    event.preventDefault()

    const lines = (text) =>
      text
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean)

    const hasSalary = form.salary_min || form.salary_max

    create.mutate({
      // Backdate the initial event rather than creating it "now" and correcting
      // afterwards: the follow-up rules read occurred_at, so an application
      // sent last week must read as a week idle from the moment it is entered.
      initial_event: form.initial_event,
      occurred_at: applyDate(form.applied_on),
      job: {
        company_name: form.company_name.trim(),
        title: form.title.trim(),
        url: form.url.trim() || null,
        source_platform: form.source_platform.trim() || null,
        location: form.location.trim() || null,
        work_mode: form.work_mode || null,
        seniority: form.seniority || null,
        employment_type: form.employment_type || null,
        salary_min: form.salary_min || null,
        salary_max: form.salary_max || null,
        salary_currency: hasSalary ? form.salary_currency : null,
        salary_period: hasSalary ? form.salary_period : null,
        description: form.description.trim() || null,
        requirements: [
          ...lines(form.must_haves).map((text) => ({ text, kind: 'must' })),
          ...lines(form.nice_to_haves).map((text) => ({ text, kind: 'nice' })),
        ],
        skill_slugs: selectedSkills,
      },
    })
  }

  const visibleSkills = (skills.data ?? []).filter((s) =>
    skillFilter ? s.name.toLowerCase().includes(skillFilter.toLowerCase()) : true,
  )

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <Link to="/" className="text-sm text-ink-muted hover:text-accent">
          ← Applications
        </Link>
        <h1 className="mt-2 text-lg font-semibold tracking-tight">Track a job</h1>
        <p className="mt-0.5 text-sm text-ink-muted">
          Entered by hand for now. Phase 2 fills all of this from a pasted job description.
        </p>
      </div>

      <form onSubmit={submit} className="space-y-6">
        <section className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Company">
              <input required value={form.company_name} onChange={set('company_name')} className={FIELD} placeholder="Amazon" />
            </Field>
            <Field label="Role">
              <input required value={form.title} onChange={set('title')} className={FIELD} placeholder="Backend Engineer" />
            </Field>
          </div>

          <Field label="Job URL">
            <input type="url" value={form.url} onChange={set('url')} className={FIELD} placeholder="https://…" />
          </Field>

          <div className="grid gap-4 sm:grid-cols-3">
            <Field label="Platform">
              <input value={form.source_platform} onChange={set('source_platform')} className={FIELD} placeholder="LinkedIn" />
            </Field>
            <Field label="Location">
              <input value={form.location} onChange={set('location')} className={FIELD} placeholder="Pune" />
            </Field>
            <Field label="Work mode">
              <select value={form.work_mode} onChange={set('work_mode')} className={FIELD}>
                <option value="">—</option>
                <option value="remote">Remote</option>
                <option value="hybrid">Hybrid</option>
                <option value="onsite">On-site</option>
              </select>
            </Field>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Seniority">
              <select value={form.seniority} onChange={set('seniority')} className={FIELD}>
                <option value="">—</option>
                {['intern', 'junior', 'mid', 'senior', 'staff', 'lead', 'principal'].map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </Field>
            <Field label="Employment type">
              <select value={form.employment_type} onChange={set('employment_type')} className={FIELD}>
                <option value="">—</option>
                {['full_time', 'part_time', 'contract', 'internship'].map((v) => (
                  <option key={v} value={v}>{v.replace('_', ' ')}</option>
                ))}
              </select>
            </Field>
          </div>
        </section>

        <section className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
          <h2 className="text-sm font-medium">Compensation</h2>
          <div className="grid gap-4 sm:grid-cols-4">
            <Field label="Min">
              <input type="number" min="0" value={form.salary_min} onChange={set('salary_min')} className={FIELD} placeholder="1800000" />
            </Field>
            <Field label="Max">
              <input type="number" min="0" value={form.salary_max} onChange={set('salary_max')} className={FIELD} placeholder="2400000" />
            </Field>
            <Field label="Currency">
              <select value={form.salary_currency} onChange={set('salary_currency')} className={FIELD}>
                {['INR', 'USD', 'EUR', 'GBP'].map((c) => <option key={c}>{c}</option>)}
              </select>
            </Field>
            <Field label="Per">
              <select value={form.salary_period} onChange={set('salary_period')} className={FIELD}>
                <option value="year">year</option>
                <option value="month">month</option>
                <option value="hour">hour</option>
              </select>
            </Field>
          </div>
          <p className="text-xs text-ink-muted">
            Leave blank if the posting did not say. A guessed number is worse than none.
          </p>
        </section>

        <section className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
          <h2 className="text-sm font-medium">Requirements</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Must-haves" hint="One per line">
              <textarea rows={4} value={form.must_haves} onChange={set('must_haves')} className={FIELD} placeholder={'3+ years Python\nStrong SQL'} />
            </Field>
            <Field label="Nice-to-haves" hint="One per line">
              <textarea rows={4} value={form.nice_to_haves} onChange={set('nice_to_haves')} className={FIELD} placeholder="Kubernetes" />
            </Field>
          </div>

          <Field label="Skills" hint={`${selectedSkills.length} selected`}>
            <input
              value={skillFilter}
              onChange={(e) => setSkillFilter(e.target.value)}
              className={FIELD}
              placeholder="Filter skills…"
            />
          </Field>
          <div className="max-h-44 overflow-y-auto rounded-lg border border-border-subtle p-2">
            <div className="flex flex-wrap gap-1.5">
              {visibleSkills.map((skill) => {
                const on = selectedSkills.includes(skill.slug)
                return (
                  <button
                    key={skill.id}
                    type="button"
                    aria-pressed={on}
                    onClick={() =>
                      setSelectedSkills((s) =>
                        on ? s.filter((x) => x !== skill.slug) : [...s, skill.slug],
                      )
                    }
                    className={`rounded-full px-2.5 py-1 text-xs ring-1 ring-inset transition ${
                      on
                        ? 'bg-accent text-white ring-accent'
                        : 'bg-surface text-ink-muted ring-border-subtle hover:bg-surface-muted'
                    }`}
                  >
                    {skill.name}
                  </button>
                )
              })}
            </div>
          </div>

          <Field label="Job description" hint="Optional now; Phase 2 extracts everything above from this.">
            <textarea rows={5} value={form.description} onChange={set('description')} className={FIELD} />
          </Field>
        </section>

        <section className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5">
          <h2 className="text-sm font-medium">Status</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Starting point">
              <select value={form.initial_event} onChange={set('initial_event')} className={FIELD}>
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
                value={form.applied_on}
                max={new Date().toISOString().slice(0, 10)}
                onChange={set('applied_on')}
                className={FIELD}
              />
            </Field>
          </div>
        </section>

        {create.error && (
          <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700" role="alert">
            {create.error.message}
          </p>
        )}

        <div className="flex justify-end gap-3">
          <Link to="/" className="rounded-lg border border-border-subtle px-4 py-2 text-sm font-medium transition hover:bg-surface-muted">
            Cancel
          </Link>
          <button
            type="submit"
            disabled={create.isPending}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
          >
            {create.isPending ? 'Saving…' : 'Track this job'}
          </button>
        </div>
      </form>
    </div>
  )
}

/** A date input yields "2026-08-18" with no time. Stamping it at local midday
 *  keeps it on the intended day in every timezone, and safely in the past —
 *  the API rejects future-dated events. */
function applyDate(value) {
  if (!value) return null
  const date = new Date(`${value}T12:00:00`)
  return (date > new Date() ? new Date() : date).toISOString()
}
