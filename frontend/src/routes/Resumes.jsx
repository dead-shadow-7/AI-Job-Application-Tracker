import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ErrorState } from '@/components/ErrorState'
import { Spinner } from '@/components/Spinner'
import { api } from '@/lib/api'
import { formatDate, formatMonth } from '@/lib/format'

const FIELD =
  'w-full rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

/** Guards against a response cached before the API returned positions at all —
 *  a resume list held over from a previous version would otherwise crash. */
const rolesOf = (resume) => resume.positions ?? []

/** Where the years figure came from. An inferred number is not a claimed one,
 *  and showing them identically would put words in the candidate's mouth. */
function yearsSummary({ years_experience, years_experience_source }) {
  if (years_experience == null) return null
  const years = `${Number(years_experience)} yrs`
  return years_experience_source === 'dates' ? `${years} from dates` : `${years} stated`
}

export function Resumes() {
  const queryClient = useQueryClient()
  const fileInput = useRef(null)
  const [mode, setMode] = useState('file')
  const [label, setLabel] = useState('')
  const [pasted, setPasted] = useState('')

  const resumes = useQuery({ queryKey: ['resumes'], queryFn: api.listResumes })

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['resumes'] })
    // Scores are computed against the default resume, so changing it makes
    // every cached score stale. A re-parse goes further and discards them
    // server-side, which clears the score denormalized onto each application.
    queryClient.invalidateQueries({ queryKey: ['match'] })
    queryClient.invalidateQueries({ queryKey: ['applications'] })
  }

  const upload = useMutation({
    mutationFn: () =>
      mode === 'file'
        ? api.uploadResumeFile(fileInput.current.files[0], label)
        : api.uploadResumeText({ label: label || 'Resume', text: pasted }),
    onSuccess: () => {
      invalidate()
      setLabel('')
      setPasted('')
      if (fileInput.current) fileInput.current.value = ''
    },
  })

  const makeDefault = useMutation({ mutationFn: api.setDefaultResume, onSuccess: invalidate })
  const remove = useMutation({ mutationFn: api.deleteResume, onSuccess: invalidate })
  const reparse = useMutation({ mutationFn: api.reparseResume, onSuccess: invalidate })

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <Link to="/" className="text-sm text-ink-muted hover:text-accent">
          ← Applications
        </Link>
        <h1 className="mt-2 text-lg font-semibold tracking-tight">Resume</h1>
        <p className="mt-0.5 text-sm text-ink-muted">
          Used to score how well each job fits you. Processed entirely on your own machine —
          the text is never sent to any model provider.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          upload.mutate()
        }}
        className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5"
      >
        <div className="flex gap-1 rounded-lg bg-surface-muted p-1 text-sm">
          {[
            ['file', 'Upload a file'],
            ['text', 'Paste text'],
          ].map(([value, text]) => (
            <button
              key={value}
              type="button"
              onClick={() => setMode(value)}
              aria-pressed={mode === value}
              className={`flex-1 rounded-md px-3 py-1.5 transition ${
                mode === value ? 'bg-surface font-medium shadow-sm' : 'text-ink-muted'
              }`}
            >
              {text}
            </button>
          ))}
        </div>

        {mode === 'file' ? (
          <label className="block space-y-1.5">
            <span className="block text-sm font-medium">PDF, DOCX or text file</span>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.md"
              required
              className="w-full text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-surface-muted file:px-3 file:py-2 file:text-sm file:font-medium"
            />
            <span className="block text-xs text-ink-muted">
              A scanned PDF will not work — its words are an image. Paste the text instead.
            </span>
          </label>
        ) : (
          <label className="block space-y-1.5">
            <span className="block text-sm font-medium">Resume text</span>
            <textarea
              rows={12}
              required
              minLength={200}
              value={pasted}
              onChange={(e) => setPasted(e.target.value)}
              placeholder="Paste your full resume…"
              className={`${FIELD} font-mono text-xs leading-relaxed`}
            />
          </label>
        )}

        <label className="block space-y-1.5">
          <span className="block text-sm font-medium">Label</span>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Backend-focused, ML-focused…"
            className={FIELD}
          />
          <span className="block text-xs text-ink-muted">
            Keep several versions and score jobs against whichever fits.
          </span>
        </label>

        {upload.error && (
          <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700" role="alert">
            {upload.error.message}
          </p>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={upload.isPending}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
          >
            {upload.isPending ? 'Reading and embedding…' : 'Save resume'}
          </button>
        </div>
      </form>

      {resumes.isPending && <Spinner label="Loading resumes" />}
      {resumes.error && <ErrorState error={resumes.error} onRetry={resumes.refetch} />}

      {resumes.data?.length === 0 && (
        <p className="rounded-xl border border-dashed border-border-subtle bg-surface p-8 text-center text-sm text-ink-muted">
          No resume yet. Add one to start scoring jobs.
        </p>
      )}

      {resumes.data?.length > 0 && (
        <ul className="divide-y divide-border-subtle rounded-xl border border-border-subtle bg-surface">
          {resumes.data.map((resume) => (
            <li key={resume.id} className="px-4 py-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-2 text-sm font-medium">
                    {resume.label}
                    {resume.is_default && (
                      <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200">
                        default
                      </span>
                    )}
                  </p>
                  <p className="mt-0.5 text-xs text-ink-muted">
                    {resume.chunk_count} passages
                    {rolesOf(resume).length > 0 &&
                      ` · ${rolesOf(resume).length} ${
                        rolesOf(resume).length === 1 ? 'role' : 'roles'
                      }`}
                    {yearsSummary(resume) && ` · ${yearsSummary(resume)}`}
                    {` · added ${formatDate(resume.created_at)}`}
                  </p>
                </div>

                {!resume.is_default && (
                  <button
                    type="button"
                    onClick={() => makeDefault.mutate(resume.id)}
                    className="rounded-lg border border-border-subtle px-3 py-1.5 text-xs transition hover:bg-surface-muted"
                  >
                    Make default
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    // Same warning as Delete carries, for the same reason: the
                    // discarded scores each cost a model call to produce again.
                    if (confirm('Re-read this resume? Scores based on it are discarded.')) {
                      reparse.mutate(resume.id)
                    }
                  }}
                  disabled={reparse.isPending}
                  title="Re-read the stored text with the current parser."
                  className="rounded-lg border border-border-subtle px-3 py-1.5 text-xs transition hover:bg-surface-muted disabled:opacity-60"
                >
                  {reparse.isPending && reparse.variables === resume.id ? 'Re-reading…' : 'Re-parse'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (confirm(`Delete "${resume.label}"? Scores based on it are discarded too.`)) {
                      remove.mutate(resume.id)
                    }
                  }}
                  className="rounded-lg border border-rose-200 px-3 py-1.5 text-xs text-rose-700 transition hover:bg-rose-50"
                >
                  Delete
                </button>
              </div>

              {/* What the parser actually read. Seniority and the experience
                  subscore are computed from these, so a wrong row here is worth
                  seeing rather than discovering as an unexplained score. */}
              {rolesOf(resume).length > 0 && (
                <ul className="mt-2 space-y-1 border-l-2 border-border-subtle pl-3 text-xs text-ink-muted">
                  {rolesOf(resume).map((position, index) => (
                    <li key={`${position.start}-${index}`} className="flex flex-wrap gap-x-2">
                      <span className="font-medium text-ink">
                        {position.title ?? 'Untitled role'}
                      </span>
                      {position.company && <span>{position.company}</span>}
                      <span>
                        {formatMonth(position.start)} – {formatMonth(position.end)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
