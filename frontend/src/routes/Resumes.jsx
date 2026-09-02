import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw, Star, Trash2, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { ErrorState } from '@/components/ErrorState'
import { PageHeader } from '@/components/PageHeader'
import { Spinner } from '@/components/Spinner'
import { api } from '@/lib/api'
import { formatDate, formatMonth } from '@/lib/format'

const FIELD =
  'well w-full rounded-xl px-3 py-2.5 text-sm outline-none transition placeholder:text-ink-faint focus:border-accent/40 focus:shadow-[0_0_0_3px] focus:shadow-accent/12 focus-visible:outline-none'

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
    <div className="mx-auto max-w-3xl space-y-5">
      <PageHeader
        back={{ to: '/', label: 'Applications' }}
        title="Resume"
        subtitle="Used to score how well each job fits you. Processed entirely on your own machine — the text is never sent to any model provider."
      />

      <form
        onSubmit={(e) => {
          e.preventDefault()
          upload.mutate()
        }}
        className="glass space-y-4 rounded-2xl p-5"
      >
        <div className="well flex gap-1 rounded-xl p-1 text-sm">
          {[
            ['file', 'Upload a file'],
            ['text', 'Paste text'],
          ].map(([value, text]) => (
            <button
              key={value}
              type="button"
              onClick={() => setMode(value)}
              aria-pressed={mode === value}
              className={`flex-1 cursor-pointer rounded-lg px-3 py-2 transition ${
                mode === value
                  ? 'bg-accent/15 font-medium text-accent'
                  : 'text-ink-faint hover:text-ink'
              }`}
            >
              {text}
            </button>
          ))}
        </div>

        {mode === 'file' ? (
          <label className="block space-y-2">
            <span className="block text-sm font-medium">PDF, DOCX or text file</span>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.md"
              required
              className="w-full cursor-pointer text-sm text-ink-muted file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-accent/15 file:px-3.5 file:py-2 file:text-sm file:font-medium file:text-accent hover:file:bg-accent/25"
            />
            <span className="block text-xs text-ink-faint">
              A scanned PDF will not work — its words are an image. Paste the text instead.
            </span>
          </label>
        ) : (
          <label className="block space-y-2">
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

        <label className="block space-y-2">
          <span className="block text-sm font-medium">Label</span>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Backend-focused, ML-focused…"
            className={FIELD}
          />
          <span className="block text-xs text-ink-faint">
            Keep several versions and score jobs against whichever fits.
          </span>
        </label>

        {upload.error && (
          <p
            className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
            role="alert"
          >
            {upload.error.message}
          </p>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={upload.isPending}
            className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Upload size={16} aria-hidden="true" />
            {upload.isPending ? 'Reading and embedding…' : 'Save resume'}
          </button>
        </div>
      </form>

      {resumes.isPending && <Spinner label="Loading resumes" />}
      {resumes.error && <ErrorState error={resumes.error} onRetry={resumes.refetch} />}

      {resumes.data?.length === 0 && (
        <p className="glass rounded-2xl border-dashed p-10 text-center text-sm text-ink-faint">
          No resume yet. Add one to start scoring jobs.
        </p>
      )}

      {resumes.data?.length > 0 && (
        <ul className="glass divide-y divide-border-subtle/50 overflow-hidden rounded-2xl">
          {resumes.data.map((resume) => (
            <li key={resume.id} className="px-4 py-4">
              <div className="flex flex-wrap items-center gap-3">
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-2 text-sm font-medium">
                    {resume.label}
                    {resume.is_default && (
                      <span className="rounded-full bg-accent/14 px-2 py-0.5 text-xs font-medium text-accent ring-1 ring-accent/30 ring-inset">
                        default
                      </span>
                    )}
                  </p>
                  <p className="mt-1 text-xs text-ink-faint">
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
                    className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border-subtle px-2.5 py-1.5 text-xs text-ink-muted transition hover:border-border-strong hover:text-ink"
                  >
                    <Star size={12} aria-hidden="true" />
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
                  className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border-subtle px-2.5 py-1.5 text-xs text-ink-muted transition hover:border-border-strong hover:text-ink disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RefreshCw
                    size={12}
                    aria-hidden="true"
                    className={
                      reparse.isPending && reparse.variables === resume.id ? 'animate-spin' : ''
                    }
                  />
                  {reparse.isPending && reparse.variables === resume.id ? 'Re-reading…' : 'Re-parse'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (confirm(`Delete "${resume.label}"? Scores based on it are discarded too.`)) {
                      remove.mutate(resume.id)
                    }
                  }}
                  className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-danger/30 px-2.5 py-1.5 text-xs text-danger transition hover:bg-danger/10"
                >
                  <Trash2 size={12} aria-hidden="true" />
                  Delete
                </button>
              </div>

              {/* What the parser actually read. Seniority and the experience
                  subscore are computed from these, so a wrong row here is worth
                  seeing rather than discovering as an unexplained score. */}
              {rolesOf(resume).length > 0 && (
                <ul className="mt-3 space-y-1.5 border-l-2 border-accent/25 pl-3 text-xs text-ink-faint">
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
