import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Pencil, X } from 'lucide-react'
import { useState } from 'react'
import { api } from '@/lib/api'

const FIELD =
  'well w-full rounded-xl px-3 py-2.5 text-sm outline-none transition placeholder:text-ink-faint focus:border-accent/40 focus:shadow-[0_0_0_3px] focus:shadow-accent/12 focus-visible:outline-none'

/**
 * What the role asks for — readable, and correctable.
 *
 * Extraction splits a bulleted list wherever the posting put a line break, so a
 * "vector databases such as FAISS, Pinecone, Chroma" bullet routinely arrives
 * as four separate requirements with three of them meaningless. That is not
 * worth another prompt revision, because the person reading it can fix it in
 * ten seconds — but only if the page lets them.
 *
 * Requirements are edited as plain text, one per line. A list of inputs with
 * add and remove buttons is more machinery for less: reordering means dragging,
 * deleting means aiming at a small target, and pasting five at once is
 * impossible. A textarea does all three with no controls at all.
 */
export function RolePanel({ job }) {
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [must, setMust] = useState('')
  const [nice, setNice] = useState('')
  const [description, setDescription] = useState('')
  const [skills, setSkills] = useState([])
  const [error, setError] = useState(null)

  // Only fetched once the editor is open — the taxonomy is a few hundred rows
  // and nobody browsing their timeline needs it.
  const taxonomy = useQuery({
    queryKey: ['skills'],
    queryFn: () => api.listSkills({ limit: 500 }),
    enabled: editing,
  })

  const mustHave = job.requirements.filter((r) => r.kind === 'must')
  const niceToHave = job.requirements.filter((r) => r.kind === 'nice')

  function open() {
    setMust(mustHave.map((r) => r.text).join('\n'))
    setNice(niceToHave.map((r) => r.text).join('\n'))
    setDescription(job.description ?? '')
    setSkills(job.skills.map(({ skill }) => skill.slug))
    setError(null)
    setEditing(true)
  }

  const save = useMutation({
    mutationFn: () =>
      api.updateJob(job.id, {
        requirements: [...toRequirements(must, 'must'), ...toRequirements(nice, 'nice')],
        skill_slugs: skills,
        description: description.trim() || null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['application'] })
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      setEditing(false)
    },
    onError: (e) => setError(e.message),
  })

  const hasContent = job.requirements.length > 0 || job.description || job.skills.length > 0

  if (!editing) {
    return (
      <section className="glass rounded-2xl p-5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">The role</h2>
          <button
            type="button"
            onClick={open}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border-subtle px-2.5 py-1.5 text-xs text-ink-muted transition hover:border-border-strong hover:text-ink"
          >
            <Pencil size={12} aria-hidden="true" />
            Edit
          </button>
        </div>

        {!hasContent && (
          <p className="mt-3 text-sm text-ink-faint">
            Nothing recorded about what this role wants. Add it here, or paste the posting to the
            assistant and it will extract the lot.
          </p>
        )}

        {job.requirements.length > 0 && (
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <RequirementList title="Must have" items={mustHave} />
            <RequirementList title="Nice to have" items={niceToHave} />
          </div>
        )}

        {job.skills.length > 0 && (
          <div className="mt-4">
            <h3 className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
              Skills
            </h3>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {job.skills.map(({ skill }) => (
                <span
                  key={skill.id}
                  className="rounded-full bg-surface-muted/70 px-2.5 py-1 text-xs text-ink-muted ring-1 ring-border-subtle/60 ring-inset"
                >
                  {skill.name}
                </span>
              ))}
            </div>
          </div>
        )}

        {job.description && (
          <p className="mt-4 border-t border-border-subtle/70 pt-4 text-sm leading-relaxed whitespace-pre-wrap text-ink-muted">
            {job.description}
          </p>
        )}
      </section>
    )
  }

  const known = taxonomy.data ?? []
  const chosen = new Set(skills)

  return (
    <section className="glass rounded-2xl p-5">
      <h2 className="text-sm font-semibold">Edit the role</h2>
      <p className="mt-1 text-xs text-ink-faint">
        One requirement per line. Skills drive the match score, so removing one that is not really
        asked for changes what you are measured against.
      </p>

      <form
        className="mt-4 space-y-4"
        onSubmit={(e) => {
          e.preventDefault()
          save.mutate()
        }}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="text-xs text-ink-muted">Must have</span>
            <textarea
              rows={8}
              value={must}
              onChange={(e) => setMust(e.target.value)}
              className={`${FIELD} mt-1`}
            />
          </label>
          <label className="block">
            <span className="text-xs text-ink-muted">Nice to have</span>
            <textarea
              rows={8}
              value={nice}
              onChange={(e) => setNice(e.target.value)}
              className={`${FIELD} mt-1`}
            />
          </label>
        </div>

        <div>
          <span className="text-xs text-ink-muted">Skills</span>
          {taxonomy.isPending ? (
            <p className="mt-1 text-sm text-ink-faint">Loading the skill list…</p>
          ) : (
            <>
              {skills.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {skills.map((slug) => (
                    <button
                      key={slug}
                      type="button"
                      onClick={() => setSkills((s) => s.filter((x) => x !== slug))}
                      title="Remove this skill"
                      className="inline-flex cursor-pointer items-center gap-1 rounded-full bg-accent/10 px-2.5 py-1 text-xs text-accent ring-1 ring-accent/20 ring-inset transition hover:bg-accent/20"
                    >
                      {known.find((s) => s.slug === slug)?.name ?? slug}
                      <X size={11} aria-hidden="true" />
                    </button>
                  ))}
                </div>
              )}
              {/* A select rather than free text: the slug has to exist in the
                  taxonomy or the save is rejected, and letting someone type
                  "Pyhton" only to be told no afterwards is a worse way to find
                  that out. */}
              <select
                value=""
                onChange={(e) => {
                  if (e.target.value) setSkills((s) => [...s, e.target.value])
                }}
                className={`${FIELD} mt-2`}
              >
                <option value="">Add a skill…</option>
                {known
                  .filter((s) => !chosen.has(s.slug))
                  .map((s) => (
                    <option key={s.slug} value={s.slug}>
                      {s.name}
                    </option>
                  ))}
              </select>
            </>
          )}
        </div>

        <label className="block">
          <span className="text-xs text-ink-muted">Job description</span>
          <textarea
            rows={10}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Paste the posting here if you have it — the assistant reads this when you ask about the role."
            className={`${FIELD} mt-1`}
          />
        </label>

        {error && (
          <p
            className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger"
            role="alert"
          >
            {error}
          </p>
        )}

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={save.isPending}
            className="cursor-pointer rounded-lg bg-accent px-3.5 py-2 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
          >
            {save.isPending ? 'Saving…' : 'Save'}
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="cursor-pointer rounded-lg border border-border-subtle px-3.5 py-2 text-sm text-ink-muted transition hover:border-border-strong hover:text-ink"
          >
            Cancel
          </button>
        </div>
      </form>
    </section>
  )
}

/** Blank lines dropped, so trailing newlines do not become empty requirements. */
function toRequirements(text, kind) {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => ({ text: line, kind }))
}

function RequirementList({ title, items }) {
  if (items.length === 0) return null
  return (
    <div>
      <h3 className="text-[10px] font-semibold tracking-[0.12em] text-ink-faint uppercase">
        {title}
      </h3>
      <ul className="mt-2 space-y-1.5 text-sm">
        {items.map((item) => (
          <li key={item.id ?? item.text} className="flex gap-2.5">
            <span
              aria-hidden="true"
              className={`mt-2 size-1 shrink-0 rounded-full ${
                title === 'Must have' ? 'bg-accent' : 'bg-ink-faint'
              }`}
            />
            <span className="min-w-0">{item.text}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
