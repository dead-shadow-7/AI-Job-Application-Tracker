import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ReviewExtraction } from '@/components/ReviewExtraction'
import { api } from '@/lib/api'

const FIELD =
  'w-full rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

export function PasteJob() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [rawText, setRawText] = useState('')
  const [url, setUrl] = useState('')
  const [platform, setPlatform] = useState('')
  const [preview, setPreview] = useState(null)

  const extract = useMutation({
    mutationFn: () =>
      api.ingestJob({
        raw_text: rawText,
        url: url.trim() || null,
        source_platform: platform.trim() || null,
      }),
    onSuccess: setPreview,
  })

  const save = useMutation({
    mutationFn: ({ job, initialEvent, occurredAt }) =>
      api.createApplication({ job, initial_event: initialEvent, occurred_at: occurredAt }),
    onSuccess: (application) => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      navigate(`/applications/${application.id}`)
    },
  })

  if (preview) {
    return (
      <ReviewExtraction
        preview={preview}
        onBack={() => setPreview(null)}
        onSave={save.mutate}
        saving={save.isPending}
        error={save.error}
      />
    )
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <Link to="/" className="text-sm text-ink-muted hover:text-accent">
          ← Applications
        </Link>
        <h1 className="mt-2 text-lg font-semibold tracking-tight">Paste a job description</h1>
        <p className="mt-0.5 text-sm text-ink-muted">
          Copy the posting from anywhere and paste it below. You review everything before it
          is saved.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          extract.mutate()
        }}
        className="space-y-4 rounded-xl border border-border-subtle bg-surface p-5"
      >
        <label className="block space-y-1.5">
          <span className="block text-sm font-medium">Job description</span>
          <textarea
            required
            rows={16}
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
            placeholder="Paste the full posting — title, responsibilities, requirements, salary…"
            className={`${FIELD} font-mono text-xs leading-relaxed`}
          />
          <span className="block text-xs text-ink-muted">
            {rawText.length.toLocaleString()} characters
            {rawText.length > 0 && rawText.length < 120 && ' — too short, paste the whole posting'}
          </span>
        </label>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block space-y-1.5">
            <span className="block text-sm font-medium">Job URL</span>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://…"
              className={FIELD}
            />
          </label>
          <label className="block space-y-1.5">
            <span className="block text-sm font-medium">Platform</span>
            <input
              value={platform}
              onChange={(e) => setPlatform(e.target.value)}
              placeholder="LinkedIn, Naukri, company site…"
              className={FIELD}
            />
          </label>
        </div>

        {extract.error && (
          <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700" role="alert">
            {extract.error.message}
          </p>
        )}

        <div className="flex items-center justify-between gap-3">
          <Link to="/applications/new" className="text-sm text-ink-muted hover:text-accent">
            Or enter it by hand
          </Link>
          <button
            type="submit"
            disabled={extract.isPending || rawText.trim().length < 120}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
          >
            {extract.isPending ? 'Reading the posting…' : 'Extract'}
          </button>
        </div>
      </form>
    </div>
  )
}
