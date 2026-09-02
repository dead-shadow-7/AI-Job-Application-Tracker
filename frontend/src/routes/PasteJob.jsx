import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Wand2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { PageHeader } from '@/components/PageHeader'
import { ReviewExtraction } from '@/components/ReviewExtraction'
import { api } from '@/lib/api'

const FIELD =
  'well w-full rounded-xl px-3 py-2.5 text-sm outline-none transition placeholder:text-ink-faint focus:border-accent/40 focus:shadow-[0_0_0_3px] focus:shadow-accent/12 focus-visible:outline-none'

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
    <div className="mx-auto max-w-3xl space-y-5">
      <PageHeader
        back={{ to: '/', label: 'Applications' }}
        title="Paste a job description"
        subtitle="Copy the posting from anywhere and paste it below. You review everything before it is saved."
      />

      <form
        onSubmit={(e) => {
          e.preventDefault()
          extract.mutate()
        }}
        className="glass space-y-4 rounded-2xl p-5"
      >
        <label className="block space-y-2">
          <span className="block text-sm font-medium">Job description</span>
          <textarea
            required
            rows={16}
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
            placeholder="Paste the full posting — title, responsibilities, requirements, salary…"
            className={`${FIELD} font-mono text-xs leading-relaxed`}
          />
          <span className="block font-mono text-xs text-ink-faint tabular-nums">
            {rawText.length.toLocaleString()} characters
            {rawText.length > 0 && rawText.length < 120 && ' — too short, paste the whole posting'}
          </span>
        </label>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block space-y-2">
            <span className="block text-sm font-medium">Job URL</span>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://…"
              className={FIELD}
            />
          </label>
          <label className="block space-y-2">
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
          <p
            className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
            role="alert"
          >
            {extract.error.message}
          </p>
        )}

        <div className="flex items-center justify-between gap-3">
          <Link
            to="/applications/new"
            className="text-sm text-ink-muted transition hover:text-accent"
          >
            Or enter it by hand
          </Link>
          <button
            type="submit"
            disabled={extract.isPending || rawText.trim().length < 120}
            className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Wand2 size={16} aria-hidden="true" />
            {extract.isPending ? 'Reading the posting…' : 'Extract'}
          </button>
        </div>
      </form>
    </div>
  )
}
