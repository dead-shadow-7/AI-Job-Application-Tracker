import { RotateCw, TriangleAlert } from 'lucide-react'

export function ErrorState({ error, onRetry }) {
  const offline = error?.message?.includes('fetch') || error?.name === 'TypeError'

  return (
    <div className="glass flex gap-3 rounded-xl border-danger/30 bg-danger/8 p-5" role="alert">
      <TriangleAlert size={18} aria-hidden="true" className="mt-0.5 shrink-0 text-danger" />
      <div className="min-w-0">
        <h2 className="text-sm font-semibold text-danger">Something went wrong</h2>
        <p className="mt-1 text-sm text-ink-muted">{error?.message ?? 'Unknown error'}</p>
        {offline && (
          <p className="mt-2 text-xs text-ink-faint">
            The API may not be running — try{' '}
            <code className="rounded bg-surface-muted px-1 py-0.5 font-mono">
              docker compose up -d
            </code>
            .
          </p>
        )}
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-4 inline-flex cursor-pointer items-center gap-2 rounded-lg border border-danger/40 px-3 py-1.5 text-sm font-medium text-danger transition hover:bg-danger/10"
          >
            <RotateCw size={14} aria-hidden="true" />
            Try again
          </button>
        )}
      </div>
    </div>
  )
}
