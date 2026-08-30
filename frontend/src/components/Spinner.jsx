export function Spinner({ label }) {
  return (
    <div className="flex flex-col items-center gap-3" role="status" aria-live="polite">
      <div className="size-6 animate-spin rounded-full border-2 border-border-subtle border-t-accent" />
      {label ? <p className="text-sm text-ink-muted">{label}</p> : null}
      <span className="sr-only">{label ?? 'Loading'}</span>
    </div>
  )
}
