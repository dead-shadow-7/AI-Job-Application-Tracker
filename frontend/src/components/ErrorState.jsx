export function ErrorState({ error, onRetry }) {
  const offline = error?.message?.includes('fetch') || error?.name === 'TypeError'

  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 p-6" role="alert">
      <h2 className="text-sm font-medium text-rose-800">Something went wrong</h2>
      <p className="mt-1 text-sm text-rose-700">{error?.message ?? 'Unknown error'}</p>
      {offline && (
        <p className="mt-2 text-xs text-rose-600">
          The API may not be running — try <code>docker compose up -d</code>.
        </p>
      )}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-lg border border-rose-300 bg-surface px-3 py-1.5 text-sm font-medium text-rose-800 transition hover:bg-rose-100"
        >
          Try again
        </button>
      )}
    </div>
  )
}
