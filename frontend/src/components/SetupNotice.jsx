/**
 * Shown when VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY are missing.
 *
 * Auth is the first thing a fresh clone hits, and an unconfigured Supabase
 * client fails somewhere deep in the SDK. Naming the missing variables here
 * turns a confusing stack trace into a two-minute fix.
 */
export function SetupNotice() {
  return (
    <main className="flex min-h-dvh items-center justify-center px-4">
      <div className="w-full max-w-lg rounded-xl border border-border-subtle bg-surface p-8 shadow-sm">
        <h1 className="text-lg font-semibold tracking-tight">Supabase isn’t configured yet</h1>
        <p className="mt-2 text-sm text-ink-muted">
          Create a Supabase project, then add its URL and anon key to{' '}
          <code className="rounded bg-surface-muted px-1 py-0.5 text-xs">frontend/.env.local</code>:
        </p>
        <pre className="mt-4 overflow-x-auto rounded-lg bg-surface-muted p-4 text-xs">
          {`VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_ANON_KEY=<anon key>
VITE_API_BASE_URL=http://localhost:8000`}
        </pre>
        <p className="mt-4 text-sm text-ink-muted">
          The backend needs the matching values in the root{' '}
          <code className="rounded bg-surface-muted px-1 py-0.5 text-xs">.env</code>. Restart the dev
          server afterwards — Vite only reads env files at startup.
        </p>
      </div>
    </main>
  )
}
