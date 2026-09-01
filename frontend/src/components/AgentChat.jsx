import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api'

/**
 * The assistant drawer.
 *
 * Nothing here writes on its own. A message gets an answer, or a confirm card
 * describing the exact change that would be made. The card renders the
 * assistant's `summary` and `details` verbatim, and `details` is built server-
 * side from the same values that would be written — so there is no field the
 * card can quietly omit. A confirmation that hides half the payload is theatre,
 * not a check.
 *
 * The card is deliberately generic rather than per-action: the assistant can
 * propose four kinds of change now, and a switch here would be a fifth place to
 * forget to update when a fifth arrives.
 */
export function AgentChat({ open, onClose }) {
  const queryClient = useQueryClient()
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  const send = useMutation({
    mutationFn: (message) => api.agentChat(message),
    onMutate: (message) => setTurns((t) => [...t, { role: 'you', text: message }]),
    onSuccess: (reply) =>
      setTurns((t) => [
        ...t,
        {
          role: 'agent',
          text: reply.message,
          action: reply.pending_action,
          tools: reply.tools_used ?? [],
        },
      ]),
    onError: (error) =>
      setTurns((t) => [...t, { role: 'agent', text: error.message, isError: true }]),
  })

  const confirm = useMutation({
    // The payload is echoed back untouched. The server re-validates it against
    // a typed schema per kind, so nothing here needs to know what is inside.
    mutationFn: (action) => api.agentConfirm({ kind: action.kind, ...action.payload }),
    onSuccess: (application, action) => {
      for (const key of [
        ['applications'],
        ['application', application.id],
        ['needs-attention'],
        ['stats'],
        ['analytics'],
      ]) {
        queryClient.invalidateQueries({ queryKey: key })
      }
      setTurns((t) => [
        ...t.map((turn) => (turn.action === action ? { ...turn, action: null, done: true } : turn)),
        { role: 'agent', text: `Done — ${action.summary.toLowerCase()}.` },
      ])
    },
    onError: (error) =>
      setTurns((t) => [...t, { role: 'agent', text: error.message, isError: true }]),
  })

  if (!open) return null

  return (
    <aside
      className="fixed inset-y-0 right-0 z-30 flex w-full max-w-md flex-col border-l border-border-subtle bg-surface shadow-xl"
      role="dialog"
      aria-label="Assistant"
    >
      <header className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Assistant</h2>
          <p className="text-xs text-ink-muted">Proposes changes; you confirm them.</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg border border-border-subtle px-2.5 py-1 text-sm transition hover:bg-surface-muted"
          aria-label="Close assistant"
        >
          ✕
        </button>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {turns.length === 0 && (
          <div className="rounded-lg bg-surface-muted p-3 text-sm text-ink-muted">
            <p>Try:</p>
            <ul className="mt-1.5 space-y-1">
              <li>“How is my search actually going?”</li>
              <li>“What should I learn next?”</li>
              <li>“Track the backend role at Zerodha, I applied 3 days ago”</li>
              <li>“Draft a follow-up for Amazon”</li>
              <li>“Compare Amazon and Razorpay”</li>
              <li>“Mark Amazon as rejected”</li>
            </ul>
          </div>
        )}

        {turns.map((turn, index) => (
          <div key={index} className={turn.role === 'you' ? 'text-right' : ''}>
            <div
              className={`inline-block max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
                turn.role === 'you'
                  ? 'bg-accent text-white'
                  : turn.isError
                    ? 'bg-rose-50 text-rose-700'
                    : 'bg-surface-muted'
              }`}
            >
              {turn.text}
            </div>

            {/* Which tools ran, so an answer can be traced to its source rather
                than taken on trust. Quiet enough to ignore when you don't care. */}
            {turn.tools?.length > 0 && (
              <p className="mt-1 text-[11px] text-ink-muted">
                looked up: {[...new Set(turn.tools)].join(', ')}
              </p>
            )}

            {turn.action && (
              <div className="mt-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-left">
                <p className="text-xs font-medium text-amber-900">Confirm this change</p>
                <p className="mt-1 text-sm font-medium">{turn.action.summary}</p>

                <ul className="mt-1.5 space-y-0.5">
                  {turn.action.details.map((line) => (
                    <li key={line} className="text-xs text-amber-900">
                      {line}
                    </li>
                  ))}
                </ul>

                {/* Only for actions aimed at an existing row. A creation has
                    nothing to have resolved, and showing "100% confidence"
                    there would imply a check that never happened. */}
                {turn.action.confidence != null && (
                  <p className="mt-1.5 text-xs text-amber-800">
                    Matched on {turn.action.matched_on} ·{' '}
                    {Math.round(turn.action.confidence * 100)}% confidence
                  </p>
                )}

                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    disabled={confirm.isPending}
                    onClick={() => confirm.mutate(turn.action)}
                    className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
                  >
                    {confirm.isPending ? 'Applying…' : 'Confirm'}
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setTurns((t) =>
                        t.map((x) => (x === turn ? { ...x, action: null } : x)),
                      )
                    }
                    className="rounded-lg border border-border-subtle bg-surface px-3 py-1.5 text-xs transition hover:bg-surface-muted"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}

        {send.isPending && <p className="text-sm text-ink-muted">Thinking…</p>}
        <div ref={endRef} />
      </div>

      <form
        className="flex gap-2 border-t border-border-subtle p-3"
        onSubmit={(e) => {
          e.preventDefault()
          if (!draft.trim()) return
          send.mutate(draft.trim())
          setDraft('')
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask, or say what happened…"
          aria-label="Message the assistant"
          className="min-w-0 flex-1 rounded-lg border border-border-subtle px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
        />
        <button
          type="submit"
          disabled={send.isPending || !draft.trim()}
          className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
        >
          Send
        </button>
      </form>
    </aside>
  )
}
