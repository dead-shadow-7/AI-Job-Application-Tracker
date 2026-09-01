import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api'
import { EVENT_LABELS } from '@/lib/format'

/**
 * The assistant drawer.
 *
 * Nothing here writes on its own. A message either gets an answer, a list of
 * candidates to choose between, or a confirm card naming the exact application
 * that would change. The card shows the *resolved* application rather than the
 * phrase that was typed — confirming "mark Amazon as rejected" without seeing
 * which Amazon defeats the point of confirming.
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
          options: reply.disambiguation,
        },
      ]),
    onError: (error) =>
      setTurns((t) => [...t, { role: 'agent', text: error.message, isError: true }]),
  })

  const confirm = useMutation({
    mutationFn: (action) =>
      api.agentConfirm({
        application_id: action.application_id,
        event_type: action.event_type,
        note: action.note ?? null,
      }),
    onSuccess: (application, action) => {
      queryClient.invalidateQueries({ queryKey: ['applications'] })
      queryClient.invalidateQueries({ queryKey: ['application', application.id] })
      queryClient.invalidateQueries({ queryKey: ['needs-attention'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setTurns((t) => [
        ...t.map((turn) => (turn.action === action ? { ...turn, action: null, done: true } : turn)),
        {
          role: 'agent',
          text: `Logged ${EVENT_LABELS[action.event_type] ?? action.event_type} on ${action.application_label}.`,
        },
      ])
    },
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
              <li>“What should I follow up on?”</li>
              <li>“Mark Amazon as rejected”</li>
              <li>“I heard back from Razorpay”</li>
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

            {turn.options?.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {turn.options.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => send.mutate(option)}
                    className="rounded-full border border-border-subtle px-2.5 py-1 text-xs transition hover:bg-surface-muted"
                  >
                    {option}
                  </button>
                ))}
              </div>
            )}

            {turn.action && (
              <div className="mt-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-left">
                <p className="text-xs font-medium text-amber-900">Confirm this change</p>
                <p className="mt-1 text-sm">
                  Log{' '}
                  <span className="font-medium">
                    {EVENT_LABELS[turn.action.event_type] ?? turn.action.event_type}
                  </span>{' '}
                  on <span className="font-medium">{turn.action.application_label}</span>
                </p>
                <p className="mt-1 text-xs text-amber-800">
                  Matched on {turn.action.matched_on} ·{' '}
                  {Math.round(turn.action.confidence * 100)}% confidence
                </p>
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    disabled={confirm.isPending}
                    onClick={() => confirm.mutate(turn.action)}
                    className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
                  >
                    {confirm.isPending ? 'Logging…' : 'Confirm'}
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
