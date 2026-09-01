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
// Mirrors MAX_MESSAGE_CHARS in backend/app/schemas/agent.py. Enforced here so
// the limit is a full box rather than a red validation error after you have
// already pasted and pressed send.
const MAX_MESSAGE_CHARS = 10_000

export function AgentChat({ open, onClose }) {
  const queryClient = useQueryClient()
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const endRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  /* Grow to fit what is in it, up to a ceiling, then scroll. Height has to be
     reset to auto first or scrollHeight only ever reports the taller of the
     two and the box can grow but never shrink back. */
  useEffect(() => {
    const box = inputRef.current
    if (!box) return
    box.style.height = 'auto'
    box.style.height = `${Math.min(box.scrollHeight, 200)}px`
  }, [draft])

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
          attachments: reply.attachments ?? [],
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
    onSuccess: (result, action) => {
      for (const key of [
        ['applications'],
        ['application', result.application?.id],
        ['needs-attention'],
        ['stats'],
        ['analytics'],
      ]) {
        queryClient.invalidateQueries({ queryKey: key })
      }
      // The server reports what it did rather than the card restating what it
      // was going to do — a deletion that hit a constraint should not leave the
      // transcript claiming success.
      setTurns((t) => [
        ...t.map((turn) => (turn.action === action ? { ...turn, action: null, done: true } : turn)),
        { role: 'agent', text: `Done — ${result.summary}.` },
      ])
    },
    onError: (error) =>
      setTurns((t) => [...t, { role: 'agent', text: error.message, isError: true }]),
  })

  function submit() {
    const message = draft.trim()
    if (!message || send.isPending) return
    send.mutate(message)
    setDraft('')
  }

  if (!open) return null

  /* max-w-xl rather than max-w-md: the drawer now renders whole job
     descriptions, and 28rem wraps a posting into a column too narrow to read. */
  return (
    <aside
      className="fixed inset-y-0 right-0 z-30 flex w-full max-w-xl flex-col border-l border-border-subtle bg-surface shadow-xl"
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

            {/* Documents come from the database, not from the model's output.
                Asked to relay a job description the model rewrote it — 3,400
                characters in, 1,900 out — and sometimes said "here's the JD"
                and reproduced none of it. Rendering the stored text directly is
                the only way it arrives whole. Open by default: asking for the
                JD means wanting to read it, not to click once more. */}
            {turn.attachments?.map((attachment) => (
              <details
                key={attachment.title}
                open
                className="mt-2 rounded-lg border border-border-subtle bg-surface-muted text-left"
              >
                <summary className="cursor-pointer px-3 py-2 text-xs font-medium">
                  {attachment.title}
                </summary>
                <pre className="max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word border-t border-border-subtle px-3 py-2 font-sans text-xs leading-relaxed">
                  {attachment.body}
                </pre>
              </details>
            ))}

            {/* Which tools ran, so an answer can be traced to its source rather
                than taken on trust. Quiet enough to ignore when you don't care. */}
            {turn.tools?.length > 0 && (
              <p className="mt-1 text-[11px] text-ink-muted">
                looked up: {[...new Set(turn.tools)].join(', ')}
              </p>
            )}

            {turn.action && (
              <div
                className={`mt-2 rounded-lg border p-3 text-left ${
                  turn.action.destructive
                    ? 'border-rose-300 bg-rose-50'
                    : 'border-amber-300 bg-amber-50'
                }`}
              >
                {/* Destructive actions read differently on purpose. Everything
                    else the assistant does is undone by appending a correcting
                    event; a deletion is not, and a card that looks identical
                    trains you to click through it at the same speed. */}
                <p
                  className={`text-xs font-medium ${
                    turn.action.destructive ? 'text-rose-900' : 'text-amber-900'
                  }`}
                >
                  {turn.action.destructive ? 'This cannot be undone' : 'Confirm this change'}
                </p>
                <p className="mt-1 text-sm font-medium">{turn.action.summary}</p>

                <ul className="mt-1.5 space-y-0.5">
                  {turn.action.details.map((line) => (
                    <li
                      key={line}
                      className={`text-xs ${
                        turn.action.destructive ? 'text-rose-900' : 'text-amber-900'
                      }`}
                    >
                      {line}
                    </li>
                  ))}
                </ul>

                {/* Only for actions aimed at an existing row. A creation has
                    nothing to have resolved, and showing "100% confidence"
                    there would imply a check that never happened. */}
                {turn.action.confidence != null && (
                  <p
                    className={`mt-1.5 text-xs ${
                      turn.action.destructive ? 'text-rose-800' : 'text-amber-800'
                    }`}
                  >
                    Matched on {turn.action.matched_on} ·{' '}
                    {Math.round(turn.action.confidence * 100)}% confidence
                  </p>
                )}

                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    disabled={confirm.isPending}
                    onClick={() => confirm.mutate(turn.action)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium text-white transition disabled:opacity-60 ${
                      turn.action.destructive
                        ? 'bg-rose-600 hover:bg-rose-700'
                        : 'bg-accent hover:bg-accent-hover'
                    }`}
                  >
                    {confirm.isPending
                      ? 'Applying…'
                      : turn.action.destructive
                        ? 'Delete permanently'
                        : 'Confirm'}
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
        className="border-t border-border-subtle p-3"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <div className="flex items-end gap-2">
          {/* A textarea, not an input: what gets pasted here is a chunk of a
              job posting, and a single line shows you the last few words of it
              with no way to check what you actually pasted. Grows to a few
              lines, then scrolls. */}
          <textarea
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter breaks the line. Without the split
              // there is no way to write a second line at all, and without
              // preventDefault Enter would insert one *and* submit.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            rows={1}
            maxLength={MAX_MESSAGE_CHARS}
            placeholder="Ask, or say what happened…  (Shift+Enter for a new line)"
            aria-label="Message the assistant"
            className="min-w-0 flex-1 resize-none overflow-y-auto rounded-lg border border-border-subtle px-3 py-2 text-sm leading-relaxed outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
          />
          <button
            type="submit"
            disabled={send.isPending || !draft.trim()}
            className="shrink-0 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-accent-hover disabled:opacity-60"
          >
            Send
          </button>
        </div>

        {/* Only once it is close to mattering. A counter sitting there from the
            first keystroke reads as a constraint on ordinary questions, which
            this is not — it is a guard against pasting an entire document. */}
        {draft.length > MAX_MESSAGE_CHARS * 0.8 && (
          <p className="mt-1 text-right text-xs text-ink-muted">
            {draft.length.toLocaleString()} / {MAX_MESSAGE_CHARS.toLocaleString()}
            {draft.length >= MAX_MESSAGE_CHARS && ' — to track a whole posting, paste it instead'}
          </p>
        )}
      </form>
    </aside>
  )
}
