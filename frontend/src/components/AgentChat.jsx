import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Markdown } from '@/components/Markdown'
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
 *
 * The answer streams. That is not decoration: a turn can take six rounds of
 * model call plus tool, and the old "Thinking…" line spent all of it saying
 * nothing — indistinguishable, from where the user sits, from a request that
 * had already failed.
 */
// Mirrors MAX_MESSAGE_CHARS in backend/app/schemas/agent.py. Enforced here so
// the limit is a full box rather than a red validation error after you have
// already pasted and pressed send.
const MAX_MESSAGE_CHARS = 10_000

// How much of the outstanding backlog to render each frame. Tokens arrive from
// the provider in uneven bursts — thirty characters, then nothing for 200ms —
// and painting each burst on arrival makes the text jerk. Draining a fraction
// per frame turns that into an even flow, and because the fraction is of
// whatever is *outstanding*, a large burst is still absorbed in a few frames
// rather than metered out at a fixed speed the answer eventually outruns.
const CATCH_UP = 8

// Below this many pixels from the bottom, the transcript follows new text.
// Above it the user has scrolled up to read something and must not be yanked.
const STICK_WITHIN = 64

// The longest the finished turn will wait for the animation to catch up. At
// sixty frames a second any realistic backlog drains in well under a second,
// so in practice this only fires when frames are not running at all.
const DRAIN_CEILING_MS = 2_000

export function AgentChat({ open, onClose }) {
  const queryClient = useQueryClient()
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const inputRef = useRef(null)
  const scrollRef = useRef(null)
  const stickRef = useRef(true)

  /* The turn in flight. `liveText` is what is on screen, which lags what has
     arrived — see the pump below. Tools are mirrored into a ref because the
     mutation's error handler needs them after React has moved on. */
  const [liveText, setLiveText] = useState('')
  const [liveTools, setLiveTools] = useState([])
  const toolsRef = useRef([])

  const receivedRef = useRef('') // everything the server has sent
  const shownRef = useRef('') // everything painted so far
  const frameRef = useRef(0)
  const settleRef = useRef(null)

  /* One character-advancing frame. Reschedules itself while it is behind and
     stops when it catches up, so an idle drawer costs nothing. */
  const pump = useCallback(function advance() {
    const behind = receivedRef.current.length - shownRef.current.length
    if (behind > 0) {
      const step = Math.max(1, Math.ceil(behind / CATCH_UP))
      shownRef.current = receivedRef.current.slice(0, shownRef.current.length + step)
      setLiveText(shownRef.current)
      frameRef.current = requestAnimationFrame(advance)
      return
    }
    frameRef.current = 0
    settleRef.current?.()
    settleRef.current = null
  }, [])

  const receive = useCallback(
    (text) => {
      receivedRef.current += text
      if (!frameRef.current) frameRef.current = requestAnimationFrame(pump)
    },
    [pump],
  )

  /* Take back what was streamed. The model narrated before calling a tool, so
     that sentence was never part of the reply and was never saved. */
  const withdraw = useCallback(() => {
    receivedRef.current = ''
    shownRef.current = ''
    setLiveText('')
  }, [])

  /* Resolves once the last character is on screen. Without it the finished
     turn replaces the live one mid-animation and the tail is never shown. */
  const drained = useCallback(
    () =>
      new Promise((resolve) => {
        if (!frameRef.current) return resolve()

        /* A browser does not run animation frames for a hidden tab, so an
           answer that finishes while the user is looking at something else
           would never drain and the turn would sit unresolved until they came
           back to it. Past the ceiling, show the rest at once. */
        const snap = setTimeout(() => {
          cancelAnimationFrame(frameRef.current)
          frameRef.current = 0
          settleRef.current = null
          shownRef.current = receivedRef.current
          setLiveText(shownRef.current)
          resolve()
        }, DRAIN_CEILING_MS)

        settleRef.current = () => {
          clearTimeout(snap)
          resolve()
        }
      }),
    [],
  )

  const clearLive = useCallback(() => {
    receivedRef.current = ''
    shownRef.current = ''
    toolsRef.current = []
    setLiveText('')
    setLiveTools([])
  }, [])

  useEffect(() => () => cancelAnimationFrame(frameRef.current), [])

  /* Follow the bottom, but only if that is where the user already is. Set
     directly rather than through scrollIntoView: this runs on every frame of
     the animation, and a smooth scroll retargeted sixty times a second never
     arrives anywhere. */
  useEffect(() => {
    const box = scrollRef.current
    if (box && stickRef.current) box.scrollTop = box.scrollHeight
  }, [turns, liveText, liveTools])

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
    mutationFn: async (message) => {
      let finished = null
      let failure = null

      await api.agentChatStream(message, {
        onEvent: (event) => {
          if (event.type === 'delta') receive(event.text)
          else if (event.type === 'superseded') withdraw()
          else if (event.type === 'tool') {
            toolsRef.current = [...toolsRef.current, event.name]
            setLiveTools(toolsRef.current)
          } else if (event.type === 'done') finished = event
          // Reported in the stream rather than as a status code, because by
          // then the response is already a 200 with its headers sent. Recorded
          // and raised after the body closes rather than thrown from here, so
          // the reader is not abandoned mid-frame.
          else if (event.type === 'error') failure = event.detail
        },
      })

      await drained()
      if (failure) throw new Error(failure)
      if (!finished) throw new Error('The assistant stopped before finishing its answer.')
      return finished
    },
    onMutate: (message) => {
      clearLive()
      stickRef.current = true
      setTurns((t) => [...t, { role: 'you', text: message }])
    },
    onSuccess: (reply) => {
      // The server's copy, not the one assembled from deltas. They agree
      // except when the loop hit its round limit or withdrew a preamble, and
      // this is the version that went into the transcript.
      setTurns((t) => [
        ...t,
        {
          role: 'agent',
          text: reply.message,
          action: reply.pending_action,
          attachments: reply.attachments ?? [],
          tools: reply.tools_used ?? [],
        },
      ])
      clearLive()
    },
    onError: (error) => {
      // Whatever had been written is kept. It is half an answer, but throwing
      // it away to show the error alone loses the only part that was working.
      const partial = shownRef.current
      const tools = toolsRef.current
      setTurns((t) => [
        ...t,
        ...(partial ? [{ role: 'agent', text: partial, tools }] : []),
        { role: 'agent', text: error.message, isError: true },
      ])
      clearLive()
    },
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

      <div
        ref={scrollRef}
        onScroll={(e) => {
          const box = e.currentTarget
          stickRef.current =
            box.scrollHeight - box.scrollTop - box.clientHeight < STICK_WITHIN
        }}
        className="flex-1 space-y-3 overflow-y-auto px-4 py-4"
      >
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
            {/* Only the assistant's prose is markdown. What you typed is shown
                as you typed it — asterisks in a pasted title are punctuation,
                not formatting — and an error message is not the model's to
                format at all. */}
            <div
              className={`inline-block max-w-[85%] rounded-lg px-3 py-2 text-left text-sm ${
                turn.role === 'you'
                  ? 'bg-accent whitespace-pre-wrap text-white'
                  : turn.isError
                    ? 'bg-rose-50 whitespace-pre-wrap text-rose-700'
                    : 'bg-surface-muted'
              }`}
            >
              {turn.role === 'you' || turn.isError ? turn.text : <Markdown>{turn.text}</Markdown>}
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

        {/* The turn in flight. Once it settles the same text arrives as an
            ordinary entry in `turns`, and because the animation has already
            drained by then the swap is invisible. */}
        {send.isPending && (
          <div>
            <div className="inline-block max-w-[85%] rounded-lg bg-surface-muted px-3 py-2 text-sm">
              {liveText ? <Markdown>{liveText}</Markdown> : <span className="text-ink-muted">Thinking…</span>}
              {/* Sits under the text rather than after it: mid-stream the last
                  block is a paragraph or a list item, and a caret spliced into
                  the flow jumps around as the markdown re-parses each frame. */}
              <span className="mt-1 block h-0.5 w-4 animate-pulse rounded-full bg-ink-muted" />
            </div>

            {/* Named as they run, not listed at the end: this is what accounts
                for the wait while the loop is several rounds deep. */}
            {liveTools.length > 0 && (
              <p className="mt-1 text-[11px] text-ink-muted">
                looking up: {[...new Set(liveTools)].join(', ')}
              </p>
            )}
          </div>
        )}
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
