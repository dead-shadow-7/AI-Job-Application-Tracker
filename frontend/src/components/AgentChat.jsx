import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowUp,
  Check,
  ChevronDown,
  FileText,
  Search,
  ShieldAlert,
  Sparkles,
  Trash2,
  TriangleAlert,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Markdown } from '@/components/Markdown'
import { api } from '@/lib/api'

/**
 * The assistant.
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
 *
 * The panel fills whatever box the shell gives it — a docked column on a wide
 * screen, a sheet on a narrow one. `onClose` is passed only in the second case;
 * a column has nothing to close.
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

/* The empty state. Clickable rather than printed, because the hard part of a
   blank assistant is not knowing what to type — and retyping an example by
   hand to find out is a tax on the one moment you know least. */
const OPENERS = [
  'How is my search actually going?',
  'What should I learn next?',
  'Draft a follow-up for Amazon',
  'Compare my two most recent applications',
]

export function AgentChat({ onClose }) {
  const queryClient = useQueryClient()
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const inputRef = useRef(null)
  const scrollRef = useRef(null)
  const stickRef = useRef(true)
  const [pinned, setPinned] = useState(true)

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
     stops when it catches up, so an idle panel costs nothing. */
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
      setPinned(true)
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

  function submit(text) {
    const message = (text ?? draft).trim()
    if (!message || send.isPending) return
    send.mutate(message)
    setDraft('')
  }

  return (
    <section
      className="glass-panel relative flex h-full flex-col overflow-hidden rounded-2xl"
      aria-label="Assistant"
    >
      {/* The panel's status light: a hairline along its top edge that breathes
          while a turn is in flight. The one moving thing in the interface, and
          it means exactly one thing. */}
      <span
        aria-hidden="true"
        className={`pointer-events-none absolute inset-x-8 top-0 h-px bg-linear-to-r from-transparent via-accent to-transparent ${
          send.isPending ? 'stream-glow' : 'opacity-25'
        }`}
      />

      <header className="flex items-center gap-3 px-4 py-3.5">
        <span
          aria-hidden="true"
          className="grid size-9 shrink-0 place-items-center rounded-xl bg-accent/15 text-accent ring-1 ring-accent/25"
        >
          <Sparkles size={17} strokeWidth={2} />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-sm font-semibold tracking-tight">Assistant</h2>
          <p className="truncate text-xs text-ink-muted">
            {send.isPending ? 'Working on it…' : 'Proposes changes; you confirm them.'}
          </p>
        </div>
        {turns.length > 0 && (
          <button
            type="button"
            onClick={() => setTurns([])}
            title="Clear the conversation"
            className="grid size-8 cursor-pointer place-items-center rounded-lg text-ink-faint transition hover:bg-surface-muted hover:text-ink"
          >
            <Trash2 size={15} aria-hidden="true" />
            <span className="sr-only">Clear the conversation</span>
          </button>
        )}
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="grid size-8 cursor-pointer place-items-center rounded-lg text-ink-faint transition hover:bg-surface-muted hover:text-ink"
          >
            <X size={16} aria-hidden="true" />
            <span className="sr-only">Close assistant</span>
          </button>
        )}
      </header>

      <div className="mx-4 h-px bg-border-subtle/60" />

      <div
        ref={scrollRef}
        onScroll={(e) => {
          const box = e.currentTarget
          const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < STICK_WITHIN
          stickRef.current = atBottom
          setPinned(atBottom)
        }}
        className="flex-1 space-y-4 overflow-y-auto px-4 py-4"
      >
        {turns.length === 0 && !send.isPending && (
          <div className="rise pt-6">
            <p className="font-display text-lg leading-snug font-semibold tracking-tight text-balance">
              Ask about the search, or just say what happened.
            </p>
            <p className="mt-1.5 text-sm text-ink-muted">
              It reads your applications and your resume. Anything it wants to change comes back as
              a card you approve first.
            </p>
            <ul className="mt-4 space-y-2">
              {OPENERS.map((opener) => (
                <li key={opener}>
                  <button
                    type="button"
                    onClick={() => submit(opener)}
                    className="glass group flex w-full cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm text-ink-muted transition hover:text-ink"
                  >
                    <Sparkles
                      size={14}
                      aria-hidden="true"
                      className="shrink-0 text-accent/60 transition group-hover:text-accent"
                    />
                    <span className="min-w-0 flex-1">{opener}</span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="mt-4 text-xs text-ink-faint">
              To track a whole posting, use <span className="text-ink-muted">Paste a job</span> —
              it keeps the description intact.
            </p>
          </div>
        )}

        {turns.map((turn, index) => (
          <div key={index} className={turn.role === 'you' ? 'flex justify-end' : ''}>
            <div className={turn.role === 'you' ? 'max-w-[88%]' : 'w-full'}>
              {/* Only the assistant's prose is markdown. What you typed is shown
                  as you typed it — asterisks in a pasted title are punctuation,
                  not formatting — and an error message is not the model's to
                  format at all. */}
              <div
                className={`text-sm leading-relaxed ${
                  turn.role === 'you'
                    ? 'rounded-2xl rounded-br-md bg-accent px-3.5 py-2.5 font-medium whitespace-pre-wrap text-accent-ink'
                    : turn.isError
                      ? 'flex gap-2.5 rounded-2xl border border-danger/30 bg-danger/10 px-3.5 py-2.5 whitespace-pre-wrap text-danger'
                      : 'glass rounded-2xl rounded-bl-md px-3.5 py-2.5'
                }`}
              >
                {turn.isError && (
                  <TriangleAlert size={16} aria-hidden="true" className="mt-0.5 shrink-0" />
                )}
                {turn.role === 'you' || turn.isError ? (
                  <span>{turn.text}</span>
                ) : (
                  <Markdown>{turn.text}</Markdown>
                )}
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
                  className="glass group mt-2 overflow-hidden rounded-xl"
                >
                  <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium text-ink-muted transition hover:text-ink">
                    <FileText size={14} aria-hidden="true" className="shrink-0 text-accent/70" />
                    <span className="min-w-0 flex-1 truncate">{attachment.title}</span>
                    <ChevronDown
                      size={14}
                      aria-hidden="true"
                      className="shrink-0 transition group-open:rotate-180"
                    />
                  </summary>
                  <pre className="max-h-96 overflow-auto border-t border-border-subtle/60 px-3 py-2.5 font-sans text-xs leading-relaxed whitespace-pre-wrap wrap-break-word text-ink-muted">
                    {attachment.body}
                  </pre>
                </details>
              ))}

              {/* Which tools ran, so an answer can be traced to its source rather
                  than taken on trust. Quiet enough to ignore when you don't care. */}
              {turn.tools?.length > 0 && <ToolTrail tools={turn.tools} verb="looked up" />}

              {turn.action && (
                <ConfirmCard
                  action={turn.action}
                  pending={confirm.isPending}
                  onConfirm={() => confirm.mutate(turn.action)}
                  onCancel={() =>
                    setTurns((t) => t.map((x) => (x === turn ? { ...x, action: null } : x)))
                  }
                />
              )}
            </div>
          </div>
        ))}

        {/* The turn in flight. Once it settles the same text arrives as an
            ordinary entry in `turns`, and because the animation has already
            drained by then the swap is invisible. */}
        {send.isPending && (
          <div>
            <div className="glass rounded-2xl rounded-bl-md px-3.5 py-2.5 text-sm leading-relaxed">
              {liveText ? (
                <Markdown>{liveText}</Markdown>
              ) : (
                <span className="flex items-center gap-1.5 text-ink-muted">
                  Thinking
                  <Dots />
                </span>
              )}
              {/* Sits under the text rather than after it: mid-stream the last
                  block is a paragraph or a list item, and a caret spliced into
                  the flow jumps around as the markdown re-parses each frame. */}
              {liveText && (
                <span className="stream-glow mt-1.5 block h-0.5 w-5 rounded-full bg-accent" />
              )}
            </div>

            {/* Named as they run, not listed at the end: this is what accounts
                for the wait while the loop is several rounds deep. */}
            {liveTools.length > 0 && <ToolTrail tools={liveTools} verb="looking up" live />}
          </div>
        )}
      </div>

      {/* Only when you have actually scrolled away from it. Otherwise this is a
          button that always says "go where you already are". */}
      {!pinned && (turns.length > 0 || send.isPending) && (
        <button
          type="button"
          onClick={() => {
            stickRef.current = true
            setPinned(true)
            const box = scrollRef.current
            if (box) box.scrollTop = box.scrollHeight
          }}
          className="glass absolute bottom-24 left-1/2 flex -translate-x-1/2 cursor-pointer items-center gap-1.5 rounded-full px-3 py-1.5 text-xs text-ink-muted transition hover:text-ink"
        >
          <ChevronDown size={13} aria-hidden="true" />
          Latest
        </button>
      )}

      <form
        className="p-3"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <div className="well flex items-end gap-2 rounded-2xl p-2 transition focus-within:border-accent/40 focus-within:shadow-[0_0_0_3px] focus-within:shadow-accent/12">
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
            placeholder="Ask, or say what happened…"
            aria-label="Message the assistant"
            className="min-w-0 flex-1 resize-none self-center overflow-y-auto bg-transparent px-2 py-1.5 text-sm leading-relaxed outline-none placeholder:text-ink-faint focus-visible:outline-none"
          />
          <button
            type="submit"
            disabled={send.isPending || !draft.trim()}
            title="Send  ·  Shift+Enter for a new line"
            className="grid size-9 shrink-0 cursor-pointer place-items-center rounded-xl bg-accent text-accent-ink transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-surface-muted disabled:text-ink-faint"
          >
            <ArrowUp size={17} strokeWidth={2.5} aria-hidden="true" />
            <span className="sr-only">Send message</span>
          </button>
        </div>

        {/* Only once it is close to mattering. A counter sitting there from the
            first keystroke reads as a constraint on ordinary questions, which
            this is not — it is a guard against pasting an entire document. */}
        {draft.length > MAX_MESSAGE_CHARS * 0.8 && (
          <p className="mt-1.5 px-1 text-right font-mono text-xs text-ink-faint tabular-nums">
            {draft.length.toLocaleString()} / {MAX_MESSAGE_CHARS.toLocaleString()}
            {draft.length >= MAX_MESSAGE_CHARS && ' — to track a whole posting, paste it instead'}
          </p>
        )}
      </form>
    </section>
  )
}

function ToolTrail({ tools, verb, live = false }) {
  return (
    <p className="mt-1.5 flex flex-wrap items-center gap-1.5 px-1 text-[11px] text-ink-faint">
      <Search
        size={11}
        aria-hidden="true"
        className={`shrink-0 ${live ? 'stream-glow text-accent' : ''}`}
      />
      <span>{verb}:</span>
      {[...new Set(tools)].map((tool) => (
        <span
          key={tool}
          className="rounded-md bg-surface-muted/70 px-1.5 py-0.5 font-mono text-[10px] text-ink-muted"
        >
          {tool}
        </span>
      ))}
    </p>
  )
}

function Dots() {
  return (
    <span aria-hidden="true" className="flex gap-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="stream-glow size-1 rounded-full bg-ink-muted"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </span>
  )
}

/**
 * The change the assistant wants to make, before it is made.
 *
 * Destructive actions read differently on purpose. Everything else the
 * assistant does is undone by appending a correcting event; a deletion is not,
 * and a card that looks identical trains you to click through it at the same
 * speed.
 */
function ConfirmCard({ action, pending, onConfirm, onCancel }) {
  const destructive = action.destructive
  const Icon = destructive ? ShieldAlert : Check

  return (
    <div
      className={`mt-2.5 overflow-hidden rounded-xl border ${
        destructive ? 'border-danger/40 bg-danger/8' : 'border-signal/35 bg-signal/8'
      }`}
    >
      <div
        className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold tracking-wide ${
          destructive
            ? 'border-b border-danger/25 bg-danger/10 text-danger'
            : 'border-b border-signal/25 bg-signal/10 text-signal'
        }`}
      >
        <Icon size={14} aria-hidden="true" className="shrink-0" />
        {destructive ? 'This cannot be undone' : 'Confirm this change'}
      </div>

      <div className="px-3.5 py-3">
        <p className="text-sm font-medium">{action.summary}</p>

        <ul className="mt-2 space-y-1">
          {action.details.map((line) => (
            <li key={line} className="flex gap-2 text-xs text-ink-muted">
              <span
                aria-hidden="true"
                className={`mt-1.5 size-1 shrink-0 rounded-full ${
                  destructive ? 'bg-danger/60' : 'bg-signal/60'
                }`}
              />
              <span className="min-w-0">{line}</span>
            </li>
          ))}
        </ul>

        {/* Only for actions aimed at an existing row. A creation has nothing to
            have resolved, and showing "100% confidence" there would imply a
            check that never happened. */}
        {action.confidence != null && (
          <p className="mt-2.5 font-mono text-[11px] text-ink-faint tabular-nums">
            matched on {action.matched_on} · {Math.round(action.confidence * 100)}% confidence
          </p>
        )}

        <div className="mt-3 flex gap-2">
          <button
            type="button"
            disabled={pending}
            onClick={onConfirm}
            className={`cursor-pointer rounded-lg px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${
              destructive
                ? 'bg-danger text-canvas hover:brightness-110'
                : 'bg-accent text-accent-ink hover:bg-accent-hover'
            }`}
          >
            {pending ? 'Applying…' : destructive ? 'Delete permanently' : 'Confirm'}
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="cursor-pointer rounded-lg border border-border-subtle px-3 py-1.5 text-xs font-medium text-ink-muted transition hover:bg-surface-muted hover:text-ink"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
