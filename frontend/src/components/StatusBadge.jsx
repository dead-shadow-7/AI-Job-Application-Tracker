import { STATUS_LABELS } from '@/lib/format'

/* Hue carries meaning: grey = dormant, blue = in motion, amber = awaiting you,
   green = won, red = lost. Colour is never the only signal — the label is
   always present — so this stays readable without colour perception.

   On the dark canvas the pairing inverts: a light tint at ~12% over the glass,
   and the text at the 300 step rather than the 700. The 700s that worked on
   white are unreadable here, and simply darkening the chip is not enough —
   the *text* is what has to clear 4.5:1. */
const STYLES = {
  saved: 'bg-slate-400/12 text-slate-300 ring-slate-400/25',
  applied: 'bg-sky-400/12 text-sky-300 ring-sky-400/30',
  screening: 'bg-cyan-400/12 text-cyan-300 ring-cyan-400/30',
  interviewing: 'bg-teal-400/14 text-teal-300 ring-teal-400/30',
  offer: 'bg-signal/14 text-signal ring-signal/35',
  accepted: 'bg-positive/14 text-positive ring-positive/35',
  rejected: 'bg-danger/12 text-danger ring-danger/30',
  withdrawn: 'bg-slate-400/10 text-slate-400 ring-slate-400/20',
  ghosted: 'bg-orange-400/12 text-orange-300 ring-orange-400/28',
}

export function StatusBadge({ status, className = '' }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
        STYLES[status] ?? STYLES.saved
      } ${className}`}
    >
      {STATUS_LABELS[status] ?? status}
    </span>
  )
}
