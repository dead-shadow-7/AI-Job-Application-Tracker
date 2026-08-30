import { STATUS_LABELS } from '@/lib/format'

/* Hue carries meaning: grey = dormant, blue = in motion, amber = awaiting you,
   green = won, red = lost. Colour is never the only signal — the label is
   always present — so this stays readable without colour perception. */
const STYLES = {
  saved: 'bg-slate-100 text-slate-700 ring-slate-200',
  applied: 'bg-blue-50 text-blue-700 ring-blue-200',
  screening: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
  interviewing: 'bg-violet-50 text-violet-700 ring-violet-200',
  offer: 'bg-amber-50 text-amber-800 ring-amber-200',
  accepted: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  rejected: 'bg-rose-50 text-rose-700 ring-rose-200',
  withdrawn: 'bg-slate-100 text-slate-600 ring-slate-200',
  ghosted: 'bg-orange-50 text-orange-700 ring-orange-200',
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
