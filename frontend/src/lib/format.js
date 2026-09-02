/** Presentation helpers. Kept in one place so the list, the detail page and the
 *  timeline all render dates and money identically. */

const RELATIVE = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })

export function formatDate(value) {
  if (!value) return '—'
  return new Date(value).toLocaleDateString('en-IN', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  })
}

/** "Aug 18" — the timeline's left rail, where the year is usually noise. */
export function formatDayMonth(value) {
  if (!value) return '—'
  return new Date(value).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })
}

/** "Jan 2021" from the "YYYY-MM" a parsed resume position carries.
 *  Built from the parts rather than parsed as a Date: "2021-01" is read as UTC
 *  midnight, which in any negative-offset timezone renders as December 2020. */
export function formatMonth(value) {
  if (!value) return 'Present'
  const [year, month] = value.split('-')
  const name = new Date(2000, Number(month) - 1, 1).toLocaleDateString('en-IN', { month: 'short' })
  return `${name} ${year}`
}

export function daysBetween(value, from = new Date()) {
  if (!value) return 0
  const ms = from - new Date(value)
  return Math.max(0, Math.floor(ms / 86_400_000))
}

export function relativeDays(value) {
  const days = daysBetween(value)
  if (days === 0) return 'today'
  return RELATIVE.format(-days, 'day')
}

/** Compact money: 18L–24L rather than 1,800,000–2,400,000.
 *  Indian salaries are quoted in lakhs, and full figures make the table
 *  unreadable at a glance. */
export function formatSalary({ salary_min, salary_max, salary_currency, salary_period }) {
  if (salary_min == null && salary_max == null) return null

  const compact = (n) => {
    const value = Number(n)
    if (salary_currency === 'INR') {
      if (value >= 1e7) return `${+(value / 1e7).toFixed(2)}Cr`
      if (value >= 1e5) return `${+(value / 1e5).toFixed(1)}L`
    }
    if (value >= 1000) return `${+(value / 1000).toFixed(0)}k`
    return String(value)
  }

  const symbol = { INR: '₹', USD: '$', EUR: '€', GBP: '£' }[salary_currency] ?? ''
  const range =
    salary_min != null && salary_max != null
      ? `${compact(salary_min)}–${compact(salary_max)}`
      : compact(salary_min ?? salary_max)
  const suffix = salary_period && salary_period !== 'year' ? `/${salary_period}` : ''

  return `${symbol}${range}${suffix}`
}

export const STATUS_LABELS = {
  saved: 'Saved',
  applied: 'Applied',
  screening: 'Screening',
  interviewing: 'Interviewing',
  offer: 'Offer',
  accepted: 'Accepted',
  rejected: 'Rejected',
  withdrawn: 'Withdrawn',
  ghosted: 'Ghosted',
}

export const EVENT_LABELS = {
  saved: 'Saved',
  applied: 'Applied',
  assessment_received: 'Assessment received',
  screening_scheduled: 'Screening scheduled',
  screening_done: 'Screening done',
  interview_scheduled: 'Interview scheduled',
  interview_done: 'Interview done',
  offer_received: 'Offer received',
  accepted: 'Accepted',
  rejected: 'Rejected',
  withdrawn: 'Withdrawn',
  marked_ghosted: 'Marked ghosted',
  recruiter_reply: 'Recruiter replied',
  follow_up_sent: 'Follow-up sent',
  note_added: 'Note added',
}

export const WORK_MODE_LABELS = { onsite: 'On-site', hybrid: 'Hybrid', remote: 'Remote' }

/** Terminal statuses — mirrors TERMINAL_STATUSES in app/domain/enums.py. */
export const TERMINAL_STATUSES = new Set(['accepted', 'rejected', 'withdrawn', 'ghosted'])
