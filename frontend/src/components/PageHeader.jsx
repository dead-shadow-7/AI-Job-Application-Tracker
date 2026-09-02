import { ArrowLeft } from 'lucide-react'
import { Link } from 'react-router-dom'

/**
 * The top of every route.
 *
 * It exists so the four pages agree on where the title sits, how big it is and
 * where the buttons go. When each page built its own header they drifted by a
 * few pixels and half a font weight, and the app read as four apps.
 *
 * The display face appears here and nowhere else. Used on every heading it
 * stops being a voice and becomes the body font.
 */
export function PageHeader({ title, subtitle, actions, back }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4 pt-1">
      <div className="min-w-0">
        {back && (
          <Link
            to={back.to}
            className="mb-2 inline-flex items-center gap-1.5 text-xs text-ink-faint transition hover:text-accent"
          >
            <ArrowLeft size={13} aria-hidden="true" />
            {back.label}
          </Link>
        )}
        <h1 className="font-display text-2xl leading-tight font-semibold tracking-tight text-balance">
          {title}
        </h1>
        {subtitle && <p className="mt-1 max-w-xl text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}
