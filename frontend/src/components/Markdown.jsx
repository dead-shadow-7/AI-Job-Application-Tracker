import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'

/**
 * The assistant's prose, rendered as what it already was.
 *
 * The model writes markdown whether or not anyone asks it to — bold labels,
 * bullet lists, the occasional table — and rendering that as plain text put
 * literal `**` around half the nouns in a comparison. Nothing here changes what
 * the model produces; it stops the client from mangling it.
 *
 * No `rehype-raw`, deliberately. Without it react-markdown drops HTML in the
 * source rather than mounting it, so a posting pasted into the conversation and
 * quoted back cannot smuggle markup onto the page. There is no case where the
 * assistant needs to emit HTML, and the plugin that would allow it is the one
 * that would make that a question worth asking.
 *
 * `remark-breaks` is here to preserve what `whitespace-pre-wrap` used to do:
 * markdown folds a single newline into a space, so without it a list of short
 * lines the model wrote one per row arrives as one run-on paragraph.
 *
 * Element styles are spelled out rather than pulled from a typography plugin.
 * These render inside a chat bubble, where the default prose rhythm — margins
 * sized for an article — is much too loose.
 */
/** An element, styled. `node` is dropped: react-markdown passes the parsed
 *  syntax node to every component and React would render it as a DOM attribute. */
const styled =
  (Tag, className, fixed) =>
  ({ node: _node, ...props }) => <Tag className={className} {...fixed} {...props} />

const BLOCK = 'mb-2 last:mb-0'

const COMPONENTS = {
  p: styled('p', BLOCK),
  ul: styled('ul', `${BLOCK} list-disc space-y-0.5 pl-4`),
  ol: styled('ol', `${BLOCK} list-decimal space-y-0.5 pl-4`),
  li: styled('li', 'pl-0.5'),
  strong: styled('strong', 'font-semibold'),
  em: styled('em', 'italic'),
  h1: styled('h1', 'mt-3 mb-1 font-semibold first:mt-0'),
  h2: styled('h2', 'mt-3 mb-1 font-semibold first:mt-0'),
  h3: styled('h3', 'mt-3 mb-1 font-semibold first:mt-0'),
  // A job URL the assistant quotes back should not navigate the drawer away
  // from the conversation that produced it.
  a: styled('a', 'underline underline-offset-2', {
    target: '_blank',
    rel: 'noopener noreferrer',
  }),
  code: styled('code', 'rounded bg-black/8 px-1 py-0.5 font-mono text-[0.9em]'),
  // The arbitrary variant undoes the inline treatment above for the `code` a
  // fence puts inside a `pre` — otherwise a block of code renders as a padded
  // pill inside a padded box.
  pre: styled(
    'pre',
    `${BLOCK} overflow-x-auto rounded-lg bg-black/8 p-2 text-xs [&_code]:bg-transparent [&_code]:p-0`,
  ),
  blockquote: styled('blockquote', `${BLOCK} border-l-2 border-border-subtle pl-2`),
  hr: styled('hr', 'my-2 border-border-subtle'),
  th: styled('th', 'border-b border-border-subtle py-1 pr-3 font-semibold'),
  td: styled('td', 'border-b border-border-subtle py-1 pr-3 align-top'),
  // Comparisons come back as tables, and a wide one has to scroll inside its
  // own box rather than widening the drawer past the edge of the screen.
  table: ({ node: _node, ...props }) => (
    <div className={`${BLOCK} overflow-x-auto`}>
      <table className="w-full border-collapse text-left" {...props} />
    </div>
  ),
}

export function Markdown({ children }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={COMPONENTS}>
      {children}
    </ReactMarkdown>
  )
}
