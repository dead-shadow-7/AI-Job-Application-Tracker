"""Turn an uploaded resume into embeddable chunks.

Section-aware rather than fixed-size. A resume is not prose: splitting every
600 characters routinely cuts a bullet in half, so the fragment that should have
evidenced "5 years of Kafka" becomes two fragments that evidence neither. Chunks
follow the document's own structure instead, and stay whole.
"""

import io
import re
from dataclasses import dataclass

from app.core.exceptions import InvalidOperationError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MIN_TEXT_LENGTH = 200

# Headings a resume actually uses, mapped to a canonical section. The label is
# carried onto each chunk so retrieval can weight experience above education
# without re-reading the text.
SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "experience",
        re.compile(r"^\s*(work\s+)?(experience|employment|professional\s+experience)\b", re.I),
    ),
    ("projects", re.compile(r"^\s*(projects?|personal\s+projects?|selected\s+work)\b", re.I)),
    (
        "skills",
        re.compile(r"^\s*(technical\s+)?(skills|technologies|tech\s+stack|competencies)\b", re.I),
    ),
    ("education", re.compile(r"^\s*(education|academics?|qualifications?)\b", re.I)),
    ("certifications", re.compile(r"^\s*(certifications?|licenses?|courses?)\b", re.I)),
    ("summary", re.compile(r"^\s*(summary|profile|objective|about\s+me)\b", re.I)),
    ("publications", re.compile(r"^\s*(publications?|papers?|research)\b", re.I)),
    ("awards", re.compile(r"^\s*(awards?|achievements?|honou?rs?)\b", re.I)),
]

# A heading is short and title-like. Length alone is unreliable, so it must also
# match a known pattern before a line is treated as one.
MAX_HEADING_LENGTH = 60

# Bullets shorter than this ("Python", "•") carry no retrievable meaning on
# their own and are merged into the previous chunk rather than embedded.
MIN_CHUNK_LENGTH = 40
MAX_CHUNK_LENGTH = 1200


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    section: str | None
    content: str


def extract_text(filename: str, data: bytes) -> str:
    """Pull plain text out of a PDF or DOCX upload."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidOperationError(
            f"That file is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB. A resume should be far smaller — "
            "check it is not a scanned image."
        )

    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        text = _extract_pdf(data)
    elif lowered.endswith((".docx", ".doc")):
        text = _extract_docx(data)
    elif lowered.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise InvalidOperationError("Upload a PDF, DOCX, or plain text file.")

    text = _normalize(text)

    if len(text) < MIN_TEXT_LENGTH:
        raise InvalidOperationError(
            "Almost no text could be read from that file. If it is a scanned "
            "PDF the words are an image — export a text-based PDF from your "
            "editor, or paste the text instead."
        )
    return text


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf raises a wide variety on malformed files
        raise InvalidOperationError(f"That PDF could not be read: {exc}") from exc


def _extract_docx(data: bytes) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise InvalidOperationError(f"That document could not be read: {exc}") from exc

    parts = [p.text for p in document.paragraphs]
    # Resumes frequently lay out experience in invisible tables, and those cells
    # hold the actual content. Skipping them loses most of the document.
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise the several bullet glyphs PDFs emit to one marker, so the
    # bullet-detection below does not depend on which tool produced the file.
    text = re.sub(r"^[\s]*[•▪◦‣∙·–—*]\s*", "- ", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_LENGTH:
        return None
    for name, pattern in SECTION_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def chunk_resume(text: str) -> list[Chunk]:
    """Split into retrievable passages, tagged by section.

    Each bullet or paragraph becomes its own chunk, because the unit that
    answers "does this candidate have Kafka experience?" is a single
    accomplishment line, not a page. Very short lines merge into the previous
    chunk, and over-long paragraphs are split on sentence boundaries.
    """
    chunks: list[Chunk] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        content = " ".join(part.strip() for part in buffer if part.strip()).strip()
        buffer.clear()
        if not content:
            return

        for piece in _split_long(content):
            if (
                chunks
                and len(piece) < MIN_CHUNK_LENGTH
                and chunks[-1].section == section
                and len(chunks[-1].content) + len(piece) < MAX_CHUNK_LENGTH
            ):
                merged = f"{chunks[-1].content} {piece}"
                chunks[-1] = Chunk(chunks[-1].ordinal, section, merged)
            else:
                chunks.append(Chunk(len(chunks), section, piece))

    for line in text.split("\n"):
        heading = detect_section(line)
        if heading:
            flush()
            section = heading
            continue

        if not line.strip():
            flush()
            continue

        # A new bullet ends the previous one; a continuation line extends it.
        if line.lstrip().startswith("- ") and buffer:
            flush()

        buffer.append(line)

    flush()
    return chunks


def _split_long(content: str) -> list[str]:
    if len(content) <= MAX_CHUNK_LENGTH:
        return [content]

    sentences = re.split(r"(?<=[.!?])\s+", content)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > MAX_CHUNK_LENGTH and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


YEARS_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:professional\s+)?experience", re.I),
    re.compile(r"(?:experience|exp)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*\+?\s*years?", re.I),
]


def guess_years_experience(text: str) -> float | None:
    """Best-effort read of stated years of experience.

    Only ever taken from an explicit statement in the resume. Inferring it from
    employment dates is unreliable — overlapping roles, internships, gaps — and
    a wrong number here would skew every match score without being visible.
    """
    for pattern in YEARS_PATTERNS:
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            if 0 < value <= 50:
                return value
    return None
