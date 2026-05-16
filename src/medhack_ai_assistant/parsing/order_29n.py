from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from medhack_ai_assistant.config import (
    ORDER_29N_CHUNKS_PATH,
    ORDER_29N_PDF_PATH,
    ORDER_29N_TEXT_PATH,
)
from medhack_ai_assistant.domain.models import RAGContext


DEFAULT_MAX_CHARS = 2800
DEFAULT_OVERLAP_CHARS = 250
MIN_CHUNK_CHARS_WITHOUT_FACTORS = 180

FACTOR_CODE_PATTERN = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\s*\.\s*\d{1,3}){0,3})(?![\d.])"
)
CODE_FRAGMENT_PATTERN = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
BROKEN_CODE_NEWLINE_PATTERN = re.compile(r"(?<=\d)\s*\.\s*\n+\s*(?=\d)")
HYPHENATED_LINEBREAK_PATTERN = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])-\s*\n\s*(?=[A-Za-zА-Яа-яЁё])")
PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n+")
SINGLE_LINEBREAK_PATTERN = re.compile(r"(?<!\n)\n(?!\n)")
SPACES_PATTERN = re.compile(r"[ \t\f\v]+")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

HEADING_KEYWORDS = (
    "приложение",
    "порядок",
    "перечень",
    "вредные",
    "опасные",
    "противопоказания",
    "медицинские противопоказания",
    "медицинский осмотр",
    "периодичность",
    "лабораторные",
    "исследования",
)
RANKING_KEYWORDS = (
    "противопоказ",
    "осмотр",
    "исследован",
    "врач",
    "заболеван",
    "нарушен",
    "стойк",
    "хроническ",
)


class Order29NParserError(RuntimeError):
    """Raised when Order 29n parsing cannot continue safely."""


@dataclass(frozen=True)
class OrderPage:
    page_number: int
    text: str
    char_count: int


@dataclass(frozen=True)
class OrderChunk:
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    factor_codes: tuple[str, ...]
    headings: tuple[str, ...]
    score: float = 0.0


@dataclass(frozen=True)
class _Paragraph:
    page_number: int
    text: str
    factor_codes: tuple[str, ...]
    headings: tuple[str, ...]
    is_anchor: bool


def extract_order_pages(
    pdf_path: Path = ORDER_29N_PDF_PATH,
    output_path: Path = ORDER_29N_TEXT_PATH,
    *,
    force: bool = False,
) -> list[OrderPage]:
    """Extract normalized page text from the Order 29n PDF and cache it as JSONL."""
    if output_path.exists() and not force:
        return _read_pages_jsonl(output_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Order 29n PDF was not found: {pdf_path}. "
            "Expected location: data/reference/order_29n/raw/29N.pdf"
        )

    fitz = _import_fitz()
    pages: list[OrderPage] = []

    try:
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                raw_text = page.get_text("text", sort=True)
                text = clean_order_text(raw_text)
                pages.append(
                    OrderPage(
                        page_number=page_index,
                        text=text,
                        char_count=len(text),
                    )
                )
    except Exception as exc:
        raise Order29NParserError(f"Failed to parse Order 29n PDF: {pdf_path}") from exc

    if not any(page.text for page in pages):
        raise Order29NParserError(
            "Order 29n PDF text layer is empty. The file may be a scan and need OCR."
        )

    _write_jsonl(output_path, pages)
    return pages


def build_order_chunks(
    text_path: Path = ORDER_29N_TEXT_PATH,
    output_path: Path = ORDER_29N_CHUNKS_PATH,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    force: bool = False,
) -> list[OrderChunk]:
    """Build bounded search chunks from extracted Order 29n pages."""
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    if output_path.exists() and not force:
        chunks = _read_chunks_jsonl(output_path)
        _validate_chunk_sizes(chunks, max_chars)
        return chunks

    pages = _read_pages_jsonl(text_path)
    paragraphs = _build_paragraphs(pages)
    chunks = _chunk_paragraphs(paragraphs, max_chars=max_chars, overlap_chars=overlap_chars)

    if not chunks:
        raise Order29NParserError(
            "No searchable chunks were built from Order 29n text. "
            "Check PDF extraction quality or rebuild text cache with force=True."
        )

    _write_jsonl(output_path, chunks)
    return chunks


def search_order_chunks(
    factor_codes: Sequence[str],
    chunks_path: Path = ORDER_29N_CHUNKS_PATH,
    *,
    limit_per_factor: int = 3,
) -> list[OrderChunk]:
    """Return ranked Order 29n chunks relevant to the given patient factor codes."""
    normalized_codes = tuple(
        dict.fromkeys(
            code
            for code in (normalize_factor_code(value) for value in factor_codes)
            if code
        )
    )
    if not normalized_codes:
        return []
    if limit_per_factor < 1:
        raise ValueError("limit_per_factor must be positive")

    chunks = _read_chunks_jsonl(chunks_path)
    selected_by_factor: dict[str, list[OrderChunk]] = defaultdict(list)

    for chunk in chunks:
        for factor_code in normalized_codes:
            score = _score_chunk_for_factor(chunk, factor_code)
            if score > 0:
                selected_by_factor[factor_code].append(_replace_chunk_score(chunk, score))

    deduped: dict[str, OrderChunk] = {}
    for factor_code in normalized_codes:
        ranked = sorted(
            selected_by_factor[factor_code],
            key=lambda chunk: (chunk.score, -len(chunk.text)),
            reverse=True,
        )
        for chunk in ranked[:limit_per_factor]:
            previous = deduped.get(chunk.chunk_id)
            if previous is None or chunk.score > previous.score:
                deduped[chunk.chunk_id] = chunk

    return sorted(deduped.values(), key=lambda chunk: chunk.score, reverse=True)


def ensure_order_search_index(*, force: bool = False) -> list[OrderChunk]:
    """Extract pages and build chunks if needed, returning the ready search corpus."""
    extract_order_pages(force=force)
    return build_order_chunks(force=force)


def build_order_context_for_prompt(
    factor_codes: Sequence[str],
    *,
    max_chars: int = 9000,
    limit_per_factor: int = 2,
) -> RAGContext:
    """Build a compact legal context block for the LLM prompt."""
    chunks = search_order_chunks(factor_codes, limit_per_factor=limit_per_factor)
    selected: list[dict[str, Any]] = []
    text_parts: list[str] = []
    total_chars = 0

    for chunk in chunks:
        header = (
            f"[{chunk.chunk_id}; pages {chunk.page_start}-{chunk.page_end}; "
            f"factors: {', '.join(chunk.factor_codes)}]"
        )
        block = f"{header}\n{chunk.text}"
        separator = "\n\n"
        next_size = len(block) + (len(separator) if text_parts else 0)
        if text_parts and total_chars + next_size > max_chars:
            break
        if not text_parts and len(block) > max_chars:
            block = block[:max_chars].rstrip()
            next_size = len(block)

        text_parts.append(block)
        total_chars += next_size
        selected.append(
            {
                "chunk_id": chunk.chunk_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "factor_codes": chunk.factor_codes,
                "score": chunk.score,
            }
        )

    text = "\n\n".join(text_parts)
    return RAGContext(
        chunks=tuple(selected),
        text=text,
        total_chars=len(text),
    )


def clean_order_text(text: str) -> str:
    """Normalize PDF text for stable factor-code search."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = CONTROL_CHARS_PATTERN.sub(" ", cleaned)
    cleaned = HYPHENATED_LINEBREAK_PATTERN.sub("", cleaned)
    cleaned = BROKEN_CODE_NEWLINE_PATTERN.sub(".", cleaned)
    cleaned = CODE_FRAGMENT_PATTERN.sub(".", cleaned)
    cleaned = PARAGRAPH_BREAK_PATTERN.sub("\n\n", cleaned)
    cleaned = SINGLE_LINEBREAK_PATTERN.sub(" ", cleaned)
    cleaned = SPACES_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r" *\n\n *", "\n\n", cleaned)
    cleaned = _normalize_factor_codes_in_text(cleaned)
    return cleaned.strip()


def normalize_factor_code(value: Any) -> str:
    """Return a canonical factor code like 18.1 or 1.36.2, or an empty string."""
    if value is None:
        return ""
    code = str(value).strip()
    if not code:
        return ""
    code = code.replace(",", ".")
    code = re.sub(r"\s+", "", code)
    code = re.sub(r"\.+", ".", code).strip(".")
    if not re.fullmatch(r"\d{1,2}(?:\.\d{1,3}){0,3}", code):
        return ""
    return code


def extract_factor_codes(text: str) -> tuple[str, ...]:
    codes = [
        normalize_factor_code(match.group(1))
        for match in FACTOR_CODE_PATTERN.finditer(text)
    ]
    return tuple(dict.fromkeys(code for code in codes if code))


def _import_fitz() -> Any:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise Order29NParserError(
            "PyMuPDF is not installed. Add it with `uv add pymupdf` "
            "and then sync the environment."
        ) from exc
    return fitz


def _normalize_factor_codes_in_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        normalized = normalize_factor_code(match.group(1))
        return normalized or match.group(0)

    return FACTOR_CODE_PATTERN.sub(replace, text)


def _build_paragraphs(pages: Sequence[OrderPage]) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []

    for page in pages:
        for raw_paragraph in _split_page_into_paragraphs(page.text):
            text = raw_paragraph.strip()
            if not text:
                continue
            factor_codes = extract_factor_codes(text)
            headings = _extract_headings(text)
            is_anchor = bool(factor_codes or headings or _looks_like_heading(text))
            paragraphs.append(
                _Paragraph(
                    page_number=page.page_number,
                    text=text,
                    factor_codes=factor_codes,
                    headings=headings,
                    is_anchor=is_anchor,
                )
            )

    return paragraphs


def _split_page_into_paragraphs(text: str) -> list[str]:
    if not text:
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    sentences = re.split(r"(?<=[.!?;:])\s+(?=[А-ЯA-ZЁ0-9])", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extract_headings(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    headings: list[str] = []

    for keyword in HEADING_KEYWORDS:
        if keyword in lowered:
            headings.append(keyword)

    if _looks_like_heading(text):
        headings.append(_compact_heading(text))

    return tuple(dict.fromkeys(headings))


def _looks_like_heading(text: str) -> bool:
    compact = text.strip()
    if not compact or len(compact) > 180:
        return False

    letters = [char for char in compact if char.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", compact)
    has_heading_keyword = any(keyword in compact.lower() for keyword in HEADING_KEYWORDS)
    return has_heading_keyword or (uppercase_ratio >= 0.55 and len(words) <= 12)


def _compact_heading(text: str) -> str:
    heading = re.sub(r"\s+", " ", text).strip()
    return heading[:120]


def _chunk_paragraphs(
    paragraphs: Sequence[_Paragraph],
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[OrderChunk]:
    chunks: list[OrderChunk] = []
    used_windows: set[tuple[int, int]] = set()

    anchor_indexes = [
        index for index, paragraph in enumerate(paragraphs)
        if paragraph.is_anchor
    ]
    if not anchor_indexes:
        anchor_indexes = list(range(0, len(paragraphs), 4))

    for anchor_index in anchor_indexes:
        start, end = _window_bounds(paragraphs, anchor_index, max_chars)
        if (start, end) in used_windows:
            continue
        used_windows.add((start, end))

        window_paragraphs = paragraphs[start:end]
        for text_part, part_start_page, part_end_page in _bounded_text_parts(
            window_paragraphs,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            factor_codes = extract_factor_codes(text_part)
            headings = _extract_headings(text_part)
            if not _should_keep_chunk(text_part, factor_codes, headings):
                continue

            chunks.append(
                OrderChunk(
                    chunk_id=f"order29n-{len(chunks) + 1:05d}",
                    page_start=part_start_page,
                    page_end=part_end_page,
                    text=text_part,
                    factor_codes=factor_codes,
                    headings=headings,
                )
            )

    return _dedupe_chunks(chunks)


def _window_bounds(
    paragraphs: Sequence[_Paragraph],
    anchor_index: int,
    max_chars: int,
) -> tuple[int, int]:
    start = anchor_index
    end = anchor_index + 1
    total = len(paragraphs[anchor_index].text)
    left = anchor_index - 1
    right = anchor_index + 1

    while total < max_chars and (left >= 0 or right < len(paragraphs)):
        if left >= 0:
            candidate_len = len(paragraphs[left].text) + 2
            if total + candidate_len <= max_chars:
                start = left
                total += candidate_len
            left -= 1

        if total >= max_chars:
            break

        if right < len(paragraphs):
            candidate_len = len(paragraphs[right].text) + 2
            if total + candidate_len <= max_chars:
                end = right + 1
                total += candidate_len
            right += 1

    return start, end


def _bounded_text_parts(
    paragraphs: Sequence[_Paragraph],
    *,
    max_chars: int,
    overlap_chars: int,
) -> Iterable[tuple[str, int, int]]:
    text = "\n\n".join(paragraph.text for paragraph in paragraphs).strip()
    if not text:
        return

    page_start = min(paragraph.page_number for paragraph in paragraphs)
    page_end = max(paragraph.page_number for paragraph in paragraphs)

    if len(text) <= max_chars:
        yield text, page_start, page_end
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        part = text[start:end].strip()
        if part:
            yield part, page_start, page_end
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)


def _should_keep_chunk(
    text: str,
    factor_codes: Sequence[str],
    headings: Sequence[str],
) -> bool:
    if factor_codes or headings:
        return True
    return len(text) >= MIN_CHUNK_CHARS_WITHOUT_FACTORS


def _dedupe_chunks(chunks: Sequence[OrderChunk]) -> list[OrderChunk]:
    seen: dict[str, OrderChunk] = {}
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.text[:500]).strip().lower()
        if key not in seen:
            seen[key] = chunk

    deduped: list[OrderChunk] = []
    for index, chunk in enumerate(seen.values(), start=1):
        deduped.append(
            OrderChunk(
                chunk_id=f"order29n-{index:05d}",
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                factor_codes=chunk.factor_codes,
                headings=chunk.headings,
                score=chunk.score,
            )
        )
    return deduped


def _score_chunk_for_factor(chunk: OrderChunk, factor_code: str) -> float:
    pattern = _exact_factor_pattern(factor_code)
    matches = len(pattern.findall(chunk.text))
    has_metadata_match = factor_code in chunk.factor_codes
    if matches == 0 and not has_metadata_match:
        return 0.0

    lowered = chunk.text.lower()
    keyword_hits = sum(1 for keyword in RANKING_KEYWORDS if keyword in lowered)
    density = matches / max(len(chunk.text) / 1000, 1)
    heading_bonus = 1.0 if chunk.headings else 0.0
    metadata_bonus = 2.0 if has_metadata_match else 0.0
    return 10.0 + metadata_bonus + matches * 3.0 + density + keyword_hits * 0.5 + heading_bonus


def _exact_factor_pattern(factor_code: str) -> re.Pattern[str]:
    escaped = re.escape(factor_code).replace(r"\.", r"\s*\.\s*")
    return re.compile(rf"(?<![\d.]){escaped}(?![\d.])")


def _replace_chunk_score(chunk: OrderChunk, score: float) -> OrderChunk:
    return OrderChunk(
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        factor_codes=chunk.factor_codes,
        headings=chunk.headings,
        score=score,
    )


def _validate_chunk_sizes(chunks: Sequence[OrderChunk], max_chars: int) -> None:
    oversized = [chunk.chunk_id for chunk in chunks if len(chunk.text) > max_chars]
    if oversized:
        raise Order29NParserError(
            f"Chunk cache contains chunks larger than {max_chars} characters: "
            f"{', '.join(oversized[:5])}. Rebuild with force=True."
        )


def _read_pages_jsonl(path: Path) -> list[OrderPage]:
    records = _read_jsonl(path)
    pages = [
        OrderPage(
            page_number=int(record["page_number"]),
            text=str(record["text"]),
            char_count=int(record.get("char_count", len(str(record["text"])))),
        )
        for record in records
    ]
    if not pages:
        raise Order29NParserError(f"Order 29n page cache is empty: {path}")
    return pages


def _read_chunks_jsonl(path: Path) -> list[OrderChunk]:
    records = _read_jsonl(path)
    chunks = [
        OrderChunk(
            chunk_id=str(record["chunk_id"]),
            page_start=int(record["page_start"]),
            page_end=int(record["page_end"]),
            text=str(record["text"]),
            factor_codes=tuple(record.get("factor_codes", ())),
            headings=tuple(record.get("headings", ())),
            score=float(record.get("score", 0.0)),
        )
        for record in records
    ]
    if not chunks:
        raise Order29NParserError(f"Order 29n chunk cache is empty: {path}")
    return chunks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL cache was not found: {path}")

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError(f"record is not an object at line {line_number}")
                records.append(record)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise Order29NParserError(
            f"JSONL cache is corrupted: {path}. Rebuild it with force=True."
        ) from exc

    return records


def _write_jsonl(path: Path, records: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            payload = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
