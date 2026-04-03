"""Analyse et parsing des PDF : extraction de texte, découpage par chapitre."""

import os
import re
import tempfile
from typing import Optional
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_PDFS = int(os.getenv("MAX_PDFS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "botentretien_pdf_previews")
TOC_MAX_LEVEL = int(os.getenv("TOC_MAX_LEVEL", "2"))
EXCLUDED_TOC_TITLES = (
    "cover",
    "title page",
    "table of contents",
    "contents",
    "copyright",
    "dedication",
    "preface",
    "business snapshots",
    "index",
    "bibliography",
    "references",
    "acknowledg",
    "about the author",
    "further reading",
    "appendix",
)
CHAPTER_TITLE_PATTERN = re.compile(
    r"^((chapter|chapitre|part|partie)\s+(\d+|[ivxlc]+)\b|(\d+|[ivxlc]+)\b(?!\.))",
    re.IGNORECASE,
)
SECTION_TITLE_PATTERN = re.compile(r"^\d+\.\d+")


def list_pdfs() -> list[str]:
    """Liste les PDF disponibles dans data/."""
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    return sorted(pdfs)[:MAX_PDFS]


def get_pdf_path(pdf_name: str) -> str:
    """Retourne le chemin absolu d'un PDF stocké dans data/."""
    return os.path.join(DATA_DIR, pdf_name)


def _page_has_visuals(page: fitz.Page) -> bool:
    """Détecte si une page contient un support visuel exploitable."""
    text_dict = page.get_text("dict")
    has_image_blocks = any(block.get("type") == 1 for block in text_dict.get("blocks", []))
    has_embedded_images = bool(page.get_images(full=True))
    has_drawings = bool(page.get_drawings())
    return has_image_blocks or has_embedded_images or has_drawings


def get_page_preview(
    pdf_name: str,
    page_number: int,
    only_if_visual: bool = True,
    zoom: float = 2.0,
) -> dict | None:
    """Génère une preview PNG d'une page PDF et indique si la page contient un visuel."""
    pdf_path = get_pdf_path(pdf_name)
    if not os.path.exists(pdf_path) or page_number < 1:
        return None

    doc = fitz.open(pdf_path)
    try:
        if page_number > len(doc):
            return None

        page = doc[page_number - 1]
        has_visuals = _page_has_visuals(page)
        if only_if_visual and not has_visuals:
            return None

        os.makedirs(PREVIEW_DIR, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(pdf_name))
        preview_path = os.path.join(PREVIEW_DIR, f"{safe_name}_p{page_number}.png")

        if not os.path.exists(preview_path):
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(preview_path)

        return {
            "path": preview_path,
            "page": page_number,
            "has_visuals": has_visuals,
        }
    finally:
        doc.close()


def _extract_markdown_with_pymupdf4llm(pdf_path: str) -> list[dict] | None:
    """Extrait le texte en Markdown (avec formules LaTeX préservées) via pymupdf4llm.

    Retourne None si la librairie n'est pas disponible ou si l'extraction échoue.
    """
    try:
        import pymupdf4llm
    except ImportError:
        return None

    try:
        pages_md = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    except Exception:
        return None

    source = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for chunk in pages_md:
        page_number = chunk.get("metadata", {}).get("page", 0) + 1
        text = chunk.get("text", "").strip()
        if not text:
            continue
        has_visuals = False
        if 0 < page_number <= len(doc):
            has_visuals = _page_has_visuals(doc[page_number - 1])
        pages.append({
            "text": text,
            "page": page_number,
            "source": source,
            "has_visuals": has_visuals,
        })
    doc.close()
    return pages or None


def extract_text_with_pages(pdf_path: str) -> list[dict]:
    """Extrait le texte page par page avec métadonnées.

    Tente d'abord pymupdf4llm (préserve les formules LaTeX),
    puis retombe sur PyMuPDF brut si indisponible.
    """
    md_pages = _extract_markdown_with_pymupdf4llm(pdf_path)
    if md_pages is not None:
        return md_pages

    # Fallback : extraction texte brut via PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": i + 1,
                "source": os.path.basename(pdf_path),
                "has_visuals": _page_has_visuals(page),
            })
    doc.close()
    return pages


def _sanitize_title(title: str, fallback: str = "Introduction") -> str:
    """Nettoie un titre de chapitre pour produire une référence stable."""
    cleaned = re.sub(r"\s+", " ", title).strip(" -:\t\r\n")
    cleaned = cleaned.strip('"')
    return cleaned[:120] if cleaned else fallback


def _is_excluded_toc_title(title: str) -> bool:
    normalized = title.lower()
    return any(keyword in normalized for keyword in EXCLUDED_TOC_TITLES)


def _extract_toc_entries(pdf_path: str) -> list[dict]:
    """Extrait les entrées de table des matières exploitables depuis le PDF."""
    doc = fitz.open(pdf_path)
    try:
        toc = doc.get_toc(simple=True)
        if not toc:
            return []

        entries = []
        for level, title, page_number in toc:
            if not title or not isinstance(page_number, int):
                continue
            if page_number < 1 or page_number > len(doc):
                continue

            clean_title = _sanitize_title(title)
            if _is_excluded_toc_title(clean_title):
                continue

            entries.append({
                "level": int(level),
                "title": clean_title,
                "start_page": int(page_number),
            })

        if not entries:
            return []

        primary_level = _select_toc_level(entries)
        selected_entries = [entry for entry in entries if entry["level"] == primary_level]

        deduped_entries = []
        seen_boundaries = set()
        for entry in sorted(selected_entries, key=lambda item: (item["start_page"], item["level"], item["title"])):
            dedupe_key = (entry["start_page"], entry["title"].lower())
            if dedupe_key in seen_boundaries:
                continue
            seen_boundaries.add(dedupe_key)
            deduped_entries.append(entry)

        return deduped_entries
    finally:
        doc.close()


def _select_toc_level(entries: list[dict]) -> int:
    """Sélectionne le niveau de TOC le plus proche d'un vrai découpage chapitre."""
    levels = sorted({entry["level"] for entry in entries if entry["level"] <= TOC_MAX_LEVEL})
    if not levels:
        return min(entry["level"] for entry in entries)

    fallback_level: Optional[int] = None
    for level in levels:
        level_entries = [entry for entry in entries if entry["level"] == level]
        if len(level_entries) < 3:
            continue

        if fallback_level is None:
            fallback_level = level

        titles = [entry["title"] for entry in level_entries]
        chapter_like_ratio = sum(
            1 for title in titles if CHAPTER_TITLE_PATTERN.match(title)
        ) / len(titles)
        section_like_ratio = sum(
            1 for title in titles if SECTION_TITLE_PATTERN.match(title)
        ) / len(titles)

        if section_like_ratio < 0.5 and chapter_like_ratio >= 0.25:
            return level

        if section_like_ratio == 0:
            return level

    if fallback_level is not None:
        return fallback_level

    return levels[0]


def _build_chapters_from_boundaries(pages: list[dict], boundaries: list[dict], chapter_origin: str) -> list[dict]:
    """Construit les chapitres à partir d'une liste de pages de début."""
    if not pages:
        return []

    page_lookup = {page_data["page"]: page_data for page_data in pages}
    page_numbers = sorted(page_lookup)
    last_page = page_numbers[-1]
    source = pages[0]["source"]
    chapters = []

    sorted_boundaries = sorted(boundaries, key=lambda item: (item["start_page"], item.get("level", 0)))
    if sorted_boundaries and sorted_boundaries[0]["start_page"] > page_numbers[0]:
        sorted_boundaries = [{
            "title": "Introduction",
            "start_page": page_numbers[0],
            "level": 0,
        }, *sorted_boundaries]

    for index, boundary in enumerate(sorted_boundaries):
        start_page = boundary["start_page"]
        next_start_page = (
            sorted_boundaries[index + 1]["start_page"] - 1
            if index + 1 < len(sorted_boundaries)
            else last_page
        )

        chapter_pages = [
            page_lookup[page_number]
            for page_number in page_numbers
            if start_page <= page_number <= next_start_page
        ]
        if not chapter_pages:
            continue

        chapters.append({
            "title": _sanitize_title(boundary["title"]),
            "text": "\n".join(page_data["text"] for page_data in chapter_pages),
            "start_page": chapter_pages[0]["page"],
            "source": source,
            "pages": chapter_pages,
            "chapter_origin": chapter_origin,
            "chapter_level": boundary.get("level", 0),
        })

    return chapters


def _extract_heading_from_page(text: str) -> str:
    """Extrait un titre plausible depuis le haut de la page pour le fallback regex."""
    lines = [line.strip() for line in text.splitlines()[:12] if line.strip()]
    if not lines:
        return "Introduction"

    chapter_pattern = re.compile(
        r"^(chapter|chapitre|part|partie)\s+(\d+|[ivxlc]+)\b.*$",
        re.IGNORECASE,
    )
    for line in lines:
        if len(line) <= 140 and chapter_pattern.match(line):
            return _sanitize_title(line)

    return "Introduction"


def _detect_chapters_with_patterns(pages: list[dict]) -> list[dict]:
    """Fallback regex quand le PDF ne fournit pas de table des matières exploitable."""
    chapter_pattern = re.compile(
        r"^(chapter|chapitre|part|partie)\s+(\d+|[ivxlc]+)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    )

    chapters = []
    current_chapter = {
        "title": "Introduction",
        "text": "",
        "start_page": 1,
        "source": "",
        "pages": [],
        "chapter_origin": "regex",
        "chapter_level": 0,
    }

    for page_data in pages:
        text = page_data["text"]
        source = page_data["source"]
        current_chapter["source"] = source

        match = chapter_pattern.search(text)
        if match and match.start() < 800:
            if current_chapter["text"].strip():
                chapters.append(current_chapter.copy())

            current_chapter = {
                "title": _extract_heading_from_page(text),
                "text": text,
                "start_page": page_data["page"],
                "source": source,
                "pages": [page_data],
                "chapter_origin": "regex",
                "chapter_level": 0,
            }
        else:
            current_chapter["text"] += ("\n" if current_chapter["text"] else "") + text
            current_chapter["pages"].append(page_data)

    if current_chapter["text"].strip():
        chapters.append(current_chapter)

    return chapters


def detect_chapters(pages: list[dict], pdf_path: str | None = None) -> list[dict]:
    """Détecte les chapitres via la table des matières du PDF, sinon via des patterns."""
    if pdf_path:
        toc_entries = _extract_toc_entries(pdf_path)
        if toc_entries:
            toc_chapters = _build_chapters_from_boundaries(pages, toc_entries, chapter_origin="toc")
            if toc_chapters:
                return toc_chapters

    return _detect_chapters_with_patterns(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe le texte en chunks avec overlap pour le RAG."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_pdf(pdf_name: str) -> list[dict]:
    """Traite un PDF : extraction, découpage, retourne les documents prêts pour l'indexation."""
    pdf_path = get_pdf_path(pdf_name)
    if not os.path.exists(pdf_path):
        return []

    pages = extract_text_with_pages(pdf_path)
    chapters = detect_chapters(pages, pdf_path=pdf_path)

    documents = []
    for chapter in chapters:
        for page_data in chapter["pages"]:
            chunks = chunk_text(page_data["text"])
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": chapter["source"],
                        "chapter": chapter["title"],
                        "start_page": chapter["start_page"],
                        "page": page_data["page"],
                        "has_visuals": page_data["has_visuals"],
                        "chapter_origin": chapter.get("chapter_origin", "regex"),
                        "chapter_level": chapter.get("chapter_level", 0),
                        "chunk_index": i,
                    },
                })
    return documents
