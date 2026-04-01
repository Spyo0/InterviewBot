"""Analyse et parsing des PDF : extraction de texte, découpage par chapitre."""

import os
import re
import tempfile
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_PDFS = int(os.getenv("MAX_PDFS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "botentretien_pdf_previews")


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


def extract_text_with_pages(pdf_path: str) -> list[dict]:
    """Extrait le texte page par page avec métadonnées."""
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


def detect_chapters(pages: list[dict]) -> list[dict]:
    """Détecte les chapitres via des patterns courants et regroupe le texte."""
    chapter_pattern = re.compile(
        r"^(chapter|chapitre|part|partie)\s+(\d+|[ivxlc]+)",
        re.IGNORECASE | re.MULTILINE,
    )

    chapters = []
    current_chapter = {
        "title": "Introduction",
        "text": "",
        "start_page": 1,
        "source": "",
        "pages": [],
    }

    for page_data in pages:
        text = page_data["text"]
        source = page_data["source"]
        current_chapter["source"] = source

        match = chapter_pattern.search(text)
        if match:
            # Sauvegarder le chapitre précédent
            if current_chapter["text"].strip():
                chapters.append(current_chapter.copy())

            # Extraire le titre du chapitre (première ligne contenant le match)
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            title = text[line_start:line_end].strip() if line_end != -1 else text[line_start:].strip()

            current_chapter = {
                "title": title[:100],
                "text": text,
                "start_page": page_data["page"],
                "source": source,
                "pages": [page_data],
            }
        else:
            current_chapter["text"] += "\n" + text
            current_chapter["pages"].append(page_data)

    if current_chapter["text"].strip():
        chapters.append(current_chapter)

    return chapters


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
    chapters = detect_chapters(pages)

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
                        "chunk_index": i,
                    },
                })
    return documents
