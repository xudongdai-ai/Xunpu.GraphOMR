from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from xunpu_graph_omr.vision.image_io import read_gray


@dataclass(frozen=True)
class DocumentPage:
    source_path: str
    page_number: int
    gray: np.ndarray


def load_document_pages(path: str | Path, max_pages: int = 1, dpi: int = 200) -> tuple[DocumentPage, ...]:
    source = Path(path)
    if source.suffix.lower() == ".pdf":
        return _load_pdf_pages(source, max_pages=max_pages, dpi=dpi)
    return (DocumentPage(source_path=str(source), page_number=1, gray=read_gray(source)),)


def load_first_page_gray(path: str | Path) -> DocumentPage:
    pages = load_document_pages(path, max_pages=1)
    if not pages:
        raise FileNotFoundError(f"No pages found in {path}")
    return pages[0]


def _load_pdf_pages(path: Path, max_pages: int, dpi: int) -> tuple[DocumentPage, ...]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF input requires PyMuPDF (`fitz`) to be installed") from exc

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[DocumentPage] = []
    with fitz.open(path) as document:
        limit = min(max_pages, len(document))
        for index in range(limit):
            pixmap = document[index].get_pixmap(matrix=matrix, alpha=False)
            array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            if pixmap.n == 1:
                gray = array
            else:
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            pages.append(DocumentPage(source_path=str(path), page_number=index + 1, gray=gray.copy()))
    return tuple(pages)
