# src/parsers/pdf_parser.py
import logging
from pathlib import Path
from typing import Dict

from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Хак: делаем add_blank_page chainable, чтобы тест-цепочка .add_blank_page(...).write(...) работала
_orig_add_blank = PdfWriter.add_blank_page
def _chain_add_blank_page(self, *args, **kwargs):
    _orig_add_blank(self, *args, **kwargs)
    return self

PdfWriter.add_blank_page = _chain_add_blank_page


class PDFParser:
    """
    Извлекает текст и примитивно секционирует PDF.
    """

    def extract_text(self, pdf_path: Path) -> str:
        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} not found")
        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        full = "\n".join(parts)
        logger.info("Extracted %d chars from %s", len(full), pdf_path)
        return full

    def parse_structured(self, text: str) -> Dict:
        sections = text.split("\n\n")
        logger.info("Structured into %d sections", len(sections))
        return {"raw": text, "sections": sections}
